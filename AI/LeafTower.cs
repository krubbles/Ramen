namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

/// <summary>
/// A complete move-scoring tower: its own move vectorizer, trunk adapter, residual blocks
/// and logit head. Several towers can hang off one trunk, which is what the small-versus-large
/// leaf comparison needs — each tower carries its own residual width, so nothing about the
/// embedding or the adapter can be shared.
/// <para>
/// Blocks are either dense <see cref="ResidualBlock"/>s or routed
/// <see cref="DeepSeekMoEResidualBlock"/>s, selected by <see cref="Settings.UseMoE"/>.
/// </para>
/// </summary>
public sealed class LeafTower : Module
{
    public struct Settings
    {
        public int ResidualWidth;
        public int BlockCount;
        /// <summary>Hidden width as a multiple of the residual width, for dense blocks.</summary>
        public float HiddenRatio;
        public bool PerLayerEmbedding;
        public ResidualBlock.ActivationType ActivationType;

        /// <summary>Routed blocks when true, dense blocks when false.</summary>
        public bool UseMoE;
        public int RoutedExpertCount;
        public int ChosenExpertCount;
        public float RoutedExpertHiddenRatio;
        public float SharedExpertHiddenRatio;
        public float RoutingBiasUpdateSpeed;
        public float ExpertCapacityFactor;
    }

    readonly Settings _settings;
    readonly int _trunkWidth;
    readonly Device _device;

    readonly MoveVectorizer _moveVectorizer;
    readonly Linear _trunkProjection;
    readonly LayerNorm _trunkLayerNorm;
    readonly LayerNorm _moveEmbeddingLayerNorm;
    readonly ModuleList<Module<Tensor, Tensor>> _denseBlocks = new();
    readonly ModuleList<DeepSeekMoEResidualBlock> _moeBlocks = new();
    readonly Linear _logitHead;

    public int ResidualWidth => _settings.ResidualWidth;
    public int BlockCount => _settings.BlockCount;
    public bool UsesMoE => _settings.UseMoE;

    public LeafTower(
        Settings settings,
        int trunkWidth,
        int scoreBucketCount,
        bool addWinningMoveEmbedding,
        Device device,
        string name) : base(name)
    {
        _settings = settings;
        _trunkWidth = trunkWidth;
        _device = device ?? CPU;

        int embeddingWidth = settings.PerLayerEmbedding
            ? settings.ResidualWidth * (settings.BlockCount + 1)
            : settings.ResidualWidth;

        _moveVectorizer = new MoveVectorizer(
            moveEmbeddingWidth: embeddingWidth,
            scoreBucketCount: scoreBucketCount,
            addWinningMoveEmbedding: addWinningMoveEmbedding,
            device: _device);
        _trunkProjection = Linear(trunkWidth, embeddingWidth, device: _device);
        _trunkLayerNorm = LayerNorm(embeddingWidth, device: _device);
        _moveEmbeddingLayerNorm = LayerNorm(embeddingWidth, device: _device);

        for (int blockIndex = 0; blockIndex < settings.BlockCount; ++blockIndex)
        {
            if (settings.UseMoE)
            {
                _moeBlocks.append(new DeepSeekMoEResidualBlock(
                    residualWidth: settings.ResidualWidth,
                    routerInputWidth: trunkWidth,
                    routedExpertCount: settings.RoutedExpertCount,
                    chosenExpertCount: settings.ChosenExpertCount,
                    routingBiasUpdateSpeed: settings.RoutingBiasUpdateSpeed,
                    activationType: settings.ActivationType,
                    routedExpertHiddenRatio: settings.RoutedExpertHiddenRatio > 0f ? settings.RoutedExpertHiddenRatio : 0.5f,
                    expertCapacityFactor: settings.ExpertCapacityFactor > 0f ? settings.ExpertCapacityFactor : 4f,
                    sharedExpertHiddenRatio: settings.SharedExpertHiddenRatio > 0f ? settings.SharedExpertHiddenRatio : 1f,
                    device: _device));
            }
            else
            {
                _denseBlocks.append(new ResidualBlock(
                    residualWidth: settings.ResidualWidth,
                    hiddenRatio: settings.HiddenRatio > 0f ? settings.HiddenRatio : 4f,
                    activationType: settings.ActivationType,
                    device: _device));
            }
        }

        _logitHead = Linear(settings.ResidualWidth, 2, device: _device);

        RegisterComponents();
    }

    /// <summary>Set on the owning network before a training forward; routed blocks only.</summary>
    public bool UpdateExpertLoadBalance { get; set; }

    public List<MoERoutingStats> DrainRoutingStats()
    {
        List<MoERoutingStats> stats = [];
        for (int blockIndex = 0; blockIndex < _moeBlocks.Count; ++blockIndex)
            stats.AddRange(_moeBlocks[blockIndex].DrainRoutingStats());
        return stats;
    }

    /// <summary>Logits for every legal move, flattened to [state, move * 2].</summary>
    public Tensor Forward(GameStateTensors gameStateTensors, Tensor trunkOutput)
    {
        using var scope = NewDisposeScope();

        Tensor moveEmbeddings = _moveEmbeddingLayerNorm.forward(_moveVectorizer.forward(gameStateTensors));
        Tensor actionLogits = BuildActionLogits(trunkOutput, moveEmbeddings);
        Tensor policyLogits = actionLogits.view([trunkOutput.size(0), moveEmbeddings.size(1) * 2]);
        Tensor maskedLogits = PolicyLogitMask.Apply(gameStateTensors, policyLogits);

        maskedLogits.MoveToOuterDisposeScope();
        return maskedLogits;
    }

    /// <summary>Logits for a chosen subset of moves, as used by sampled-softmax training.</summary>
    public Tensor ForwardSelected(GameStateTensors gameStateTensors, Tensor trunkOutput, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        Tensor moveEmbeddings = _moveEmbeddingLayerNorm.forward(_moveVectorizer.forward(gameStateTensors, moveIndices));
        Tensor actionLogits = BuildActionLogits(trunkOutput, moveEmbeddings);
        Tensor selectedActionIndices = moveIndices.to(_device).to_type(ScalarType.Int64).remainder(2).unsqueeze(-1);
        Tensor selectedLogits = actionLogits.gather(dim: 2, index: selectedActionIndices).squeeze(2);
        Tensor maskedLogits = PolicyLogitMask.Apply(gameStateTensors, selectedLogits, moveIndices);

        maskedLogits.MoveToOuterDisposeScope();
        return maskedLogits;
    }

    Tensor BuildActionLogits(Tensor trunkOutput, Tensor moveEmbeddings)
    {
        using var scope = NewDisposeScope();

        int width = _settings.ResidualWidth;
        Tensor trunkLeaf = _trunkLayerNorm.forward(_trunkProjection.forward(trunkOutput));
        Tensor moveNetworkOutput;

        if (_settings.PerLayerEmbedding)
        {
            Tensor perLayerMoveEmbedding = moveEmbeddings.view([
                moveEmbeddings.size(0), moveEmbeddings.size(1), _settings.BlockCount + 1, width]);
            Tensor perLayerTrunkLeaf = trunkLeaf.view([
                trunkOutput.size(0), _settings.BlockCount + 1, width]);

            moveNetworkOutput = perLayerMoveEmbedding.narrow(2, 0, 1).squeeze(2) +
                perLayerTrunkLeaf.narrow(1, 0, 1).squeeze(1).unsqueeze(1).expand(
                    trunkOutput.size(0), moveEmbeddings.size(1), width);
            for (int blockIndex = 0; blockIndex < _settings.BlockCount; ++blockIndex)
            {
                moveNetworkOutput = RunBlock(blockIndex, moveNetworkOutput, trunkOutput);
                moveNetworkOutput = moveNetworkOutput +
                    perLayerMoveEmbedding.narrow(2, blockIndex + 1, 1).squeeze(2) +
                    perLayerTrunkLeaf.narrow(1, blockIndex + 1, 1).squeeze(1).unsqueeze(1).expand(
                        trunkOutput.size(0), moveEmbeddings.size(1), width);
            }
        }
        else
        {
            moveNetworkOutput = moveEmbeddings + trunkLeaf.unsqueeze(1).expand(
                trunkOutput.size(0), moveEmbeddings.size(1), width);
            for (int blockIndex = 0; blockIndex < _settings.BlockCount; ++blockIndex)
                moveNetworkOutput = RunBlock(blockIndex, moveNetworkOutput, trunkOutput);
        }

        Tensor actionLogits = _logitHead.forward(functional.gelu(moveNetworkOutput));

        actionLogits.MoveToOuterDisposeScope();
        return actionLogits;
    }

    Tensor RunBlock(int blockIndex, Tensor moveNetworkOutput, Tensor trunkOutput)
    {
        if (!_settings.UseMoE)
            return _denseBlocks[blockIndex].forward(moveNetworkOutput);

        return _moeBlocks[blockIndex].Forward(
            moveNetworkOutput,
            trunkOutput,
            UpdateExpertLoadBalance,
            routerInputIsBlockInput: false);
    }
}
