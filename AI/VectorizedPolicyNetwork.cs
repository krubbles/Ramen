namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class NewPolicyNetwork : Module, IPolicyNetwork
{
    readonly GameStateVectorizer _stateVectorizer;
    readonly MoveVectorizer _moveVectorizer;
    readonly ModuleList<ResidualBlock> _trunkBlocks = new();
    readonly ModuleList<ResidualBlock> _moveBlocks = new();
    readonly Linear _valueHead;
    readonly LayerNorm _trunkLayerNorm;
    readonly Linear _trunkProjection;
    readonly Parameter _trunkGain;
    readonly Linear _logitHead;
    readonly Settings _settings;
    readonly Device _device;

    public struct Settings
    {
        public int StateResidualWidth;
        public int ScoreBucketCount;
        public int TrunkBlockCount;
        public float TrunkResidualRatio;
        public ResidualBlock.ActivationType TrunkActivationType;
        public bool TrunkPerLayerEmbedding;
        public int MoveResidualWidth;
        public int MoveBlockCount;
        public float MoveResidualRatio;
        public ResidualBlock.ActivationType MoveActivationType;
        public bool AddWinningMoveEmbedding;
        public Device Device;
    }

    public NewPolicyNetwork(Settings settings) : base(nameof(NewPolicyNetwork))
    {
        ValidateSettings(settings);

        _settings = settings;
        _device = settings.Device ?? CPU;

        _stateVectorizer = new GameStateVectorizer(
            embeddingWidth: settings.TrunkPerLayerEmbedding
                ? settings.StateResidualWidth * settings.TrunkBlockCount
                : settings.StateResidualWidth,
            scoreBucketCount: settings.ScoreBucketCount,
            device: _device);
        _moveVectorizer = new MoveVectorizer(
            moveEmbeddingWidth: settings.MoveResidualWidth,
            scoreBucketCount: settings.ScoreBucketCount,
            addWinningMoveEmbedding: settings.AddWinningMoveEmbedding,
            device: _device);

        for (int blockIndex = 0; blockIndex < settings.TrunkBlockCount; ++blockIndex)
        {
            _trunkBlocks.append(new ResidualBlock(
                residualWidth: settings.StateResidualWidth,
                hiddenRatio: settings.TrunkResidualRatio,
                activationType: settings.TrunkActivationType,
                device: _device));
        }

        for (int blockIndex = 0; blockIndex < settings.MoveBlockCount; ++blockIndex)
        {
            _moveBlocks.append(new ResidualBlock(
                residualWidth: settings.MoveResidualWidth,
                hiddenRatio: settings.MoveResidualRatio,
                activationType: settings.MoveActivationType,
                device: _device));
        }

        _valueHead = Linear(settings.StateResidualWidth, 1, device: _device);
        _trunkLayerNorm = LayerNorm(settings.StateResidualWidth, device: _device);
        _trunkProjection = Linear(settings.StateResidualWidth, settings.MoveResidualWidth, device: _device);
        _logitHead = Linear(settings.MoveResidualWidth, 2, device: _device);

        float trunkGainInitialization = settings.MoveResidualWidth / (float)settings.StateResidualWidth * 0.5f;
        _trunkGain = Parameter(full([settings.MoveResidualWidth], trunkGainInitialization, device: _device));

        RegisterComponents();
    }


    public (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor trunkOutput = BuildTrunkOutput(gameStateTensors);
        Tensor policyLogits = BuildPolicyLogits(gameStateTensors, trunkOutput);
        Tensor value = _valueHead.forward(trunkOutput);

        policyLogits.MoveToOuterDisposeScope();
        value.MoveToOuterDisposeScope();
        return (policyLogits, value);
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor trunkOutput = BuildTrunkOutput(gameStateTensors);
        Tensor policyLogits = BuildPolicyLogits(gameStateTensors, trunkOutput);

        policyLogits.MoveToOuterDisposeScope();
        return policyLogits;
    }


    public (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        Tensor trunkOutput = BuildTrunkOutput(gameStateTensors);
        Tensor selectedPolicyLogits = BuildSelectedPolicyLogits(gameStateTensors, trunkOutput, moveIndices);
        Tensor value = _valueHead.forward(trunkOutput);

        selectedPolicyLogits.MoveToOuterDisposeScope();
        value.MoveToOuterDisposeScope();
        return (selectedPolicyLogits, value);
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }


    Tensor BuildTrunkOutput(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor stateEmbedding = _stateVectorizer.forward(gameStateTensors);
        Tensor trunkOutput;
        if (_settings.TrunkPerLayerEmbedding)
        {
            Tensor perLayerEmbedding = stateEmbedding.view([
                stateEmbedding.size(0),
                _settings.TrunkBlockCount,
                _settings.StateResidualWidth]);
            trunkOutput = zeros([stateEmbedding.size(0), _settings.StateResidualWidth], device: _device);
            for (int blockIndex = 0; blockIndex < _trunkBlocks.Count; ++blockIndex)
                trunkOutput = _trunkBlocks[blockIndex].forward(trunkOutput + perLayerEmbedding.narrow(1, blockIndex, 1).squeeze(1));
        }
        else
        {
            trunkOutput = stateEmbedding;
            for (int blockIndex = 0; blockIndex < _trunkBlocks.Count; ++blockIndex)
                trunkOutput = _trunkBlocks[blockIndex].forward(trunkOutput);
        }

        trunkOutput.MoveToOuterDisposeScope();
        return trunkOutput;
    }


    Tensor BuildPolicyLogits(GameStateTensors gameStateTensors, Tensor trunkOutput)
    {
        using var scope = NewDisposeScope();

        Tensor moveEmbeddings = _moveVectorizer.forward(gameStateTensors);
        Tensor flattenedLogits = BuildPolicyLogits(trunkOutput, moveEmbeddings);

        flattenedLogits.MoveToOuterDisposeScope();
        return flattenedLogits;
    }


    Tensor BuildSelectedPolicyLogits(GameStateTensors gameStateTensors, Tensor trunkOutput, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        Tensor moveEmbeddings = _moveVectorizer.forward(gameStateTensors, moveIndices);
        Tensor actionLogits = BuildActionLogits(trunkOutput, moveEmbeddings);
        Tensor selectedActionIndices = moveIndices.to(_device).to_type(ScalarType.Int64).remainder(2).unsqueeze(-1);
        Tensor selectedLogits = actionLogits.gather(dim: 2, index: selectedActionIndices).squeeze(2);

        selectedLogits.MoveToOuterDisposeScope();
        return selectedLogits;
    }


    Tensor BuildPolicyLogits(Tensor trunkOutput, Tensor moveEmbeddings)
    {
        using var scope = NewDisposeScope();

        Tensor actionLogits = BuildActionLogits(trunkOutput, moveEmbeddings);
        Tensor policyLogits = actionLogits.view([trunkOutput.size(0), moveEmbeddings.size(1) * 2]);

        policyLogits.MoveToOuterDisposeScope();
        return policyLogits;
    }


    Tensor BuildActionLogits(Tensor trunkOutput, Tensor moveEmbeddings)
    {
        using var scope = NewDisposeScope();

        Tensor trunkLeaf = _trunkProjection.forward(_trunkLayerNorm.forward(trunkOutput)) * _trunkGain;
        Tensor moveNetworkOutput = moveEmbeddings + trunkLeaf.unsqueeze(1).expand(
            trunkOutput.size(0),
            moveEmbeddings.size(1),
            _settings.MoveResidualWidth);
        for (int blockIndex = 0; blockIndex < _moveBlocks.Count; ++blockIndex)
            moveNetworkOutput = _moveBlocks[blockIndex].forward(moveNetworkOutput);

        Tensor actionLogits = _logitHead.forward(functional.gelu(moveNetworkOutput));

        actionLogits.MoveToOuterDisposeScope();
        return actionLogits;
    }



    static void ValidateSettings(Settings settings)
    {
#pragma warning disable CA2208 // not passing argument name to AOOR constructor
        if (settings.StateResidualWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(settings.StateResidualWidth), "State residual width must be positive.");
        if (settings.ScoreBucketCount < 2)
            throw new ArgumentOutOfRangeException(nameof(settings.ScoreBucketCount), "Score bucket count must be at least 2.");
        if (settings.TrunkBlockCount < 0)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkBlockCount), "Trunk block count must be non-negative.");
        if (settings.TrunkPerLayerEmbedding && settings.TrunkBlockCount == 0)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkBlockCount), "Trunk block count must be positive when trunk per-layer embedding is enabled.");
        if (settings.TrunkResidualRatio <= 0f)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkResidualRatio), "Trunk hidden-to-residual width ratio must be positive.");
        if (settings.MoveResidualWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveResidualWidth), "Move residual width must be positive.");
        if (settings.MoveBlockCount < 0)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveBlockCount), "Move block count must be non-negative.");
        if (settings.MoveResidualRatio <= 0f)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveResidualRatio), "Move hidden-to-residual width ratio must be positive.");
#pragma warning restore CA2208
    }
}
