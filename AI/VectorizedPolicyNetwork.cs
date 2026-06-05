namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class VectorizedPolicyNetwork : Module, IPolicyNetwork
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
        public int MoveResidualWidth;
        public int MoveBlockCount;
        public float MoveResidualRatio;
        public ResidualBlock.ActivationType MoveActivationType;
        public bool AddWinningMoveEmbedding;
        public Device Device;
    }

    public VectorizedPolicyNetwork(Settings settings) : base(nameof(VectorizedPolicyNetwork))
    {
        ValidateSettings(settings);

        _settings = settings;
        _device = settings.Device ?? CPU;

        _stateVectorizer = new GameStateVectorizer(
            embeddingWidth: settings.StateResidualWidth,
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
        _logitHead = Linear(settings.MoveResidualWidth, 1, device: _device);

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
        Tensor allPolicyLogits = BuildPolicyLogits(gameStateTensors, trunkOutput);
        Tensor selectedMoveIndices = moveIndices.to(_device).to_type(ScalarType.Int64);
        Tensor selectedPolicyLogits = allPolicyLogits.gather(dim: 1, index: selectedMoveIndices);
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

        Tensor trunkOutput = _stateVectorizer.forward(gameStateTensors);
        for (int blockIndex = 0; blockIndex < _trunkBlocks.Count; ++blockIndex)
            trunkOutput = _trunkBlocks[blockIndex].forward(trunkOutput);

        trunkOutput.MoveToOuterDisposeScope();
        return trunkOutput;
    }


    Tensor BuildPolicyLogits(GameStateTensors gameStateTensors, Tensor trunkOutput)
    {
        using var scope = NewDisposeScope();

        Tensor moveEmbeddings = _moveVectorizer.forward(gameStateTensors)
            .view([trunkOutput.size(0), MoveVectorizer.PlayableHandCount, 2, _settings.MoveResidualWidth]);
        Tensor trunkLeaf = _trunkProjection.forward(_trunkLayerNorm.forward(trunkOutput)) * _trunkGain;
        Tensor moveNetworkOutput = moveEmbeddings + trunkLeaf.unsqueeze(1).unsqueeze(2).expand(
            trunkOutput.size(0),
            MoveVectorizer.PlayableHandCount,
            2,
            _settings.MoveResidualWidth);
        for (int blockIndex = 0; blockIndex < _moveBlocks.Count; ++blockIndex)
            moveNetworkOutput = _moveBlocks[blockIndex].forward(moveNetworkOutput);

        Tensor actionLogits = _logitHead.forward(moveNetworkOutput).squeeze(3);
        Tensor flattenedLogits = actionLogits.view([trunkOutput.size(0), MoveVectorizer.PlayableHandCount * 2]);

        flattenedLogits.MoveToOuterDisposeScope();
        return flattenedLogits;
    }



    static void ValidateSettings(Settings settings)
    {
        if (settings.StateResidualWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(settings.StateResidualWidth), "State residual width must be positive.");
        if (settings.ScoreBucketCount < 2)
            throw new ArgumentOutOfRangeException(nameof(settings.ScoreBucketCount), "Score bucket count must be at least 2.");
        if (settings.TrunkBlockCount < 0)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkBlockCount), "Trunk block count must be non-negative.");
        if (settings.TrunkResidualRatio <= 0f)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkResidualRatio), "Trunk hidden-to-residual width ratio must be positive.");
        if (settings.MoveResidualWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveResidualWidth), "Move residual width must be positive.");
        if (settings.MoveBlockCount < 0)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveBlockCount), "Move block count must be non-negative.");
        if (settings.MoveResidualRatio <= 0f)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveResidualRatio), "Move hidden-to-residual width ratio must be positive.");
    }
}
