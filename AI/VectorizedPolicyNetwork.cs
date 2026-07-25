namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class NewPolicyNetwork : Module, IPolicyNetwork, IAuxiliaryLossFreeLoadBalancedNetwork
{
    readonly GameStateVectorizer _stateVectorizer;
    readonly MoveVectorizer _moveVectorizer;
    readonly ModuleList<DeepSeekMoEResidualBlock> _trunkBlocks = new();
    readonly ModuleList<DeepSeekMoEResidualBlock> _moveBlocks = new();
    readonly Linear _valueHead;
    readonly LayerNorm _trunkLayerNorm;
    readonly LayerNorm _moveEmbeddingLayerNorm;
    readonly Linear _trunkProjection;
    readonly Linear _logitHead;
    readonly Settings _settings;
    readonly Device _device;

    public struct Settings
    {
        public int StateResidualWidth;
        public int ScoreBucketCount;
        public int TrunkBlockCount;
        public int TrunkRoutedExpertCount;
        public int TrunkChosenExpertCount;
        public float TrunkRoutingBiasUpdateSpeed;
        public ResidualBlock.ActivationType TrunkActivationType;
        public bool TrunkPerLayerEmbedding;
        public int MoveResidualWidth;
        public int MoveBlockCount;
        public int MoveRoutedExpertCount;
        public int MoveChosenExpertCount;
        public float MoveRoutingBiasUpdateSpeed;
        public ResidualBlock.ActivationType MoveActivationType;
        public bool LeafPerLayerEmbedding;
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
                ? settings.StateResidualWidth * (settings.TrunkBlockCount + 1)
                : settings.StateResidualWidth,
            scoreBucketCount: settings.ScoreBucketCount,
            device: _device);
        _moveVectorizer = new MoveVectorizer(
            moveEmbeddingWidth: settings.LeafPerLayerEmbedding
                ? settings.MoveResidualWidth * (settings.MoveBlockCount + 1)
                : settings.MoveResidualWidth,
            scoreBucketCount: settings.ScoreBucketCount,
            addWinningMoveEmbedding: settings.AddWinningMoveEmbedding,
            device: _device);

        for (int blockIndex = 0; blockIndex < settings.TrunkBlockCount; ++blockIndex)
        {
            _trunkBlocks.append(new DeepSeekMoEResidualBlock(
                residualWidth: settings.StateResidualWidth,
                routerInputWidth: settings.StateResidualWidth,
                routedExpertCount: settings.TrunkRoutedExpertCount,
                chosenExpertCount: settings.TrunkChosenExpertCount,
                routingBiasUpdateSpeed: settings.TrunkRoutingBiasUpdateSpeed,
                activationType: settings.TrunkActivationType,
                device: _device));
        }

        for (int blockIndex = 0; blockIndex < settings.MoveBlockCount; ++blockIndex)
        {
            _moveBlocks.append(new DeepSeekMoEResidualBlock(
                residualWidth: settings.MoveResidualWidth,
                routerInputWidth: settings.StateResidualWidth,
                routedExpertCount: settings.MoveRoutedExpertCount,
                chosenExpertCount: settings.MoveChosenExpertCount,
                routingBiasUpdateSpeed: settings.MoveRoutingBiasUpdateSpeed,
                activationType: settings.MoveActivationType,
                device: _device));
        }

        int moveEmbeddingWidth = settings.LeafPerLayerEmbedding
            ? settings.MoveResidualWidth * (settings.MoveBlockCount + 1)
            : settings.MoveResidualWidth;

        _valueHead = Linear(settings.StateResidualWidth, 1, device: _device);
        _trunkLayerNorm = LayerNorm(moveEmbeddingWidth, device: _device);
        _moveEmbeddingLayerNorm = LayerNorm(moveEmbeddingWidth, device: _device);
        _trunkProjection = Linear(
            settings.StateResidualWidth,
            moveEmbeddingWidth,
            device: _device);
        _logitHead = Linear(settings.MoveResidualWidth, 2, device: _device);

        RegisterComponents();
    }


    public (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs);
        Tensor policyLogits = BuildPolicyLogits(networkInputs, trunkOutput);
        Tensor value = _valueHead.forward(trunkOutput);

        policyLogits.MoveToOuterDisposeScope();
        value.MoveToOuterDisposeScope();
        return (policyLogits, value);
    }

    public bool UpdateExpertLoadBalance { get; set; }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs);
        Tensor policyLogits = BuildPolicyLogits(networkInputs, trunkOutput);

        policyLogits.MoveToOuterDisposeScope();
        return policyLogits;
    }


    public (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs);
        Tensor selectedPolicyLogits = BuildSelectedPolicyLogits(networkInputs, trunkOutput, moveIndices);
        Tensor value = _valueHead.forward(trunkOutput);

        selectedPolicyLogits.MoveToOuterDisposeScope();
        value.MoveToOuterDisposeScope();
        return (selectedPolicyLogits, value);
    }

    GameStateTensors MoveNetworkInputsToDevice(GameStateTensors gameStateTensors)
    {
        using var dScope = NewDisposeScope();

        GameStateTensors networkInputs = new()
        {
            FullHand = gameStateTensors.FullHand.to(_device),
            RemainingDeck = gameStateTensors.RemainingDeck.to(_device),
            Score = gameStateTensors.Score.to(_device),
            ScoreThreshold = gameStateTensors.ScoreThreshold.to(_device),
            PlayHandScores = gameStateTensors.PlayHandScores.to(_device),
            RemainingHands = gameStateTensors.RemainingHands.to(_device),
            RemainingDiscards = gameStateTensors.RemainingDiscards.to(_device),
        };
        networkInputs.ToOuterScope();
        return networkInputs;
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
                _settings.TrunkBlockCount + 1,
                _settings.StateResidualWidth]);
            trunkOutput = zeros([stateEmbedding.size(0), _settings.StateResidualWidth], device: _device);
            for (int blockIndex = 0; blockIndex < _trunkBlocks.Count; ++blockIndex)
            {
                Tensor blockInput = trunkOutput + perLayerEmbedding.narrow(1, blockIndex, 1).squeeze(1);
                trunkOutput = _trunkBlocks[blockIndex].Forward(
                    blockInput,
                    blockInput,
                    UpdateExpertLoadBalance,
                    routerInputIsBlockInput: true);
            }
            trunkOutput = trunkOutput + perLayerEmbedding.narrow(1, _settings.TrunkBlockCount, 1).squeeze(1);
        }
        else
        {
            trunkOutput = stateEmbedding;
            for (int blockIndex = 0; blockIndex < _trunkBlocks.Count; ++blockIndex)
                trunkOutput = _trunkBlocks[blockIndex].Forward(
                    trunkOutput,
                    trunkOutput,
                    UpdateExpertLoadBalance,
                    routerInputIsBlockInput: true);
        }

        trunkOutput.MoveToOuterDisposeScope();
        return trunkOutput;
    }


    Tensor BuildPolicyLogits(GameStateTensors gameStateTensors, Tensor trunkOutput)
    {
        using var scope = NewDisposeScope();

        Tensor moveEmbeddings = _moveEmbeddingLayerNorm.forward(_moveVectorizer.forward(gameStateTensors));
        Tensor flattenedLogits = BuildPolicyLogits(trunkOutput, moveEmbeddings);
        Tensor maskedLogits = PolicyLogitMask.Apply(gameStateTensors, flattenedLogits);

        maskedLogits.MoveToOuterDisposeScope();
        return maskedLogits;
    }


    Tensor BuildSelectedPolicyLogits(GameStateTensors gameStateTensors, Tensor trunkOutput, Tensor moveIndices)
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

        Tensor trunkLeaf = _trunkLayerNorm.forward(_trunkProjection.forward(trunkOutput));
        Tensor moveNetworkOutput;
        if (_settings.LeafPerLayerEmbedding)
        {
            Tensor perLayerMoveEmbedding = moveEmbeddings.view([
                moveEmbeddings.size(0),
                moveEmbeddings.size(1),
                _settings.MoveBlockCount + 1,
                _settings.MoveResidualWidth]);
            Tensor perLayerTrunkLeaf = trunkLeaf.view([
                trunkOutput.size(0),
                _settings.MoveBlockCount + 1,
                _settings.MoveResidualWidth]);

            moveNetworkOutput = perLayerMoveEmbedding.narrow(2, 0, 1).squeeze(2) +
                perLayerTrunkLeaf.narrow(1, 0, 1).squeeze(1).unsqueeze(1).expand(
                    trunkOutput.size(0),
                    moveEmbeddings.size(1),
                    _settings.MoveResidualWidth);
            for (int blockIndex = 0; blockIndex < _moveBlocks.Count; ++blockIndex)
            {
                moveNetworkOutput = _moveBlocks[blockIndex].Forward(
                    moveNetworkOutput,
                    trunkOutput,
                    UpdateExpertLoadBalance,
                    routerInputIsBlockInput: false);
                moveNetworkOutput = moveNetworkOutput +
                    perLayerMoveEmbedding.narrow(2, blockIndex + 1, 1).squeeze(2) +
                    perLayerTrunkLeaf.narrow(1, blockIndex + 1, 1).squeeze(1).unsqueeze(1).expand(
                        trunkOutput.size(0),
                        moveEmbeddings.size(1),
                        _settings.MoveResidualWidth);
            }
        }
        else
        {
            moveNetworkOutput = moveEmbeddings + trunkLeaf.unsqueeze(1).expand(
                trunkOutput.size(0),
                moveEmbeddings.size(1),
                _settings.MoveResidualWidth);
            for (int blockIndex = 0; blockIndex < _moveBlocks.Count; ++blockIndex)
                moveNetworkOutput = _moveBlocks[blockIndex].Forward(
                    moveNetworkOutput,
                    trunkOutput,
                    UpdateExpertLoadBalance,
                    routerInputIsBlockInput: false);
        }

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
        if (settings.TrunkRoutedExpertCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkRoutedExpertCount), "Trunk routed expert count must be positive.");
        if (settings.TrunkChosenExpertCount <= 0 || settings.TrunkChosenExpertCount > settings.TrunkRoutedExpertCount)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkChosenExpertCount), "Trunk chosen expert count must be between one and the routed expert count.");
        if (settings.TrunkRoutingBiasUpdateSpeed <= 0f)
            throw new ArgumentOutOfRangeException(nameof(settings.TrunkRoutingBiasUpdateSpeed), "Trunk routing bias update speed must be positive.");
        if (settings.MoveResidualWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveResidualWidth), "Move residual width must be positive.");
        if (settings.MoveBlockCount < 0)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveBlockCount), "Move block count must be non-negative.");
        if (settings.MoveRoutedExpertCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveRoutedExpertCount), "Move routed expert count must be positive.");
        if (settings.MoveChosenExpertCount <= 0 || settings.MoveChosenExpertCount > settings.MoveRoutedExpertCount)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveChosenExpertCount), "Move chosen expert count must be between one and the routed expert count.");
        if (settings.MoveRoutingBiasUpdateSpeed <= 0f)
            throw new ArgumentOutOfRangeException(nameof(settings.MoveRoutingBiasUpdateSpeed), "Move routing bias update speed must be positive.");
#pragma warning restore CA2208
    }
}
