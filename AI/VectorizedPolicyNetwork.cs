namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class NewPolicyNetwork : Module, IPolicyNetwork, IAuxiliaryLossFreeLoadBalancedNetwork
{
    readonly GameStateVectorizer _stateVectorizer;
    readonly ModuleList<DeepSeekMoEResidualBlock> _trunkBlocks = new();
    readonly LeafTower _primaryLeaf;
    readonly LeafTower _secondaryLeaf;
    readonly Linear _valueHead;
    readonly Settings _settings;
    readonly Device _device;

    public struct Settings
    {
        public int StateResidualWidth;
        public int ScoreBucketCount;
        public int TrunkBlockCount;
        public int TrunkRoutedExpertCount;
        public int TrunkChosenExpertCount;
        /// <summary>
        /// Routed expert hidden width as a fraction of the residual width. Zero uses the
        /// 0.5 default. The shared expert is unaffected.
        /// </summary>
        public float TrunkRoutedExpertHiddenRatio;
        /// <summary>
        /// Shared expert hidden width as a fraction of the residual width. Zero uses the
        /// 1.0 default, which matches the residual width.
        /// </summary>
        public float TrunkSharedExpertHiddenRatio;
        public float TrunkRoutingBiasUpdateSpeed;
        public ResidualBlock.ActivationType TrunkActivationType;
        public bool TrunkPerLayerEmbedding;
        public int MoveResidualWidth;
        public int MoveBlockCount;
        public int MoveRoutedExpertCount;
        public int MoveChosenExpertCount;
        /// <summary>
        /// Routed expert hidden width as a fraction of the residual width. Zero uses the
        /// 0.5 default. The shared expert is unaffected.
        /// </summary>
        public float MoveRoutedExpertHiddenRatio;
        /// <summary>
        /// Shared expert hidden width as a fraction of the residual width. Zero uses the
        /// 1.0 default, which matches the residual width.
        /// </summary>
        public float MoveSharedExpertHiddenRatio;
        public float MoveRoutingBiasUpdateSpeed;
        public ResidualBlock.ActivationType MoveActivationType;
        public bool LeafPerLayerEmbedding;
        /// <summary>
        /// When positive the move blocks are dense <see cref="ResidualBlock"/>s with this
        /// hidden-to-residual ratio. Zero keeps the routed MoE blocks.
        /// </summary>
        public float MoveDenseHiddenRatio;
        /// <summary>
        /// An optional second move tower hanging off the same trunk, with its own move
        /// vectorizer, adapter and head. Used to train a small tower alongside the policy
        /// tower and measure how well it approximates it.
        /// </summary>
        public LeafTower.Settings? SecondaryLeaf;
        public bool AddWinningMoveEmbedding;
        /// <summary>
        /// Expert capacity as a multiple of the mean expert load. Assignments past the
        /// cap get no routed expert output. Zero uses the default of 4.
        /// </summary>
        public float ExpertCapacityFactor;
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
        for (int blockIndex = 0; blockIndex < settings.TrunkBlockCount; ++blockIndex)
        {
            _trunkBlocks.append(new DeepSeekMoEResidualBlock(
                residualWidth: settings.StateResidualWidth,
                routerInputWidth: settings.StateResidualWidth,
                routedExpertCount: settings.TrunkRoutedExpertCount,
                chosenExpertCount: settings.TrunkChosenExpertCount,
                routingBiasUpdateSpeed: settings.TrunkRoutingBiasUpdateSpeed,
                activationType: settings.TrunkActivationType,
                routedExpertHiddenRatio: settings.TrunkRoutedExpertHiddenRatio > 0f
                    ? settings.TrunkRoutedExpertHiddenRatio
                    : 0.5f,
                expertCapacityFactor: settings.ExpertCapacityFactor > 0f
                    ? settings.ExpertCapacityFactor
                    : 4f,
                sharedExpertHiddenRatio: settings.TrunkSharedExpertHiddenRatio > 0f
                    ? settings.TrunkSharedExpertHiddenRatio
                    : 1f,
                device: _device));
        }

        LeafTower.Settings primaryLeafSettings = new()
        {
            ResidualWidth = settings.MoveResidualWidth,
            BlockCount = settings.MoveBlockCount,
            HiddenRatio = settings.MoveDenseHiddenRatio,
            PerLayerEmbedding = settings.LeafPerLayerEmbedding,
            ActivationType = settings.MoveActivationType,
            UseMoE = settings.MoveDenseHiddenRatio <= 0f,
            RoutedExpertCount = settings.MoveRoutedExpertCount,
            ChosenExpertCount = settings.MoveChosenExpertCount,
            RoutedExpertHiddenRatio = settings.MoveRoutedExpertHiddenRatio,
            SharedExpertHiddenRatio = settings.MoveSharedExpertHiddenRatio,
            RoutingBiasUpdateSpeed = settings.MoveRoutingBiasUpdateSpeed,
            ExpertCapacityFactor = settings.ExpertCapacityFactor,
        };
        _primaryLeaf = new LeafTower(
            primaryLeafSettings,
            trunkWidth: settings.StateResidualWidth,
            scoreBucketCount: settings.ScoreBucketCount,
            addWinningMoveEmbedding: settings.AddWinningMoveEmbedding,
            device: _device,
            name: "PrimaryLeaf");

        if (settings.SecondaryLeaf is { } secondaryLeafSettings)
        {
            _secondaryLeaf = new LeafTower(
                secondaryLeafSettings,
                trunkWidth: settings.StateResidualWidth,
                scoreBucketCount: settings.ScoreBucketCount,
                addWinningMoveEmbedding: settings.AddWinningMoveEmbedding,
                device: _device,
                name: "SecondaryLeaf");
        }

        _valueHead = Linear(settings.StateResidualWidth, 1, device: _device);

        RegisterComponents();
    }


    public (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs);
        Tensor policyLogits = _primaryLeaf.Forward(networkInputs, trunkOutput);
        Tensor value = _valueHead.forward(trunkOutput);

        policyLogits.MoveToOuterDisposeScope();
        value.MoveToOuterDisposeScope();
        return (policyLogits, value);
    }

    bool _updateExpertLoadBalance;

    public bool UpdateExpertLoadBalance
    {
        get => _updateExpertLoadBalance;
        set
        {
            _updateExpertLoadBalance = value;
            _primaryLeaf.UpdateExpertLoadBalance = value;
            if (_secondaryLeaf is not null)
                _secondaryLeaf.UpdateExpertLoadBalance = value;
        }
    }

    public bool HasSecondaryLeaf => _secondaryLeaf is not null;

    /// <summary>
    /// Full-move logits from both towers plus the value, from a single trunk pass. The
    /// secondary tower reads a detached trunk so its loss cannot perturb trunk training,
    /// keeping the policy tower's dynamics comparable to a single-tower run.
    /// </summary>
    public (Tensor primaryLogits, Tensor secondaryLogits, Tensor value) GetDualPolicyValue(
        GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs);
        Tensor primaryLogits = _primaryLeaf.Forward(networkInputs, trunkOutput);
        Tensor secondaryLogits = _secondaryLeaf is null
            ? null
            : _secondaryLeaf.Forward(networkInputs, trunkOutput.detach());
        Tensor value = _valueHead.forward(trunkOutput);

        primaryLogits.MoveToOuterDisposeScope();
        secondaryLogits?.MoveToOuterDisposeScope();
        value.MoveToOuterDisposeScope();
        return (primaryLogits, secondaryLogits, value);
    }

    /// <summary>Secondary-tower logits for a chosen subset of moves, on a detached trunk.</summary>
    public Tensor GetSecondaryPolicyLogits(GameStateTensors gameStateTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs).detach();
        Tensor logits = _secondaryLeaf.ForwardSelected(networkInputs, trunkOutput, moveIndices);

        logits.MoveToOuterDisposeScope();
        return logits;
    }

    public List<MoERoutingStats> DrainRoutingStats()
    {
        List<MoERoutingStats> stats = [];
        for (int blockIndex = 0; blockIndex < _trunkBlocks.Count; ++blockIndex)
            stats.AddRange(_trunkBlocks[blockIndex].DrainRoutingStats());
        stats.AddRange(_primaryLeaf.DrainRoutingStats());
        if (_secondaryLeaf is not null)
            stats.AddRange(_secondaryLeaf.DrainRoutingStats());
        return stats;
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs);
        Tensor policyLogits = _primaryLeaf.Forward(networkInputs, trunkOutput);

        policyLogits.MoveToOuterDisposeScope();
        return policyLogits;
    }


    public (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        GameStateTensors networkInputs = MoveNetworkInputsToDevice(gameStateTensors);
        Tensor trunkOutput = BuildTrunkOutput(networkInputs);
        Tensor selectedPolicyLogits = _primaryLeaf.ForwardSelected(networkInputs, trunkOutput, moveIndices);
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
        using var profileScope = ProfileScope.New("TrunkBlocks" + Profiling.PhaseSuffix);

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
        // Routed-expert settings only apply when the move blocks are MoE blocks.
        if (settings.MoveDenseHiddenRatio <= 0f)
        {
            if (settings.MoveRoutedExpertCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(settings.MoveRoutedExpertCount), "Move routed expert count must be positive.");
            if (settings.MoveChosenExpertCount <= 0 || settings.MoveChosenExpertCount > settings.MoveRoutedExpertCount)
                throw new ArgumentOutOfRangeException(nameof(settings.MoveChosenExpertCount), "Move chosen expert count must be between one and the routed expert count.");
            if (settings.MoveRoutingBiasUpdateSpeed <= 0f)
                throw new ArgumentOutOfRangeException(nameof(settings.MoveRoutingBiasUpdateSpeed), "Move routing bias update speed must be positive.");
        }
#pragma warning restore CA2208
    }
}
