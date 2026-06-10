namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class ModularPolicyModel : Module, IPolicyNetwork
{
    public static readonly Device EvalDevice = mps_is_available() ? MPS : CPU;
    public static readonly int UseableHandCount = Combinatorics.CalculateCombinationCount(
        setSize: GameData.HandSize,
        minSubsetSize: 1,
        maxSubsetSize: GameData.MaxPlayedHandSize);

    readonly Module<Tensor, Tensor> _fullHandCardSetEmbedder;
    readonly int _fullHandCardSetEmbeddingWidth;
    readonly Module<Tensor, Tensor> _remainingDeckCardSetEmbedder;
    readonly int _remainingDeckCardSetEmbeddingWidth;
    readonly Module<Tensor, Tensor> _usedHandCardSetEmbedder;
    readonly int _usedHandCardSetEmbeddingWidth;
    readonly Module<Tensor, Tensor> _remainingHandCardSetEmbedder;
    readonly int _remainingHandCardSetEmbeddingWidth;
    readonly Module<Tensor, Tensor> _preMoveScoreEmbedder;
    readonly int _preMoveScoreEmbeddingWidth;
    readonly Module<Tensor, Tensor> _postMoveScoreEmbedder;
    readonly int _postMoveScoreEmbeddingWidth;
    readonly Module<Tensor, Tensor> _remainingCountEmbedder;
    readonly int _remainingCountEmbeddingWidth;
    readonly int _residualWidth;
    readonly int _compressedStateWidth;
    readonly Module<Tensor, Tensor> _moveProcessor;
    readonly ModuleList<ModularResidualBlock> _residualBlocks = new();
    readonly Linear _compressedStateProjection;
    readonly Linear _valueHead;
    readonly Tensor _usedHandGatherIndices;
    readonly Tensor _usedHandValidMask;
    readonly Tensor _remainingHandGatherIndices;
    readonly Tensor _remainingHandValidMask;

    public enum ActivationFunctionKind
    {
        GELU,
        ReluSquared,
        SwiGLU,
    }

    public struct EmbedderSettings
    {
        public Module<Tensor, Tensor> Embedder;
        public int EmbeddingWidth;
    }

    public struct Settings
    {
        public EmbedderSettings FullHandCardSet;
        public EmbedderSettings RemainingDeckCardSet;
        public EmbedderSettings UsedHandCardSet;
        public EmbedderSettings RemainingHandCardSet;
        public EmbedderSettings PreMoveScore;
        public EmbedderSettings PostMoveScore;
        public EmbedderSettings RemainingCount;
        public int ResidualWidth;
        public int ResidualBlockCount;
        public float HiddenToResidualWidthRatio;
        public ActivationFunctionKind ActivationFunction;
        public int CompressedStateWidth;
        public Module<Tensor, Tensor> MoveProcessor;
    }

    public ModularPolicyModel(Settings settings) : base(nameof(ModularPolicyModel))
    {
        _fullHandCardSetEmbedder = settings.FullHandCardSet.Embedder;
        _fullHandCardSetEmbeddingWidth = settings.FullHandCardSet.EmbeddingWidth;
        _remainingDeckCardSetEmbedder = settings.RemainingDeckCardSet.Embedder;
        _remainingDeckCardSetEmbeddingWidth = settings.RemainingDeckCardSet.EmbeddingWidth;
        _usedHandCardSetEmbedder = settings.UsedHandCardSet.Embedder;
        _usedHandCardSetEmbeddingWidth = settings.UsedHandCardSet.EmbeddingWidth;
        _remainingHandCardSetEmbedder = settings.RemainingHandCardSet.Embedder;
        _remainingHandCardSetEmbeddingWidth = settings.RemainingHandCardSet.EmbeddingWidth;
        _preMoveScoreEmbedder = settings.PreMoveScore.Embedder;
        _preMoveScoreEmbeddingWidth = settings.PreMoveScore.EmbeddingWidth;
        _postMoveScoreEmbedder = settings.PostMoveScore.Embedder;
        _postMoveScoreEmbeddingWidth = settings.PostMoveScore.EmbeddingWidth;
        _remainingCountEmbedder = settings.RemainingCount.Embedder;
        _remainingCountEmbeddingWidth = settings.RemainingCount.EmbeddingWidth;
        _residualWidth = settings.ResidualWidth;
        _compressedStateWidth = settings.CompressedStateWidth;
        _moveProcessor = settings.MoveProcessor;

        ValidateSettings(settings);

        int initialStateInputWidth = GetInitialStateInputWidth(settings);
        if (initialStateInputWidth > _residualWidth)
            throw new InvalidOperationException($"Residual width {_residualWidth} is smaller than initial state width {initialStateInputWidth}.");

        int residualHiddenWidth = GetResidualBlockHiddenWidth(settings);
        for (int blockIndex = 0; blockIndex < settings.ResidualBlockCount; ++blockIndex)
        {
            _residualBlocks.append(new ModularResidualBlock(
                width: _residualWidth,
                hiddenWidth: residualHiddenWidth,
                activationFunction: settings.ActivationFunction,
                device: EvalDevice));
        }

        _compressedStateProjection = Linear(_residualWidth, _compressedStateWidth, device: EvalDevice);
        _valueHead = Linear(_compressedStateWidth, 1, device: EvalDevice);

        (long[,] usedHandGatherIndices, long[,] usedHandValidMask, long[,] remainingHandGatherIndices, long[,] remainingHandValidMask) = BuildHandSelectionTables();
        _usedHandGatherIndices = tensor(usedHandGatherIndices, dtype: ScalarType.Int64, device: EvalDevice);
        _usedHandValidMask = tensor(usedHandValidMask, dtype: ScalarType.Int64, device: EvalDevice);
        _remainingHandGatherIndices = tensor(remainingHandGatherIndices, dtype: ScalarType.Int64, device: EvalDevice);
        _remainingHandValidMask = tensor(remainingHandValidMask, dtype: ScalarType.Int64, device: EvalDevice);

        RegisterComponents();
    }


    public (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors)
    {
        Tensor policyLogits = GetPolicyLogits(gameStateTensors);
        Tensor value = GetValue(gameStateTensors);
        return (policyLogits, value);
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor compressedState = BuildCompressedState(gameStateTensors);
        Tensor fullHand = gameStateTensors.FullHand.to(EvalDevice);
        Tensor postPlayScores = GetPostPlayScores(gameStateTensors);
        int moveCount = (int)postPlayScores.size(1);

        Tensor compressedStateExpanded = ExpandAcrossMoves(compressedState, moveCount);
        Tensor usedHandEmbedding = BuildAllUsedHandEmbeddings(fullHand);
        Tensor remainingHandEmbedding = BuildAllRemainingHandEmbeddings(fullHand);
        Tensor playMoveFeatures = BuildAllActionMoveFeatures(
            compressedStateExpanded: compressedStateExpanded,
            usedHandEmbedding: usedHandEmbedding,
            remainingHandEmbedding: remainingHandEmbedding,
            gameStateTensors: gameStateTensors,
            postPlayScores: postPlayScores,
            isDiscardAction: false);
        Tensor discardMoveFeatures = BuildAllActionMoveFeatures(
            compressedStateExpanded: compressedStateExpanded,
            usedHandEmbedding: usedHandEmbedding,
            remainingHandEmbedding: remainingHandEmbedding,
            gameStateTensors: gameStateTensors,
            postPlayScores: postPlayScores,
            isDiscardAction: true);

        Tensor stackedMoveFeatures = stack([playMoveFeatures, discardMoveFeatures], dim: 2);
        Tensor flattenedMoveFeatures = stackedMoveFeatures.view([stackedMoveFeatures.size(0), moveCount * 2, stackedMoveFeatures.size(3)]);
        Tensor flattenedLogits = RunMoveProcessor(flattenedMoveFeatures);
        Tensor maskedLogits = PolicyLogitMask.Apply(gameStateTensors, flattenedLogits);

        maskedLogits.MoveToOuterDisposeScope();
        return maskedLogits;
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        Tensor selectedMoveIndices = moveIndices.to_type(ScalarType.Int64).to(EvalDevice);
        Tensor selectedHandIndices = selectedMoveIndices.div(2).to_type(ScalarType.Int64);
        Tensor actionIndices = selectedMoveIndices.remainder(2).to_type(ScalarType.Int64);

        Tensor compressedState = BuildCompressedState(gameStateTensors);
        Tensor fullHand = gameStateTensors.FullHand.to(EvalDevice);
        int moveCount = (int)selectedHandIndices.size(1);

        Tensor compressedStateExpanded = ExpandAcrossMoves(compressedState, moveCount);
        Tensor usedHandEmbedding = BuildSelectedUsedHandEmbeddings(fullHand, selectedHandIndices);
        Tensor remainingHandEmbedding = BuildSelectedRemainingHandEmbeddings(fullHand, selectedHandIndices);
        Tensor selectedMoveFeatures = BuildSelectedMoveFeatures(
            compressedStateExpanded: compressedStateExpanded,
            usedHandEmbedding: usedHandEmbedding,
            remainingHandEmbedding: remainingHandEmbedding,
            gameStateTensors: gameStateTensors,
            selectedHandIndices: selectedHandIndices,
            actionIndices: actionIndices);
        Tensor selectedMoveLogits = RunMoveProcessor(selectedMoveFeatures);
        Tensor maskedLogits = PolicyLogitMask.Apply(gameStateTensors, selectedMoveLogits, moveIndices);

        maskedLogits.MoveToOuterDisposeScope();
        return maskedLogits;
    }


    (Tensor policyLogits, Tensor value) IPolicyNetwork.GetPolicyValue(GameStateTensors gameStateTensors, Tensor moveIndices)
    {
        Tensor policyLogits = GetPolicyLogits(gameStateTensors, moveIndices);
        Tensor value = GetValue(gameStateTensors);
        return (policyLogits, value);
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }


    public static int GetInitialStateInputWidth(Settings settings)
    {
        return GetOptionalEmbeddingWidth(settings.FullHandCardSet) +
            GetOptionalEmbeddingWidth(settings.RemainingDeckCardSet) +
            GetOptionalEmbeddingWidth(settings.PreMoveScore) +
            GetOptionalEmbeddingWidth(settings.RemainingCount) * 2;
    }


    public static int GetMoveProcessorInputWidth(Settings settings)
    {
        return settings.CompressedStateWidth +
            GetOptionalEmbeddingWidth(settings.UsedHandCardSet) +
            GetOptionalEmbeddingWidth(settings.RemainingHandCardSet) +
            GetOptionalEmbeddingWidth(settings.PostMoveScore) +
            GetOptionalEmbeddingWidth(settings.RemainingCount) * 2;
    }


    public static int GetResidualBlockHiddenWidth(Settings settings)
    {
        float widthMultiplier = settings.ActivationFunction == ActivationFunctionKind.SwiGLU ? 2f / 3f : 1f;
        float rawHiddenWidth = settings.ResidualWidth * settings.HiddenToResidualWidthRatio * widthMultiplier;
        int roundedHiddenWidth = RoundToNearestMultipleOf16(rawHiddenWidth);
        return Math.Max(16, roundedHiddenWidth);
    }


    Tensor GetValue(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor compressedState = BuildCompressedState(gameStateTensors);
        Tensor value = _valueHead.forward(compressedState);
        value.MoveToOuterDisposeScope();
        return value;
    }


    Tensor BuildCompressedState(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor fullHand = gameStateTensors.FullHand.to(EvalDevice);
        Tensor remainingDeck = gameStateTensors.RemainingDeck.to(EvalDevice);
        Tensor preMoveScore = gameStateTensors.Score.to(EvalDevice);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64);

        List<Tensor> stateFeatures = new(capacity: 5);
        AppendStateFeature(stateFeatures, EmbedStateCardSet(_fullHandCardSetEmbedder, fullHand, "full hand"));
        AppendStateFeature(stateFeatures, EmbedStateCardSet(_remainingDeckCardSetEmbedder, remainingDeck, "remaining deck"));
        AppendStateFeature(stateFeatures, EmbedStateCount(remainingHands, "remaining hands"));
        AppendStateFeature(stateFeatures, EmbedStateCount(remainingDiscards, "remaining discards"));
        AppendStateFeature(stateFeatures, EmbedStateScore(_preMoveScoreEmbedder, preMoveScore, "pre move score"));

        Tensor stateFeatureTensor = ConcatStateFeatures(stateFeatures, batchSize: fullHand.size(0));
        Tensor residualStream = PadResidualStream(stateFeatureTensor);
        for (int blockIndex = 0; blockIndex < _residualBlocks.Count; ++blockIndex)
            residualStream = residualStream + _residualBlocks[blockIndex].forward(residualStream);

        Tensor compressedState = _compressedStateProjection.forward(residualStream);
        compressedState.MoveToOuterDisposeScope();
        return compressedState;
    }


    Tensor BuildAllUsedHandEmbeddings(Tensor fullHand)
    {
        if (_usedHandCardSetEmbedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor usedHands = GatherAllHandVariants(fullHand, _usedHandGatherIndices, _usedHandValidMask);
        Tensor usedHandEmbedding = EnsureMoveEmbedding(
            embedding: _usedHandCardSetEmbedder.forward(usedHands),
            expectedMoveCount: UseableHandCount,
            name: "used hand");
        usedHandEmbedding.MoveToOuterDisposeScope();
        return usedHandEmbedding;
    }


    Tensor BuildAllRemainingHandEmbeddings(Tensor fullHand)
    {
        if (_remainingHandCardSetEmbedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor remainingHands = GatherAllHandVariants(fullHand, _remainingHandGatherIndices, _remainingHandValidMask);
        Tensor remainingHandEmbedding = EnsureMoveEmbedding(
            embedding: _remainingHandCardSetEmbedder.forward(remainingHands),
            expectedMoveCount: UseableHandCount,
            name: "remaining hand");
        remainingHandEmbedding.MoveToOuterDisposeScope();
        return remainingHandEmbedding;
    }


    Tensor BuildSelectedUsedHandEmbeddings(Tensor fullHand, Tensor selectedHandIndices)
    {
        if (_usedHandCardSetEmbedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor usedHands = GatherSelectedHandVariants(
            fullHand: fullHand,
            selectedHandIndices: selectedHandIndices,
            gatherIndices: _usedHandGatherIndices,
            validMask: _usedHandValidMask);
        Tensor usedHandEmbedding = EnsureMoveEmbedding(
            embedding: _usedHandCardSetEmbedder.forward(usedHands),
            expectedMoveCount: selectedHandIndices.size(1),
            name: "selected used hand");
        usedHandEmbedding.MoveToOuterDisposeScope();
        return usedHandEmbedding;
    }


    Tensor BuildSelectedRemainingHandEmbeddings(Tensor fullHand, Tensor selectedHandIndices)
    {
        if (_remainingHandCardSetEmbedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor remainingHands = GatherSelectedHandVariants(
            fullHand: fullHand,
            selectedHandIndices: selectedHandIndices,
            gatherIndices: _remainingHandGatherIndices,
            validMask: _remainingHandValidMask);
        Tensor remainingHandEmbedding = EnsureMoveEmbedding(
            embedding: _remainingHandCardSetEmbedder.forward(remainingHands),
            expectedMoveCount: selectedHandIndices.size(1),
            name: "selected remaining hand");
        remainingHandEmbedding.MoveToOuterDisposeScope();
        return remainingHandEmbedding;
    }


    Tensor BuildAllActionMoveFeatures(
        Tensor compressedStateExpanded,
        Tensor usedHandEmbedding,
        Tensor remainingHandEmbedding,
        GameStateTensors gameStateTensors,
        Tensor postPlayScores,
        bool isDiscardAction)
    {
        using var scope = NewDisposeScope();

        int moveCount = (int)postPlayScores.size(1);
        Tensor stateScore = gameStateTensors.Score.to(EvalDevice);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64);

        Tensor postMoveHands = isDiscardAction ? remainingHands : (remainingHands - 1).clamp_min(0);
        Tensor postMoveDiscards = isDiscardAction ? (remainingDiscards - 1).clamp_min(0) : remainingDiscards;

        Tensor postMoveHandsEmbedding = ExpandAcrossMoves(EmbedStateCount(postMoveHands, "post move hands"), moveCount);
        Tensor postMoveDiscardsEmbedding = ExpandAcrossMoves(EmbedStateCount(postMoveDiscards, "post move discards"), moveCount);
        Tensor postMoveScoreEmbedding = isDiscardAction
            ? ExpandAcrossMoves(EmbedStateScore(_postMoveScoreEmbedder, stateScore, "discard score"), moveCount)
            : EmbedMoveScore(postPlayScores, "play score");

        List<Tensor> moveFeatures = new(capacity: 6)
        {
            compressedStateExpanded,
        };
        AppendMoveFeature(moveFeatures, usedHandEmbedding);
        AppendMoveFeature(moveFeatures, remainingHandEmbedding);
        AppendMoveFeature(moveFeatures, postMoveHandsEmbedding);
        AppendMoveFeature(moveFeatures, postMoveDiscardsEmbedding);
        AppendMoveFeature(moveFeatures, postMoveScoreEmbedding);

        Tensor actionMoveFeatures = ConcatMoveFeatures(
            moveFeatures: moveFeatures,
            batchSize: compressedStateExpanded.size(0),
            moveCount: moveCount);
        actionMoveFeatures.MoveToOuterDisposeScope();
        return actionMoveFeatures;
    }


    Tensor BuildSelectedMoveFeatures(
        Tensor compressedStateExpanded,
        Tensor usedHandEmbedding,
        Tensor remainingHandEmbedding,
        GameStateTensors gameStateTensors,
        Tensor selectedHandIndices,
        Tensor actionIndices)
    {
        using var scope = NewDisposeScope();

        int moveCount = (int)selectedHandIndices.size(1);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor stateScore = gameStateTensors.Score.to(EvalDevice);
        Tensor actionMask = actionIndices.eq(1).unsqueeze(-1);

        Tensor playHandsEmbedding = ExpandAcrossMoves(EmbedStateCount((remainingHands - 1).clamp_min(0), "play hands"), moveCount);
        Tensor discardHandsEmbedding = ExpandAcrossMoves(EmbedStateCount(remainingHands, "discard hands"), moveCount);
        Tensor selectedPostMoveHandsEmbedding = SelectByAction(playHandsEmbedding, discardHandsEmbedding, actionMask);

        Tensor playDiscardsEmbedding = ExpandAcrossMoves(EmbedStateCount(remainingDiscards, "play discards"), moveCount);
        Tensor discardDiscardsEmbedding = ExpandAcrossMoves(EmbedStateCount((remainingDiscards - 1).clamp_min(0), "discard discards"), moveCount);
        Tensor selectedPostMoveDiscardsEmbedding = SelectByAction(playDiscardsEmbedding, discardDiscardsEmbedding, actionMask);

        Tensor selectedPostPlayScores = GetPostPlayScores(gameStateTensors)
            .gather(dim: 1, index: selectedHandIndices);
        Tensor playScoreEmbedding = EmbedMoveScore(selectedPostPlayScores, "selected play score");
        Tensor discardScoreEmbedding = ExpandAcrossMoves(EmbedStateScore(_postMoveScoreEmbedder, stateScore, "selected discard score"), moveCount);
        Tensor selectedPostMoveScoreEmbedding = SelectByAction(playScoreEmbedding, discardScoreEmbedding, actionMask);

        List<Tensor> moveFeatures = new(capacity: 6)
        {
            compressedStateExpanded,
        };
        AppendMoveFeature(moveFeatures, usedHandEmbedding);
        AppendMoveFeature(moveFeatures, remainingHandEmbedding);
        AppendMoveFeature(moveFeatures, selectedPostMoveHandsEmbedding);
        AppendMoveFeature(moveFeatures, selectedPostMoveDiscardsEmbedding);
        AppendMoveFeature(moveFeatures, selectedPostMoveScoreEmbedding);

        Tensor selectedMoveFeatures = ConcatMoveFeatures(
            moveFeatures: moveFeatures,
            batchSize: compressedStateExpanded.size(0),
            moveCount: moveCount);
        selectedMoveFeatures.MoveToOuterDisposeScope();
        return selectedMoveFeatures;
    }


    static Tensor GetPostPlayScores(GameStateTensors gameStateTensors)
    {
        return gameStateTensors.Score.to(EvalDevice) + gameStateTensors.PlayHandScores.to(EvalDevice);
    }


    Tensor GatherAllHandVariants(Tensor fullHand, Tensor gatherIndices, Tensor validMask)
    {
        using var scope = NewDisposeScope();

        long batchSize = fullHand.size(0);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor expandedGatherIndices = gatherIndices.unsqueeze(0).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor gatheredHands = expandedFullHand.gather(dim: 2, index: expandedGatherIndices);
        Tensor maskedHands = gatheredHands * validMask.unsqueeze(0).expand(batchSize, UseableHandCount, GameData.HandSize);
        maskedHands.MoveToOuterDisposeScope();
        return maskedHands;
    }


    Tensor GatherSelectedHandVariants(Tensor fullHand, Tensor selectedHandIndices, Tensor gatherIndices, Tensor validMask)
    {
        using var scope = NewDisposeScope();

        long batchSize = fullHand.size(0);
        int moveCount = (int)selectedHandIndices.size(1);
        Tensor flatSelectedHandIndices = selectedHandIndices.view(-1);
        Tensor selectedGatherIndices = gatherIndices
            .index_select(dim: 0, index: flatSelectedHandIndices)
            .view(batchSize, moveCount, GameData.HandSize);
        Tensor selectedValidMask = validMask
            .index_select(dim: 0, index: flatSelectedHandIndices)
            .view(batchSize, moveCount, GameData.HandSize);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, moveCount, GameData.HandSize);
        Tensor gatheredHands = expandedFullHand.gather(dim: 2, index: selectedGatherIndices);
        Tensor maskedHands = gatheredHands * selectedValidMask;
        maskedHands.MoveToOuterDisposeScope();
        return maskedHands;
    }


    Tensor EmbedStateCardSet(Module<Tensor, Tensor> embedder, Tensor cardSet, string name)
    {
        if (embedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor embedding = EnsureStateEmbedding(embedder.forward(cardSet), name);
        embedding.MoveToOuterDisposeScope();
        return embedding;
    }


    Tensor EmbedStateScore(Module<Tensor, Tensor> embedder, Tensor score, string name)
    {
        if (embedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor embedding = EnsureStateEmbedding(embedder.forward(score), name);
        embedding.MoveToOuterDisposeScope();
        return embedding;
    }


    Tensor EmbedMoveScore(Tensor score, string name)
    {
        if (_postMoveScoreEmbedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor embedding = EnsureMoveEmbedding(
            embedding: _postMoveScoreEmbedder.forward(score),
            expectedMoveCount: score.size(1),
            name: name);
        embedding.MoveToOuterDisposeScope();
        return embedding;
    }


    Tensor EmbedStateCount(Tensor count, string name)
    {
        if (_remainingCountEmbedder is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor embedding = EnsureStateEmbedding(_remainingCountEmbedder.forward(count), name);
        embedding.MoveToOuterDisposeScope();
        return embedding;
    }


    Tensor PadResidualStream(Tensor stateFeatures)
    {
        using var scope = NewDisposeScope();

        int stateWidth = (int)stateFeatures.size(1);
        if (stateWidth > _residualWidth)
            throw new InvalidOperationException($"State width {stateWidth} exceeds residual width {_residualWidth}.");
        if (stateWidth == _residualWidth)
        {
            stateFeatures.MoveToOuterDisposeScope();
            return stateFeatures;
        }

        Tensor zeroPadding = zeros(
            [stateFeatures.size(0), _residualWidth - stateWidth],
            dtype: stateFeatures.dtype,
            device: stateFeatures.device);
        Tensor paddedStateFeatures = cat([stateFeatures, zeroPadding], dim: -1);
        paddedStateFeatures.MoveToOuterDisposeScope();
        return paddedStateFeatures;
    }


    Tensor RunMoveProcessor(Tensor moveFeatures)
    {
        using var scope = NewDisposeScope();

        Tensor rawLogits = _moveProcessor.forward(moveFeatures);
        Tensor logits = rawLogits.Dimensions switch
        {
            2 => rawLogits,
            3 when rawLogits.size(2) == 1 => rawLogits.squeeze(2),
            _ => throw new InvalidOperationException($"Move processor must return shape (batch, moveCount) or (batch, moveCount, 1), got {rawLogits.shape}."),
        };
        logits.MoveToOuterDisposeScope();
        return logits;
    }


    static Tensor EnsureStateEmbedding(Tensor embedding, string name)
    {
        using var scope = NewDisposeScope();

        Tensor normalized = embedding.Dimensions switch
        {
            2 => embedding,
            3 when embedding.size(1) == 1 => embedding.squeeze(1),
            _ => throw new InvalidOperationException($"{name} embedding must have shape (batch, width) or (batch, 1, width), got {embedding.shape}."),
        };
        normalized.MoveToOuterDisposeScope();
        return normalized;
    }


    static Tensor EnsureMoveEmbedding(Tensor embedding, long expectedMoveCount, string name)
    {
        if (embedding.Dimensions != 3 || embedding.size(1) != expectedMoveCount)
            throw new InvalidOperationException($"{name} embedding must have shape (batch, {expectedMoveCount}, width), got {embedding.shape}.");
        return embedding;
    }


    static Tensor ExpandAcrossMoves(Tensor stateEmbedding, int moveCount)
    {
        if (stateEmbedding is null)
            return null;

        using var scope = NewDisposeScope();

        Tensor expandedEmbedding = stateEmbedding
            .unsqueeze(1)
            .expand(stateEmbedding.size(0), moveCount, stateEmbedding.size(1));
        expandedEmbedding.MoveToOuterDisposeScope();
        return expandedEmbedding;
    }


    static Tensor SelectByAction(Tensor playTensor, Tensor discardTensor, Tensor actionMask)
    {
        using var scope = NewDisposeScope();

        Tensor selectedTensor = where(actionMask, discardTensor, playTensor);
        selectedTensor.MoveToOuterDisposeScope();
        return selectedTensor;
    }


    static Tensor ConcatStateFeatures(List<Tensor> stateFeatures, long batchSize)
    {
        using var scope = NewDisposeScope();

        Tensor result = stateFeatures.Count == 0
            ? zeros([batchSize, 0], dtype: ScalarType.Float32, device: EvalDevice)
            : cat([.. stateFeatures], dim: -1);
        result.MoveToOuterDisposeScope();
        return result;
    }


    static Tensor ConcatMoveFeatures(List<Tensor> moveFeatures, long batchSize, int moveCount)
    {
        using var scope = NewDisposeScope();

        Tensor result = moveFeatures.Count == 0
            ? zeros([batchSize, moveCount, 0], dtype: ScalarType.Float32, device: EvalDevice)
            : cat([.. moveFeatures], dim: -1);
        result.MoveToOuterDisposeScope();
        return result;
    }


    static void AppendStateFeature(List<Tensor> features, Tensor feature)
    {
        if (feature is not null)
            features.Add(feature);
    }


    static void AppendMoveFeature(List<Tensor> features, Tensor feature)
    {
        if (feature is not null)
            features.Add(feature);
    }


    static void ValidateSettings(Settings settings)
    {
        ValidateEmbedderSettings(settings.FullHandCardSet, nameof(settings.FullHandCardSet));
        ValidateEmbedderSettings(settings.RemainingDeckCardSet, nameof(settings.RemainingDeckCardSet));
        ValidateEmbedderSettings(settings.UsedHandCardSet, nameof(settings.UsedHandCardSet));
        ValidateEmbedderSettings(settings.RemainingHandCardSet, nameof(settings.RemainingHandCardSet));
        ValidateEmbedderSettings(settings.PreMoveScore, nameof(settings.PreMoveScore));
        ValidateEmbedderSettings(settings.PostMoveScore, nameof(settings.PostMoveScore));
        ValidateEmbedderSettings(settings.RemainingCount, nameof(settings.RemainingCount));

        if (settings.ResidualWidth <= 0)
            throw new InvalidOperationException("ResidualWidth must be positive.");
        if (settings.ResidualBlockCount < 0)
            throw new InvalidOperationException("ResidualBlockCount must be non-negative.");
        if (settings.HiddenToResidualWidthRatio <= 0f)
            throw new InvalidOperationException("HiddenToResidualWidthRatio must be positive.");
        if (settings.CompressedStateWidth <= 0)
            throw new InvalidOperationException("CompressedStateWidth must be positive.");
        if (settings.MoveProcessor is null)
            throw new InvalidOperationException("MoveProcessor must not be null.");
    }


    static void ValidateEmbedderSettings(EmbedderSettings settings, string name)
    {
        if (settings.Embedder is null && settings.EmbeddingWidth != 0)
            throw new InvalidOperationException($"{name}.EmbeddingWidth must be 0 when the embedder is null.");
        if (settings.Embedder is not null && settings.EmbeddingWidth <= 0)
            throw new InvalidOperationException($"{name}.EmbeddingWidth must be positive when the embedder is not null.");
    }


    static int GetOptionalEmbeddingWidth(EmbedderSettings settings)
    {
        return settings.Embedder is null ? 0 : settings.EmbeddingWidth;
    }


    static int RoundToNearestMultipleOf16(float value)
    {
        return (int)(MathF.Round(value / 16f) * 16f);
    }


    static (long[,] usedHandGatherIndices, long[,] usedHandValidMask, long[,] remainingHandGatherIndices, long[,] remainingHandValidMask) BuildHandSelectionTables()
    {
        int[][] combinations = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: GameData.MaxPlayedHandSize);
        long[,] usedHandGatherIndices = new long[UseableHandCount, GameData.HandSize];
        long[,] usedHandValidMask = new long[UseableHandCount, GameData.HandSize];
        long[,] remainingHandGatherIndices = new long[UseableHandCount, GameData.HandSize];
        long[,] remainingHandValidMask = new long[UseableHandCount, GameData.HandSize];

        for (int handIndex = 0; handIndex < combinations.Length; ++handIndex)
        {
            int[] usedCardIndices = combinations[handIndex];
            bool[] usedCardLookup = new bool[GameData.HandSize];

            for (int usedCardIndex = 0; usedCardIndex < usedCardIndices.Length; ++usedCardIndex)
            {
                int fullHandIndex = usedCardIndices[usedCardIndex];
                usedHandGatherIndices[handIndex, usedCardIndex] = fullHandIndex;
                usedHandValidMask[handIndex, usedCardIndex] = 1;
                usedCardLookup[fullHandIndex] = true;
            }

            int remainingWriteIndex = 0;
            for (int fullHandIndex = 0; fullHandIndex < GameData.HandSize; ++fullHandIndex)
            {
                if (usedCardLookup[fullHandIndex])
                    continue;

                remainingHandGatherIndices[handIndex, remainingWriteIndex] = fullHandIndex;
                remainingHandValidMask[handIndex, remainingWriteIndex] = 1;
                remainingWriteIndex++;
            }
        }

        return (usedHandGatherIndices, usedHandValidMask, remainingHandGatherIndices, remainingHandValidMask);
    }
}

sealed class ModularResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _hiddenProjection;
    readonly Linear _gateProjection;
    readonly Linear _valueProjection;
    readonly Linear _outputProjection;
    readonly ModularPolicyModel.ActivationFunctionKind _activationFunction;

    public ModularResidualBlock(
        int width,
        int hiddenWidth,
        ModularPolicyModel.ActivationFunctionKind activationFunction,
        Device device) : base(nameof(ModularResidualBlock))
    {
        _activationFunction = activationFunction;
        _layerNorm = LayerNorm(width, device: device);

        if (_activationFunction == ModularPolicyModel.ActivationFunctionKind.SwiGLU)
        {
            _gateProjection = Linear(width, hiddenWidth, hasBias: false, device: device);
            _valueProjection = Linear(width, hiddenWidth, hasBias: false, device: device);
            _outputProjection = Linear(hiddenWidth, width, hasBias: false, device: device);
        }
        else
        {
            _hiddenProjection = Linear(width, hiddenWidth, device: device);
            _outputProjection = Linear(hiddenWidth, width, device: device);
        }

        RegisterComponents();
    }


    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor normalized = _layerNorm.forward(input);
        Tensor residual = _activationFunction switch
        {
            ModularPolicyModel.ActivationFunctionKind.GELU => _outputProjection.forward(functional.gelu(_hiddenProjection.forward(normalized))),
            ModularPolicyModel.ActivationFunctionKind.ReluSquared => _outputProjection.forward(GetReluSquared(_hiddenProjection.forward(normalized))),
            ModularPolicyModel.ActivationFunctionKind.SwiGLU => _outputProjection.forward(functional.silu(_gateProjection.forward(normalized)) * _valueProjection.forward(normalized)),
            _ => throw new InvalidOperationException($"Unknown activation function {_activationFunction}."),
        };
        Tensor output = input + residual;
        output.MoveToOuterDisposeScope();
        return output;
    }


    static Tensor GetReluSquared(Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor output = functional.relu(input).square();
        output.MoveToOuterDisposeScope();
        return output;
    }
}
