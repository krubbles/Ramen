namespace Ramen.AI;

using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public sealed class PpoPolicyValueModel : Module, IPolicyValueModel
{
    public static readonly Device EvalDevice = mps_is_available() ? MPS : CPU;
    public static readonly int[][] HandCombinations = Combinatorics.GetCombinations(
        setSize: GameData.HandSize,
        minSubsetSize: 1,
        maxSubsetSize: GameData.MaxPlayedHandSize);
    public static readonly int UseableHandCount = HandCombinations.Length;
    public static readonly int MoveCount = UseableHandCount * 2;

    static readonly long[,] PlayedHandMaskData = BuildHandMaskData(playedCards: true);
    static readonly long[,] RemainingHandMaskData = BuildHandMaskData(playedCards: false);

    readonly MaskedMeanCardSetEmbedding _fullHandEmbedding = new(embeddingWidth: 128, device: EvalDevice);
    readonly MaskedMeanCardSetEmbedding _remainingDeckEmbedding = new(embeddingWidth: 64, device: EvalDevice);
    readonly MaskedMeanCardSetEmbedding _playedHandEmbedding = new(embeddingWidth: 32, device: EvalDevice);
    readonly MaskedMeanCardSetEmbedding _remainingHandEmbedding = new(embeddingWidth: 32, device: EvalDevice);
    readonly BilinearOneHotScoreEmbedder _scoreEmbedding = new();
    readonly Linear _stateProjection = Linear(TrunkStateFeatureWidth, TrunkWidth, device: EvalDevice);
    readonly ModuleList<GeluResidualBlock> _stateResidualBlocks = new();
    readonly Linear _compressedStateProjection = Linear(TrunkWidth, CompactWidth, device: EvalDevice);
    readonly GELU _stateActivation = GELU();
    readonly Linear _moveOnlyProjection = Linear(MoveOnlyFeatureWidth, CompactWidth, device: EvalDevice);
    readonly GELU _moveMergeActivation = GELU();
    readonly GeluResidualBlock _moveResidualBlock = new(width: CompactWidth, hiddenWidth: CompactWidth, device: EvalDevice);
    readonly Linear _moveOutputProjection = Linear(CompactWidth, 1, device: EvalDevice);
    readonly Linear _valueHead = Linear(TrunkWidth, 1, device: EvalDevice);
    readonly Tensor _playedHandMask;
    readonly Tensor _remainingHandMask;

    const int TrunkWidth = 512;
    const int TrunkHiddenWidth = 1024;
    const int CompactWidth = 256;
    const int ScorePaddingWidth = 32;
    const int CountWidth = 20;
    const int CountPaddingWidth = 20;
    const int ScoreEmbeddingWidth = BilinearOneHotScoreEmbedder.BucketCount;
    const int TrunkStateFeatureWidth = 128 + 64 + ScoreEmbeddingWidth + ScorePaddingWidth + CountWidth + CountPaddingWidth;
    const int MoveOnlyFeatureWidth = 32 + 32 + ScoreEmbeddingWidth + CountWidth;
    const int TrunkResidualBlockCount = 4;

    public PpoPolicyValueModel() : base(nameof(PpoPolicyValueModel))
    {
        _playedHandMask = tensor(PlayedHandMaskData, dtype: ScalarType.Int64, device: EvalDevice);
        _remainingHandMask = tensor(RemainingHandMaskData, dtype: ScalarType.Int64, device: EvalDevice);
        _playedHandMask.DetachFromScope();
        _remainingHandMask.DetachFromScope();
        TensorManager.PersistForever(_playedHandMask);
        TensorManager.PersistForever(_remainingHandMask);

        for (int blockIndex = 0; blockIndex < TrunkResidualBlockCount; ++blockIndex)
        {
            _stateResidualBlocks.append(new GeluResidualBlock(
                width: TrunkWidth,
                hiddenWidth: TrunkHiddenWidth,
                device: EvalDevice));
        }

        RegisterComponents();
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        return GetPolicyLogitsAndValues(gameStateTensors, useHandTensors).logits;
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor moveIndices)
    {
        return GetSelectedPolicyLogitsAndValues(
            gameStateTensors: gameStateTensors,
            useHandTensors: useHandTensors,
            moveIndices: moveIndices).logits;
    }


    public (Tensor logits, Tensor values) GetPolicyLogitsAndValues(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        using var scope = NewDisposeScope();

        (Tensor compactState, Tensor trunkState) = EncodeState(gameStateTensors);
        Tensor playedHandEmbeddings = BuildHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _playedHandMask,
            embedder: _playedHandEmbedding);
        Tensor remainingHandEmbeddings = BuildHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _remainingHandMask,
            embedder: _remainingHandEmbedding);
        Tensor preScoreEmbedding = _scoreEmbedding
            .forward(gameStateTensors.Score.to(EvalDevice) * 300f)
            .squeeze(1);
        Tensor postPlayScoreEmbedding = _scoreEmbedding.forward(useHandTensors.Score.to(EvalDevice) * 300f);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor playPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState((remainingHands - 1).clamp_min(0), remainingDiscards),
            moveCount: UseableHandCount);
        Tensor discardPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState(remainingHands, (remainingDiscards - 1).clamp_min(0)),
            moveCount: UseableHandCount);
        Tensor compactStateExpanded = ExpandAcrossMoves(compactState, UseableHandCount);
        Tensor preScoreExpanded = ExpandAcrossMoves(preScoreEmbedding, UseableHandCount);

        Tensor playFeatures = cat(
            [
                compactStateExpanded,
                playedHandEmbeddings,
                remainingHandEmbeddings,
                postPlayScoreEmbedding,
                playPostCountEmbedding,
            ],
            dim: -1);
        Tensor discardFeatures = cat(
            [
                compactStateExpanded,
                playedHandEmbeddings,
                remainingHandEmbeddings,
                preScoreExpanded,
                discardPostCountEmbedding,
            ],
            dim: -1);
        Tensor playLogits = ScoreMoveFeatures(playFeatures).squeeze(-1);
        Tensor discardLogits = ScoreMoveFeatures(discardFeatures).squeeze(-1);
        Tensor logits = stack([playLogits, discardLogits], dim: 2).view([playLogits.size(0), MoveCount]);
        Tensor values = _valueHead.forward(trunkState).squeeze(-1);
        logits.MoveToOuterDisposeScope();
        values.MoveToOuterDisposeScope();
        return (logits, values);
    }


    public (Tensor logits, Tensor values) GetSelectedPolicyLogitsAndValues(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        Tensor selectedMoveIndices = moveIndices.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor selectedHandIndices = selectedMoveIndices.div(2).to_type(ScalarType.Int64);
        Tensor selectedActionIndices = selectedMoveIndices.remainder(2).to_type(ScalarType.Int64);
        int selectedMoveCount = (int)selectedMoveIndices.size(1);

        (Tensor compactState, Tensor trunkState) = EncodeState(gameStateTensors);
        Tensor selectedPlayedHandEmbeddings = BuildSelectedHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _playedHandMask,
            embedder: _playedHandEmbedding,
            selectedHandIndices: selectedHandIndices);
        Tensor selectedRemainingHandEmbeddings = BuildSelectedHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _remainingHandMask,
            embedder: _remainingHandEmbedding,
            selectedHandIndices: selectedHandIndices);
        Tensor preScoreEmbedding = _scoreEmbedding
            .forward(gameStateTensors.Score.to(EvalDevice) * 300f)
            .squeeze(1);
        Tensor selectedPostPlayScoreEmbedding = _scoreEmbedding
            .forward(useHandTensors.Score.to(EvalDevice).gather(dim: 1, index: selectedHandIndices) * 300f);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor playPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState((remainingHands - 1).clamp_min(0), remainingDiscards),
            moveCount: selectedMoveCount);
        Tensor discardPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState(remainingHands, (remainingDiscards - 1).clamp_min(0)),
            moveCount: selectedMoveCount);
        Tensor compactStateExpanded = ExpandAcrossMoves(compactState, selectedMoveCount);
        Tensor preScoreExpanded = ExpandAcrossMoves(preScoreEmbedding, selectedMoveCount);

        Tensor playFeatures = cat(
            [
                compactStateExpanded,
                selectedPlayedHandEmbeddings,
                selectedRemainingHandEmbeddings,
                selectedPostPlayScoreEmbedding,
                playPostCountEmbedding,
            ],
            dim: -1);
        Tensor discardFeatures = cat(
            [
                compactStateExpanded,
                selectedPlayedHandEmbeddings,
                selectedRemainingHandEmbeddings,
                preScoreExpanded,
                discardPostCountEmbedding,
            ],
            dim: -1);
        Tensor discardActionMask = selectedActionIndices.to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor mixedFeatures = playFeatures * (1f - discardActionMask) + discardFeatures * discardActionMask;
        Tensor selectedLogits = ScoreMoveFeatures(mixedFeatures).squeeze(-1);
        Tensor values = _valueHead.forward(trunkState).squeeze(-1);

        selectedLogits.MoveToOuterDisposeScope();
        values.MoveToOuterDisposeScope();
        return (selectedLogits, values);
    }


    public Tensor GetValues(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        (_, Tensor trunkState) = EncodeState(gameStateTensors);
        Tensor values = _valueHead.forward(trunkState).squeeze(-1);
        values.MoveToOuterDisposeScope();
        return values;
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }


    Tensor ScoreMoveFeatures(Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor compactStream = input[.., .., ..CompactWidth];
        Tensor moveOnlyFeatures = input[.., .., CompactWidth..];
        Tensor residualStream = compactStream + _moveOnlyProjection.forward(moveOnlyFeatures);
        residualStream = _moveMergeActivation.forward(residualStream);
        residualStream = _moveResidualBlock.forward(residualStream);
        Tensor output = _moveOutputProjection.forward(residualStream);
        output.MoveToOuterDisposeScope();
        return output;
    }


    (Tensor compactState, Tensor trunkState) EncodeState(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor fullHandEmbedding = _fullHandEmbedding.forward(gameStateTensors.FullHand.to(EvalDevice));
        Tensor remainingDeckEmbedding = _remainingDeckEmbedding.forward(gameStateTensors.RemainingDeck.to(EvalDevice));
        Tensor scoreEmbedding = _scoreEmbedding.forward(gameStateTensors.Score.to(EvalDevice) * 300f).squeeze(1);
        Tensor countEmbedding = EncodeCountState(
            remainingHands: gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64),
            remainingDiscards: gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64));
        Tensor paddedScoreEmbedding = PadLastDimWithZeros(scoreEmbedding, ScorePaddingWidth);
        Tensor paddedCountEmbedding = PadLastDimWithZeros(countEmbedding, CountPaddingWidth);

        Tensor stateFeatures = cat(
            [
                fullHandEmbedding,
                remainingDeckEmbedding,
                paddedScoreEmbedding,
                paddedCountEmbedding,
            ],
            dim: -1);
        Tensor trunkState = _stateProjection.forward(stateFeatures);
        for (int blockIndex = 0; blockIndex < _stateResidualBlocks.Count; ++blockIndex)
            trunkState = _stateResidualBlocks[blockIndex].forward(trunkState);

        Tensor compactState = _stateActivation.forward(_compressedStateProjection.forward(trunkState));
        compactState.MoveToOuterDisposeScope();
        trunkState.MoveToOuterDisposeScope();
        return (compactState, trunkState);
    }


    Tensor PadLastDimWithZeros(Tensor input, int paddingWidth)
    {
        using var scope = NewDisposeScope();

        if (paddingWidth <= 0)
        {
            input.MoveToOuterDisposeScope();
            return input;
        }

        Tensor padding = zeros([input.size(0), paddingWidth], dtype: input.dtype, device: input.device);
        Tensor padded = cat([input, padding], dim: -1);
        padded.MoveToOuterDisposeScope();
        return padded;
    }


    Tensor BuildHandEmbeddings(Tensor fullHand, Tensor handMask, MaskedMeanCardSetEmbedding embedder)
    {
        using var scope = NewDisposeScope();

        int batchSize = (int)fullHand.size(0);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor expandedMask = handMask.unsqueeze(0).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor maskedHands = expandedFullHand * expandedMask;
        Tensor embeddings = embedder.forward(maskedHands);
        embeddings.MoveToOuterDisposeScope();
        return embeddings;
    }


    Tensor BuildSelectedHandEmbeddings(Tensor fullHand, Tensor handMask, MaskedMeanCardSetEmbedding embedder, Tensor selectedHandIndices)
    {
        using var scope = NewDisposeScope();

        int batchSize = (int)fullHand.size(0);
        int selectedMoveCount = (int)selectedHandIndices.size(1);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, selectedMoveCount, GameData.HandSize);
        Tensor selectedMasks = handMask
            .index_select(dim: 0, index: selectedHandIndices.view(-1))
            .view(batchSize, selectedMoveCount, GameData.HandSize);
        Tensor maskedHands = expandedFullHand * selectedMasks;
        Tensor embeddings = embedder.forward(maskedHands);
        embeddings.MoveToOuterDisposeScope();
        return embeddings;
    }


    Tensor EncodeCountState(Tensor remainingHands, Tensor remainingDiscards)
    {
        using var scope = NewDisposeScope();

        Tensor combinedIndex = remainingHands.mul(4).add(remainingDiscards).to_type(ScalarType.Int64);
        Tensor encoded = functional.one_hot(combinedIndex, CountWidth).to_type(ScalarType.Float32);
        encoded.MoveToOuterDisposeScope();
        return encoded;
    }


    static Tensor ExpandAcrossMoves(Tensor tensorToExpand, int moveCount)
    {
        using var scope = NewDisposeScope();

        Tensor expanded = tensorToExpand
            .unsqueeze(1)
            .expand(tensorToExpand.size(0), moveCount, tensorToExpand.size(1));
        expanded.MoveToOuterDisposeScope();
        return expanded;
    }


    static long[,] BuildHandMaskData(bool playedCards)
    {
        long[,] handMask = new long[UseableHandCount, GameData.HandSize];
        for (int handIndex = 0; handIndex < HandCombinations.Length; ++handIndex)
        {
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                handMask[handIndex, cardIndex] = playedCards ? 0 : 1;

            int[] combination = HandCombinations[handIndex];
            for (int cardIndex = 0; cardIndex < combination.Length; ++cardIndex)
                handMask[handIndex, combination[cardIndex]] = playedCards ? 1 : 0;
        }

        return handMask;
    }
}

sealed class MaskedMeanCardSetEmbedding : Module<Tensor, Tensor>
{
    readonly Embedding _cardEmbedding;

    public Embedding CardEmbedding => _cardEmbedding;

    public MaskedMeanCardSetEmbedding(int embeddingWidth, Device device = null) : base(nameof(MaskedMeanCardSetEmbedding))
    {
        Device targetDevice = device ?? CPU;
        _cardEmbedding = Embedding(Card.RankCount * Card.SuitCount + 1, embeddingWidth, device: targetDevice);

        using var noGrad = no_grad();
        _cardEmbedding.weight[0].fill_(0f);
        RegisterComponents();
    }


    public override Tensor forward(Tensor cardSet)
    {
        using var scope = NewDisposeScope();

        Tensor cardIndices = cardSet.to_type(ScalarType.Int64);
        Tensor validMask = cardIndices.gt(0).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor embeddedCards = _cardEmbedding.forward(cardIndices);
        Tensor summed = (embeddedCards * validMask).sum(dim: embeddedCards.Dimensions - 2);
        Tensor counts = validMask.sum(dim: validMask.Dimensions - 2).clamp_min(1f);
        Tensor pooled = summed / counts;
        pooled.MoveToOuterDisposeScope();
        return pooled;
    }
}

sealed class GeluResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _hiddenProjection;
    readonly GELU _activation = GELU();
    readonly Linear _outputProjection;

    public LayerNorm LayerNorm => _layerNorm;

    public Linear HiddenProjection => _hiddenProjection;

    public Linear OutputProjection => _outputProjection;

    public GeluResidualBlock(int width, int hiddenWidth, Device device = null) : base(nameof(GeluResidualBlock))
    {
        Device targetDevice = device ?? CPU;
        _layerNorm = LayerNorm(width, device: targetDevice);
        _hiddenProjection = Linear(width, hiddenWidth, device: targetDevice);
        _outputProjection = Linear(hiddenWidth, width, device: targetDevice);
        RegisterComponents();
    }


    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor normalized = _layerNorm.forward(input);
        Tensor hidden = _hiddenProjection.forward(normalized);
        Tensor activated = _activation.forward(hidden);
        Tensor residual = _outputProjection.forward(activated);
        Tensor output = input + residual;
        output.MoveToOuterDisposeScope();
        return output;
    }
}
