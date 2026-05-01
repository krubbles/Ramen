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

    readonly BilinearOneHotScoreEmbedder _roundScoreEmbedding = new();
    readonly Linear _roundStateProjection = Linear(TrunkStateFeatureWidth, TrunkWidth, device: EvalDevice);
    readonly ModuleList<GeluResidualBlock> _roundStateResidualBlocks = new();
    readonly Linear _roundCompressedStateProjection = Linear(TrunkWidth, CompactWidth, device: EvalDevice);
    readonly GELU _roundStateActivation = GELU();
    readonly Linear _roundMoveOnlyProjection = Linear(MoveOnlyFeatureWidth, CompactWidth, device: EvalDevice);
    readonly GELU _roundMoveMergeActivation = GELU();
    readonly GeluResidualBlock _roundMoveResidualBlock = new(width: CompactWidth, hiddenWidth: CompactWidth, device: EvalDevice);
    readonly Linear _roundMoveOutputProjection = Linear(CompactWidth, 1, device: EvalDevice);
    readonly Linear _roundValueHead = Linear(TrunkWidth, 1, device: EvalDevice);
    readonly BilinearOneHotScoreEmbedder _storeScoreEmbedding = new();
    readonly Linear _storeStateProjection = Linear(TrunkStateFeatureWidth, TrunkWidth, device: EvalDevice);
    readonly ModuleList<GeluResidualBlock> _storeStateResidualBlocks = new();
    readonly Linear _storeCompressedStateProjection = Linear(TrunkWidth, CompactWidth, device: EvalDevice);
    readonly GELU _storeStateActivation = GELU();
    readonly Embedding _storeActionEmbedding = Embedding(StoreSpecialActionCount, CompactWidth, device: EvalDevice);
    readonly Embedding _storeJokerEmbedding = Embedding(JokerCountWidth + 1, CompactWidth, device: EvalDevice);
    readonly Embedding _storePriceEmbedding = Embedding(MaxStorePrice + 1, CompactWidth, device: EvalDevice);
    readonly GELU _storeMergeActivation = GELU();
    readonly GeluResidualBlock _storeResidualBlock = new(width: CompactWidth, hiddenWidth: CompactWidth, device: EvalDevice);
    readonly Linear _storeOutputProjection = Linear(CompactWidth, 1, device: EvalDevice);
    readonly Linear _storeValueHead = Linear(TrunkWidth, 1, device: EvalDevice);
    readonly Tensor _playedHandMask;
    readonly Tensor _remainingHandMask;

    const int TrunkWidth = 768;
    const int TrunkHiddenWidth = 2000;
    const int CompactWidth = 256;
    const int CardCountWidth = Card.RankCount * Card.SuitCount;
    const int CountWidth = 20;
    const int MoneyEmbeddingWidth = 51;
    const int RoundEmbeddingWidth = 25;
    const int StageEmbeddingWidth = 2;
    const int ScoreEmbeddingWidth = BilinearOneHotScoreEmbedder.BucketCount;
    const int MoveOnlyFeatureWidth = CardCountWidth * 2 + ScoreEmbeddingWidth + CountWidth;
    const int TrunkResidualBlockCount = 4;
    const int MaxStorePrice = 10;
    const int StoreSpecialActionCount = 2;
    static readonly int JokerCountWidth = Joker.Page1Jokers.Length;
    static readonly int TrunkStateFeatureWidth = CardCountWidth * 2 + JokerCountWidth * 2 + MoneyEmbeddingWidth + RoundEmbeddingWidth + StageEmbeddingWidth + ScoreEmbeddingWidth + CountWidth;

    public PpoPolicyValueModel() : base(nameof(PpoPolicyValueModel))
    {
        _playedHandMask = tensor(PlayedHandMaskData, dtype: ScalarType.Int64, device: EvalDevice);
        _remainingHandMask = tensor(RemainingHandMaskData, dtype: ScalarType.Int64, device: EvalDevice);
        _playedHandMask.DetachFromScope();
        _remainingHandMask.DetachFromScope();
        TensorManager.PersistForever(_playedHandMask);
        TensorManager.PersistForever(_remainingHandMask);

        using var noGrad = no_grad();
        _storeJokerEmbedding.weight[0].fill_(0f);

        for (int blockIndex = 0; blockIndex < TrunkResidualBlockCount; ++blockIndex)
        {
            _roundStateResidualBlocks.append(new GeluResidualBlock(
                width: TrunkWidth,
                hiddenWidth: TrunkHiddenWidth,
                device: EvalDevice));
            _storeStateResidualBlocks.append(new GeluResidualBlock(
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


    public Tensor GetStorePolicyLogits(GameStateTensors gameStateTensors)
    {
        return GetStorePolicyLogits(
            gameStateTensors: gameStateTensors,
            scoreEmbedding: _storeScoreEmbedding,
            stateProjection: _storeStateProjection,
            stateResidualBlocks: _storeStateResidualBlocks,
            compressedStateProjection: _storeCompressedStateProjection,
            stateActivation: _storeStateActivation,
            storeActionEmbedding: _storeActionEmbedding,
            storeJokerEmbedding: _storeJokerEmbedding,
            storePriceEmbedding: _storePriceEmbedding,
            storeMergeActivation: _storeMergeActivation,
            storeResidualBlock: _storeResidualBlock,
            storeOutputProjection: _storeOutputProjection);
    }


    public (Tensor logits, Tensor values) GetPolicyLogitsAndValues(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        return GetPolicyLogitsAndValues(
            gameStateTensors: gameStateTensors,
            useHandTensors: useHandTensors,
            playedHandMask: _playedHandMask,
            remainingHandMask: _remainingHandMask,
            scoreEmbedding: _roundScoreEmbedding,
            stateProjection: _roundStateProjection,
            stateResidualBlocks: _roundStateResidualBlocks,
            compressedStateProjection: _roundCompressedStateProjection,
            stateActivation: _roundStateActivation,
            moveOnlyProjection: _roundMoveOnlyProjection,
            moveMergeActivation: _roundMoveMergeActivation,
            moveResidualBlock: _roundMoveResidualBlock,
            moveOutputProjection: _roundMoveOutputProjection,
            valueHead: _roundValueHead);
    }


    public (Tensor logits, Tensor values) GetSelectedPolicyLogitsAndValues(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor moveIndices)
    {
        return GetSelectedPolicyLogitsAndValues(
            gameStateTensors: gameStateTensors,
            useHandTensors: useHandTensors,
            moveIndices: moveIndices,
            playedHandMask: _playedHandMask,
            remainingHandMask: _remainingHandMask,
            scoreEmbedding: _roundScoreEmbedding,
            stateProjection: _roundStateProjection,
            stateResidualBlocks: _roundStateResidualBlocks,
            compressedStateProjection: _roundCompressedStateProjection,
            stateActivation: _roundStateActivation,
            moveOnlyProjection: _roundMoveOnlyProjection,
            moveMergeActivation: _roundMoveMergeActivation,
            moveResidualBlock: _roundMoveResidualBlock,
            moveOutputProjection: _roundMoveOutputProjection,
            valueHead: _roundValueHead);
    }


    public Tensor GetValues(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor roundValues = GetValues(
            gameStateTensors: gameStateTensors,
            scoreEmbedding: _roundScoreEmbedding,
            stateProjection: _roundStateProjection,
            stateResidualBlocks: _roundStateResidualBlocks,
            compressedStateProjection: _roundCompressedStateProjection,
            stateActivation: _roundStateActivation,
            valueHead: _roundValueHead);
        Tensor storeValues = GetValues(
            gameStateTensors: gameStateTensors,
            scoreEmbedding: _storeScoreEmbedding,
            stateProjection: _storeStateProjection,
            stateResidualBlocks: _storeStateResidualBlocks,
            compressedStateProjection: _storeCompressedStateProjection,
            stateActivation: _storeStateActivation,
            valueHead: _storeValueHead);
        Tensor stageValues = gameStateTensors.Stage.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor storeMask = stageValues.eq((long)StageOfGame.InShop).to_type(ScalarType.Float32);
        Tensor values = roundValues * (1f - storeMask) + storeValues * storeMask;
        values.MoveToOuterDisposeScope();
        return values;
    }


    static Tensor GetStorePolicyLogits(
        GameStateTensors gameStateTensors,
        BilinearOneHotScoreEmbedder scoreEmbedding,
        Linear stateProjection,
        ModuleList<GeluResidualBlock> stateResidualBlocks,
        Linear compressedStateProjection,
        GELU stateActivation,
        Embedding storeActionEmbedding,
        Embedding storeJokerEmbedding,
        Embedding storePriceEmbedding,
        GELU storeMergeActivation,
        GeluResidualBlock storeResidualBlock,
        Linear storeOutputProjection)
    {
        using var scope = NewDisposeScope();

        (_, Tensor trunkState) = EncodeState(
            gameStateTensors: gameStateTensors,
            scoreEmbedding: scoreEmbedding,
            stateProjection: stateProjection,
            stateResidualBlocks: stateResidualBlocks,
            compressedStateProjection: compressedStateProjection,
            stateActivation: stateActivation);
        Tensor compactState = stateActivation.forward(compressedStateProjection.forward(trunkState));
        Tensor storeJokerIndices = gameStateTensors.StoreJokers.to(EvalDevice).to_type(ScalarType.Int64);
        int batchSize = (int)storeJokerIndices.size(0);

        Tensor exitIndices = zeros([batchSize], dtype: ScalarType.Int64, device: EvalDevice);
        Tensor rerollIndices = ones([batchSize], dtype: ScalarType.Int64, device: EvalDevice);
        Tensor exitInputs = compactState + storeActionEmbedding.forward(exitIndices);
        Tensor rerollInputs = compactState +
            storeActionEmbedding.forward(rerollIndices) +
            EncodeStorePriceEmbeddings(gameStateTensors.RerollPrice.to(EvalDevice), storePriceEmbedding);

        Tensor storeJokerEmbeddings = storeJokerEmbedding.forward(storeJokerIndices);
        Tensor storePriceEmbeddings = EncodeStorePriceEmbeddings(gameStateTensors.StorePrices.to(EvalDevice), storePriceEmbedding);
        Tensor expandedCompactState = compactState
            .unsqueeze(1)
            .expand(batchSize, GameStateEmbedder.MaxStoreJokerCount, CompactWidth);
        Tensor buyInputs = expandedCompactState + storeJokerEmbeddings + storePriceEmbeddings;

        Tensor specialInputs = stack([exitInputs, rerollInputs], dim: 1);
        Tensor specialLogits = ScoreStoreFeatures(specialInputs, storeMergeActivation, storeResidualBlock, storeOutputProjection).squeeze(-1);
        Tensor buyLogits = ScoreStoreFeatures(buyInputs, storeMergeActivation, storeResidualBlock, storeOutputProjection).squeeze(-1);
        Tensor nullMask = storeJokerIndices.eq(0).to_type(ScalarType.Float32) * -1e9f;
        Tensor logits = cat([specialLogits, buyLogits + nullMask], dim: 1);
        logits.MoveToOuterDisposeScope();
        return logits;
    }


    static (Tensor logits, Tensor values) GetPolicyLogitsAndValues(
        GameStateTensors gameStateTensors,
        UseHandTensors useHandTensors,
        Tensor playedHandMask,
        Tensor remainingHandMask,
        BilinearOneHotScoreEmbedder scoreEmbedding,
        Linear stateProjection,
        ModuleList<GeluResidualBlock> stateResidualBlocks,
        Linear compressedStateProjection,
        GELU stateActivation,
        Linear moveOnlyProjection,
        GELU moveMergeActivation,
        GeluResidualBlock moveResidualBlock,
        Linear moveOutputProjection,
        Linear valueHead)
    {
        using var scope = NewDisposeScope();

        (Tensor compactState, Tensor trunkState) = EncodeState(
            gameStateTensors: gameStateTensors,
            scoreEmbedding: scoreEmbedding,
            stateProjection: stateProjection,
            stateResidualBlocks: stateResidualBlocks,
            compressedStateProjection: compressedStateProjection,
            stateActivation: stateActivation);
        Tensor playedHandEmbeddings = BuildHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: playedHandMask);
        Tensor remainingHandEmbeddings = BuildHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: remainingHandMask);
        Tensor preScoreEmbedding = scoreEmbedding
            .forward(gameStateTensors.Score.to(EvalDevice) * 300f)
            .squeeze(1);
        Tensor postPlayScoreEmbedding = scoreEmbedding.forward(useHandTensors.Score.to(EvalDevice) * 300f);
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
        Tensor playLogits = ScoreMoveFeatures(playFeatures, moveOnlyProjection, moveMergeActivation, moveResidualBlock, moveOutputProjection).squeeze(-1);
        Tensor discardLogits = ScoreMoveFeatures(discardFeatures, moveOnlyProjection, moveMergeActivation, moveResidualBlock, moveOutputProjection).squeeze(-1);
        Tensor logits = stack([playLogits, discardLogits], dim: 2).view([playLogits.size(0), MoveCount]);
        Tensor values = valueHead.forward(trunkState).squeeze(-1);
        logits.MoveToOuterDisposeScope();
        values.MoveToOuterDisposeScope();
        return (logits, values);
    }


    static (Tensor logits, Tensor values) GetSelectedPolicyLogitsAndValues(
        GameStateTensors gameStateTensors,
        UseHandTensors useHandTensors,
        Tensor moveIndices,
        Tensor playedHandMask,
        Tensor remainingHandMask,
        BilinearOneHotScoreEmbedder scoreEmbedding,
        Linear stateProjection,
        ModuleList<GeluResidualBlock> stateResidualBlocks,
        Linear compressedStateProjection,
        GELU stateActivation,
        Linear moveOnlyProjection,
        GELU moveMergeActivation,
        GeluResidualBlock moveResidualBlock,
        Linear moveOutputProjection,
        Linear valueHead)
    {
        using var scope = NewDisposeScope();

        Tensor selectedMoveIndices = moveIndices.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor selectedHandIndices = selectedMoveIndices.div(2).to_type(ScalarType.Int64);
        Tensor selectedActionIndices = selectedMoveIndices.remainder(2).to_type(ScalarType.Int64);
        int selectedMoveCount = (int)selectedMoveIndices.size(1);

        (Tensor compactState, Tensor trunkState) = EncodeState(
            gameStateTensors: gameStateTensors,
            scoreEmbedding: scoreEmbedding,
            stateProjection: stateProjection,
            stateResidualBlocks: stateResidualBlocks,
            compressedStateProjection: compressedStateProjection,
            stateActivation: stateActivation);
        Tensor selectedPlayedHandEmbeddings = BuildSelectedHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: playedHandMask,
            selectedHandIndices: selectedHandIndices);
        Tensor selectedRemainingHandEmbeddings = BuildSelectedHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: remainingHandMask,
            selectedHandIndices: selectedHandIndices);
        Tensor preScoreEmbedding = scoreEmbedding
            .forward(gameStateTensors.Score.to(EvalDevice) * 300f)
            .squeeze(1);
        Tensor selectedPostPlayScoreEmbedding = scoreEmbedding
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
        Tensor selectedLogits = ScoreMoveFeatures(mixedFeatures, moveOnlyProjection, moveMergeActivation, moveResidualBlock, moveOutputProjection).squeeze(-1);
        Tensor values = valueHead.forward(trunkState).squeeze(-1);

        selectedLogits.MoveToOuterDisposeScope();
        values.MoveToOuterDisposeScope();
        return (selectedLogits, values);
    }


    static Tensor GetValues(
        GameStateTensors gameStateTensors,
        BilinearOneHotScoreEmbedder scoreEmbedding,
        Linear stateProjection,
        ModuleList<GeluResidualBlock> stateResidualBlocks,
        Linear compressedStateProjection,
        GELU stateActivation,
        Linear valueHead)
    {
        using var scope = NewDisposeScope();

        (_, Tensor trunkState) = EncodeState(
            gameStateTensors: gameStateTensors,
            scoreEmbedding: scoreEmbedding,
            stateProjection: stateProjection,
            stateResidualBlocks: stateResidualBlocks,
            compressedStateProjection: compressedStateProjection,
            stateActivation: stateActivation);
        Tensor values = valueHead.forward(trunkState).squeeze(-1);
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


    static Tensor ScoreMoveFeatures(Tensor input, Linear moveOnlyProjection, GELU moveMergeActivation, GeluResidualBlock moveResidualBlock, Linear moveOutputProjection)
    {
        using var scope = NewDisposeScope();

        Tensor compactStream = input[.., .., ..CompactWidth];
        Tensor moveOnlyFeatures = input[.., .., CompactWidth..];
        Tensor residualStream = compactStream + moveOnlyProjection.forward(moveOnlyFeatures);
        residualStream = moveResidualBlock.forward(residualStream);
        residualStream = moveMergeActivation.forward(residualStream);
        Tensor output = moveOutputProjection.forward(residualStream);
        output.MoveToOuterDisposeScope();
        return output;
    }


    static Tensor ScoreStoreFeatures(Tensor input, GELU storeMergeActivation, GeluResidualBlock storeResidualBlock, Linear storeOutputProjection)
    {
        using var scope = NewDisposeScope();

        Tensor residualStream = storeMergeActivation.forward(input);
        residualStream = storeResidualBlock.forward(residualStream);
        Tensor output = storeOutputProjection.forward(residualStream);
        output.MoveToOuterDisposeScope();
        return output;
    }


    static (Tensor compactState, Tensor trunkState) EncodeState(
        GameStateTensors gameStateTensors,
        BilinearOneHotScoreEmbedder scoreEmbedding,
        Linear stateProjection,
        ModuleList<GeluResidualBlock> stateResidualBlocks,
        Linear compressedStateProjection,
        GELU stateActivation)
    {
        using var scope = NewDisposeScope();

        Tensor fullHandEmbedding = EncodeCardCounts(gameStateTensors.FullHand.to(EvalDevice));
        Tensor remainingDeckEmbedding = EncodeCardCounts(gameStateTensors.RemainingDeck.to(EvalDevice));
        Tensor ownedJokers = EncodeJokerCounts(gameStateTensors.OwnedJokers.to(EvalDevice));
        Tensor storeJokers = EncodeJokerCounts(gameStateTensors.StoreJokers.to(EvalDevice));
        Tensor moneyEmbedding = EncodeMoneyState(gameStateTensors.Money.to(EvalDevice));
        Tensor roundEmbedding = EncodeRoundState(gameStateTensors.Round.to(EvalDevice));
        Tensor stageEmbedding = EncodeStageState(gameStateTensors.Stage.to(EvalDevice));
        Tensor scoreEmbeddingValue = scoreEmbedding.forward(gameStateTensors.Score.to(EvalDevice) * 300f).squeeze(1);
        Tensor countEmbedding = EncodeCountState(
            remainingHands: gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64),
            remainingDiscards: gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64));

        Tensor stateFeatures = cat(
            [
                fullHandEmbedding,
                remainingDeckEmbedding,
                ownedJokers,
                storeJokers,
                moneyEmbedding,
                roundEmbedding,
                stageEmbedding,
                scoreEmbeddingValue,
                countEmbedding,
            ],
            dim: -1);
        Tensor trunkState = stateProjection.forward(stateFeatures);
        for (int blockIndex = 0; blockIndex < stateResidualBlocks.Count; ++blockIndex)
            trunkState = stateResidualBlocks[blockIndex].forward(trunkState);

        Tensor compactState = stateActivation.forward(compressedStateProjection.forward(trunkState));
        compactState.MoveToOuterDisposeScope();
        trunkState.MoveToOuterDisposeScope();
        return (compactState, trunkState);
    }


    static Tensor BuildHandEmbeddings(Tensor fullHand, Tensor handMask)
    {
        using var scope = NewDisposeScope();

        int batchSize = (int)fullHand.size(0);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor expandedMask = handMask.unsqueeze(0).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor maskedHands = expandedFullHand * expandedMask;
        Tensor embeddings = EncodeCardCounts(maskedHands);
        embeddings.MoveToOuterDisposeScope();
        return embeddings;
    }


    static Tensor BuildSelectedHandEmbeddings(Tensor fullHand, Tensor handMask, Tensor selectedHandIndices)
    {
        using var scope = NewDisposeScope();

        int batchSize = (int)fullHand.size(0);
        int selectedMoveCount = (int)selectedHandIndices.size(1);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, selectedMoveCount, GameData.HandSize);
        Tensor selectedMasks = handMask
            .index_select(dim: 0, index: selectedHandIndices.view(-1))
            .view(batchSize, selectedMoveCount, GameData.HandSize);
        Tensor maskedHands = expandedFullHand * selectedMasks;
        Tensor embeddings = EncodeCardCounts(maskedHands);
        embeddings.MoveToOuterDisposeScope();
        return embeddings;
    }


    static Tensor EncodeCardCounts(Tensor cardSet)
    {
        using var scope = NewDisposeScope();

        Tensor cardSetIndices = cardSet.to_type(ScalarType.Int64);
        int cardDim = (int)cardSetIndices.Dimensions - 1;
        Tensor countsWithNull = functional.one_hot(cardSetIndices, CardCountWidth + 1)
            .to_type(ScalarType.Float32)
            .sum(dim: cardDim);
        Tensor counts = countsWithNull.narrow(dim: countsWithNull.Dimensions - 1, start: 1, length: CardCountWidth);
        counts.MoveToOuterDisposeScope();
        return counts;
    }


    static Tensor EncodeJokerCounts(Tensor jokerSet)
    {
        using var scope = NewDisposeScope();

        Tensor jokerIndices = jokerSet.to_type(ScalarType.Int64);
        int jokerDim = (int)jokerIndices.Dimensions - 1;
        Tensor countsWithNull = functional.one_hot(jokerIndices, JokerCountWidth + 1)
            .to_type(ScalarType.Float32)
            .sum(dim: jokerDim);
        Tensor counts = countsWithNull.narrow(dim: countsWithNull.Dimensions - 1, start: 1, length: JokerCountWidth);
        counts.MoveToOuterDisposeScope();
        return counts;
    }


    static Tensor EncodeCountState(Tensor remainingHands, Tensor remainingDiscards)
    {
        using var scope = NewDisposeScope();

        Tensor combinedIndex = remainingHands.mul(4).add(remainingDiscards).to_type(ScalarType.Int64);
        Tensor encoded = functional.one_hot(combinedIndex, CountWidth).to_type(ScalarType.Float32);
        encoded.MoveToOuterDisposeScope();
        return encoded;
    }


    static Tensor EncodeStorePriceEmbeddings(Tensor price, Embedding storePriceEmbedding)
    {
        using var scope = NewDisposeScope();

        Tensor clampedPrices = price.to_type(ScalarType.Int64).clamp(0, MaxStorePrice);
        Tensor encoded = storePriceEmbedding.forward(clampedPrices);
        encoded.MoveToOuterDisposeScope();
        return encoded;
    }


    static Tensor EncodeMoneyState(Tensor money)
    {
        using var scope = NewDisposeScope();

        Tensor moneyValues = money.to_type(ScalarType.Int64).clamp_min(0);
        Tensor exactMask = moneyValues.le(35).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor exactIndices = moneyValues.clamp(0, 35);
        Tensor exactEmbedding = functional.one_hot(exactIndices, 36).to_type(ScalarType.Float32) * exactMask;

        Tensor bucketMask = moneyValues.gt(35).logical_and(moneyValues.le(100)).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor bucketIndices = moneyValues.sub(35).div(5).clamp(0, 13).to_type(ScalarType.Int64);
        Tensor bucketEmbedding = functional.one_hot(bucketIndices, 14).to_type(ScalarType.Float32) * bucketMask;

        Tensor overflowEmbedding = moneyValues.gt(100).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor encoded = cat([exactEmbedding, bucketEmbedding, overflowEmbedding], dim: -1);
        encoded.MoveToOuterDisposeScope();
        return encoded;
    }


    static Tensor EncodeRoundState(Tensor round)
    {
        using var scope = NewDisposeScope();

        Tensor roundValues = round.to_type(ScalarType.Int64);
        Tensor validMask = roundValues.greater_equal(1).logical_and(roundValues.le(RoundEmbeddingWidth)).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor roundIndices = roundValues.sub(1).clamp(0, RoundEmbeddingWidth - 1);
        Tensor encoded = functional.one_hot(roundIndices, RoundEmbeddingWidth).to_type(ScalarType.Float32) * validMask;
        encoded.MoveToOuterDisposeScope();
        return encoded;
    }


    static Tensor EncodeStageState(Tensor stage)
    {
        using var scope = NewDisposeScope();

        Tensor stageValues = stage.to_type(ScalarType.Int64).clamp(0, StageEmbeddingWidth - 1);
        Tensor encoded = functional.one_hot(stageValues, StageEmbeddingWidth).to_type(ScalarType.Float32);
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
