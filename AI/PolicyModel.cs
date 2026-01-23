namespace Ramen.AI;

using Ramen.Game;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


/// <summary>
/// The policy network used for move evaluation.
/// </summary>
public class PolicyModel : Module
{
    public const int
        RankCount = 13,
        SuitCount = 4;
    
    const int RankEmbedWidth = 64;
    readonly Embedding _rankEmbedding = Embedding(RankCount, RankEmbedWidth);

    const int UseHandCardSetOutputWidth = 64;
    readonly Sequential _useHandCardSetProcessor = 
        Sequential(
            new Bias(RankEmbedWidth * 2),
            ReLU(),
            Linear(RankEmbedWidth * 2, UseHandCardSetOutputWidth)
        );

    const int RemainingDeckCardSetWidth = 0;
    readonly Sequential _remainingDeckCardSetProcessor = 
        Sequential(
            new Bias(RankEmbedWidth * 2),
            ReLU(),
            Linear(RankEmbedWidth * 2, RemainingDeckCardSetWidth)
        );

    public const int HandsAndDiscardsEmbedWidth = 25;
    readonly Embedding _embedHandsAndDiscards = Embedding(25, HandsAndDiscardsEmbedWidth);

    public const int StateScoreEmbedWidth = 1;
    public const int StateProcessorInputWidth = RemainingDeckCardSetWidth + HandsAndDiscardsEmbedWidth + 1;
    public const int StateProcessorOutputWidth = 64;
    readonly Sequential _stateProcessor = 
        Sequential(
            Linear(StateProcessorInputWidth, StateProcessorOutputWidth),
            ReLU()
        );

    public const int UseHandScoreEmbedWidth = 1;
    public const int UseHandProcessorInputWidth = StateProcessorOutputWidth + UseHandCardSetOutputWidth + UseHandScoreEmbedWidth;
    public const int UseHandProcessorHiddenWidth = 128;
    readonly Sequential _useHandProcessor = 
        Sequential(
            Linear(UseHandProcessorInputWidth, UseHandProcessorHiddenWidth),
            ReLU(),
            Linear(UseHandProcessorHiddenWidth, 2)
        );

    public PolicyModel() : base(nameof(PolicyModel))
    {
        RegisterComponents();
    }

    static readonly Dictionary<(Suit suit, int cardIndex), Tensor> _playedHandSelectionIndices = [];
    static readonly Dictionary<(Suit suit, int cardIndex), Tensor> _remainingHandSelectionIndices = [];

    static PolicyModel()
    {
        // Input: A (batch x cardCount x 1) rank index vector and a (batch x cardCount x 10) suit vector
        // - Suit vector format is (0, isDiamond, isClub, isHeart, isSpade)
        // Output: A batch x playedHandCount x rankEmbedWidth tensor for each suit. 
        // Steps:
        // 1. Embed the ranks
        // 2. Outer product the rank embeddings with the suit vectors (bce, bcs -> bcse)
        // 3. Flatten to (batch x cardCount * 5 x rankEmbedWidth). This lets us select cards of a given suit easily.
        // 4. We can generate sum of the embeddings of each card of a particular suit in each possible hand by 
        //    generating 5 index vectors, one for each card position in the played hand (for hands size < 5, some indices will be zero).
        // So thats what this does. 
        for (Suit suit = Suit.Diamond; suit <= Suit.Spade; ++suit)
        {
            for (int usedCardIndex = 0; usedCardIndex < 5; ++usedCardIndex)
            {
                int[] indices = new int[Combinatorics.CalculateCombinationCount(setSize: 8, maxSubsetSize: 5, minSubsetSize: 1)];
                int indicesIndex = 0;
                foreach (int[] cardIndices in Combinatorics.GetCombinations(setSize: 8, maxSubsetSize: 5, minSubsetSize: 1))
                {
                    indices[indicesIndex++] = 
                        usedCardIndex < cardIndices.Length ? 
                        cardIndices[usedCardIndex] * 5 + (int)suit : 
                        0;
                }
                for (int i = 0; i < 5; ++i)
                    indices[i] = i + (int)suit * 5;
                Tensor tensorIndices = tensor(indices, ScalarType.Int32);
                _playedHandSelectionIndices.Add((suit, usedCardIndex), tensorIndices);
            }
        }
    }

    Tensor FullHandToUsedHands(Tensor fullHand)
    {
        Tensor fullHandRanks = _rankEmbedding.forward(fullHand.select(2, 0)); // BatchSize x CardCount x RankEmbedWidth
        Tensor fullHandSuits = functional.one_hot(fullHand.select(2, 1), SuitCount + 1).to_type(ScalarType.Float32); // BatchSize x CardCount x SuitCount (including null)
        
        int useableHandCount = Combinatorics.CalculateCombinationCount(setSize: 8, maxSubsetSize: 5, minSubsetSize: 1);
        Tensor embeddedFullHand = fullHandRanks.sum(dim: 1); // BatchSize x RankEmbedWidth
        Tensor embeddedFullHandExpanded = embeddedFullHand.unsqueeze(1).expand(embeddedFullHand.size(0), useableHandCount, embeddedFullHand.size(1)); // BatchSize x UseableHandCount x RankEmbedWidth
        
        Tensor perSuitEmbeds = einsum("bce, bcs -> bcse", fullHandRanks, fullHandSuits); // BatchSize x CardCount x SuitCount (including null) x RankEmbedWidth
        Tensor perSuitEmbedsFlattened = perSuitEmbeds.view([perSuitEmbeds.size(0), perSuitEmbeds.size(1) * perSuitEmbeds.size(2), -1]); // BatchSize x CardCount x SuitCount (including null) * RankEmbedWidth
        Tensor[] perSuitDataArray = new Tensor[4]; // BatchSize x UseableHandCount x SuitProcessOutputWidth
        for (Suit suit = Suit.Diamond; suit <= Suit.Spade; ++suit)
        {
            Tensor[] perCardInPlayedHandInputData = new Tensor[5]; // SuitCount + 1 (null)
            for (int cardIndex = 0; cardIndex < perCardInPlayedHandInputData.Length; ++cardIndex)
            {
                Tensor indices = _playedHandSelectionIndices[(suit, cardIndex)];
                perCardInPlayedHandInputData[cardIndex] = perSuitEmbedsFlattened.index_select(1, indices); // BatchSize x PlayedHandCount x RankEmbedWidth
            }
            Tensor usedHandSuitData = stack(perCardInPlayedHandInputData, dim: 2).sum(dim: 2); // BatchSize x UseableHandCount x RankEmbedWidth
            Tensor cardSetSuitInput = concat([embeddedFullHandExpanded, usedHandSuitData], dim: -1); // BatchSize x UseableHandCount x (RankEmbedWidth * 2)
            perSuitDataArray[(int)suit - 1] = _useHandCardSetProcessor.forward(cardSetSuitInput); // BatchSize x UseableHandCount x SuitProcessOutputWidth
        }
        Tensor result = stack(perSuitDataArray, dim: 2).sum(dim: 2); // BatchSize x RankEmbedWidth
        return result;
    }

    Tensor ProcessCardSet(Tensor cardSet, Sequential processor)
    {
        int cardDim = (int)cardSet.Dimensions - 2;
        Tensor cardRanks = cardSet.select(cardSet.Dimensions - 1, 0);
        Tensor cardSuits = cardSet.select(cardSet.Dimensions - 1, 1);

        Tensor rankIndex = cardRanks.clamp(0, RankCount - 1);
        Tensor suitIndex = cardSuits.clamp(0, SuitCount - 1);

        Tensor rankOneHot = functional.one_hot(rankIndex, RankCount).to_type(ScalarType.Float32);
        Tensor suitOneHot = functional.one_hot(suitIndex, SuitCount).to_type(ScalarType.Float32);
        Tensor cardEmbed = cat([rankOneHot, suitOneHot], dim: -1);

        Tensor[] suitSums = new Tensor[SuitCount];
        Tensor[] suitMaxes = new Tensor[SuitCount];

        Tensor processedSum = null;
        for (Suit suit = Suit.Diamond; suit <= Suit.Spade; ++suit)
        {
            Tensor suitMask = cardSuits.eq((int)suit);
            Tensor suitMaskFloat = suitMask.unsqueeze(-1).to_type(cardEmbed.dtype);
            Tensor suitEmbed = cardEmbed * suitMaskFloat;
            Tensor suitSum = suitEmbed.sum(dim: cardDim);
            Tensor suitMax = suitEmbed.amax(dims: [cardDim]);
            suitSums[(int)suit - 1] = suitSum;
            suitMaxes[(int)suit - 1] = suitMax;
        }

        Tensor pooledSum = stack(suitSums, dim: 0).sum(dim: 0);
        Tensor pooledMax = stack(suitMaxes, dim: 0).amax(dims: [0]);
        Tensor pooledGeneral = cat([pooledSum, pooledMax], dim: -1);

        for (Suit suit = Suit.Diamond; suit <= Suit.Spade; ++suit)
        {
            Tensor suitConcat = cat([pooledGeneral, suitSums[(int)suit - 1], suitMaxes[(int)suit - 1]], dim: -1);
            Tensor suitProcessed = processor.forward(suitConcat);
            processedSum = processedSum is null ? suitProcessed : processedSum + suitProcessed;
        }

        return processedSum;
    }

    Tensor ProcessState(Tensor embeddedState)
    {
        return _stateProcessor.forward(embeddedState);
    }

    Tensor ProcessState(GameStateTensors gameState)
    {
        return ProcessState(EmbedState(gameState));
    }

    Tensor EmbedState(GameStateTensors gameState)
    {
        Tensor handsAndDiscards = _embedHandsAndDiscards.forward(gameState.HandsAndDiscards);
        return cat([handsAndDiscards, gameState.Score], dim: -1);
    }
    
    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        Tensor processedUseHandTensors = FullHandToUsedHands(gameStateTensors.FullHand);
        Tensor processedState = ProcessState(gameStateTensors);
        Tensor processedStateExpanded = processedState.unsqueeze(1).expand(processedUseHandTensors.shape);
        Tensor useHandInputs = cat([processedStateExpanded, processedUseHandTensors, useHandTensors.Score.unsqueeze(2)], dim: -1);
        Tensor moveLogits = _useHandProcessor.forward(useHandInputs).view([processedState.size(0), -1]);
        return moveLogits;
    }
}

class Bias : Module<Tensor, Tensor>
{
    Parameter _bias;

    public Bias(int size) : base("Bias")
    {
        _bias = Parameter(zeros(size));
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return input + _bias;
    }
}

class ResidualMLP : Module<Tensor, Tensor>
{
    private ModuleList<Linear> upLayers = new();
    private ModuleList<Linear> downLayers = new();

    private ModuleList<LayerNorm> norms = new();

    private ModuleList<ReLU> activationsA = new();
    private ModuleList<ReLU> activationsB = new();

    public ResidualMLP(int size, int depth) : base("ResidualMLP")
    {
        for (int i = 0; i < depth; ++i)
        {
            int factor = 1;
            upLayers.append(Linear(size, size * factor));
            downLayers.append(Linear(size * factor, size));
            activationsA.append(ReLU());
            activationsB.append(ReLU());
            norms.append(LayerNorm(size));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        for (int i = 0; i < upLayers.Count; i++)
        {
            Tensor normed = norms[i].forward(x);
            Tensor activated = activationsA[i].forward(normed);
            Tensor up = upLayers[i].forward(activated);
            Tensor activated2 = activationsB[i].forward(up);
            Tensor down = downLayers[i].forward(activated2);
            x += down;
        }
        return x;
    }
}

public class Residual : Module<Tensor, Tensor>
{
    public Module<Tensor, Tensor> F;

    public Residual(Module<Tensor, Tensor> f) : base(nameof(Residual))
    {
        F = f;

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return input + F.forward(input);
    }
}

public class SwiGLUFeedForward : Module<Tensor, Tensor>
{
    private readonly Linear w1; // Gate projection
    private readonly Linear w2; // Up projection
    private readonly Linear w3; // Down projection
    private readonly LayerNorm ln;

    public SwiGLUFeedForward(long inputDim, long hiddenDim) : base(nameof(SwiGLUFeedForward))
    {
        ln = LayerNorm(inputDim);

        // w1 and w2 are the two projections for the GLU
        w1 = Linear(inputDim, hiddenDim, hasBias: false);
        w2 = Linear(inputDim, hiddenDim, hasBias: false);
        w3 = Linear(hiddenDim, inputDim, hasBias: false);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = ln.forward(x);
        var swishGate = functional.silu(w1.forward(x));
        var intermediate = swishGate * w2.forward(x);
        return w3.forward(intermediate);
    }
}
