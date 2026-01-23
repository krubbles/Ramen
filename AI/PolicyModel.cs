namespace Ramen.AI;

using Ramen.Game;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

/// High level policy network topology:
/// - Input: Embedded GameState + Embedded Moves (<see cref="GameStateTensors"/> + <see cref="UseHandTensors"/>)
/// - Always assume there is a batch dimension at dim 0.
/// - Process <see cref="GameStateTensors.RemainingDeck"/> using card set processing described below.
/// - Process <see cref="GameStateTensors.FullHand"/>
/// - Embed <see cref="GameStateTensors.HandsAndDiscards"/> using a learnable embedding
/// - Include <see cref="GameStateTensors.Score"/> as a single scalar input logit
/// - Concat all the above vectors.
/// - Pass through a residual MLP to get a processed state vector.
/// - For each move (discard and play for same hand are grouped):
///    - Process <see cref="UseHandTensors.PlayedHand"/> 
///    - Process <see cref="UseHandTensors.RemainingHand"/>
///    - Include <see cref="UseHandTensors.Score"/> as a single scalar input logit
///    - Concat the above vectors + the processed state vectors.
///    - Pass through a residual MLP to get 2 logits: play logit, and discard logit.
/// - Use a view to collapse the Nx2 logits from all the [play, discard] pairs into a single policy logits tensor.
/// - Return the output policy tensor.
/// Card set processing: 
/// - Embed each card into a length (13 + 4) vector, 1-hot encoding rank and suit.
/// - Generate a pooled length (13 + 4) * 2 vector, the concat of the sum and max over the embedded cards.
/// - Generate 4 similar pooled vectors for each suit, where it only pools over cards of that suit.
/// - Process the concat of [general pooled vector, suit pooled vector] with a small sequential NN to get four processed vectors, then sum them.
/// - The resulting vector is the output.
/// - This process makes the NN mostly suit agnostic, improving generalization.

/// <summary>
/// The policy network used for move evaluation.
/// </summary>
public class PolicyModel : Module
{
    public const int
        RankCount = 13,
        SuitCount = 4,
        CardEmbedWidth = RankCount + SuitCount,
        CardPoolWidth = CardEmbedWidth * 2,
        CardSuitConcatWidth = CardPoolWidth * 2,
        CardSuitProcessorHiddenWidth = 128,
        CardSetWidth = 64,
        HandsAndDiscardsEmbedWidth = 32,
        StateWidth = 192,
        MoveWidth = 128,
        StateDepth = 2,
        MoveDepth = 2,
        Tiers = 1;

    public const int
        StateProcessorInputWidth = CardSetWidth + CardSetWidth + HandsAndDiscardsEmbedWidth + 1,
        MoveEvaluatorInputWidth = CardSetWidth + CardSetWidth + 1 + StateWidth;

    readonly Embedding _embedHandsAndDiscards = Embedding(25, HandsAndDiscardsEmbedWidth);
    readonly Sequential _cardSetSuitProcessor;

    public readonly Sequential StateProcessor;
    public readonly Sequential UseHandEvaluator;
    public readonly Sequential ForcastPolicy;

    public PolicyModel() : base(nameof(PolicyModel))
    {
        _cardSetSuitProcessor = Sequential(
            Linear(CardSuitConcatWidth, CardSuitProcessorHiddenWidth),
            ReLU(),
            Linear(CardSuitProcessorHiddenWidth, CardSetWidth)
        );

        StateProcessor = Sequential(
            Linear(StateProcessorInputWidth, StateWidth),
            ReLU(),
            new ResidualMLP(StateWidth, StateDepth)
        );

        UseHandEvaluator = Sequential(
            Linear(MoveEvaluatorInputWidth, MoveWidth),
            ReLU(),
            new ResidualMLP(MoveWidth, MoveDepth),
            Linear(MoveWidth, 2)
        );

        RegisterComponents();
    }

    Tensor ProcessCardSet(Tensor cardSet)
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
            Tensor suitProcessed = _cardSetSuitProcessor.forward(suitConcat);
            processedSum = processedSum is null ? suitProcessed : processedSum + suitProcessed;
        }

        return processedSum;
    }

    public Tensor ProcessState(Tensor embeddedState)
    {
        return StateProcessor.forward(embeddedState);
    }

    public Tensor ProcessState(GameStateTensors gameState)
    {
        return ProcessState(EmbedState(gameState));
    }

    Tensor EmbedState(GameStateTensors gameState)
    {
        Tensor hand = ProcessCardSet(gameState.FullHand);
        Tensor deck = ProcessCardSet(gameState.RemainingDeck);
        Tensor handsAndDiscards = _embedHandsAndDiscards.forward(gameState.HandsAndDiscards);
        return cat([deck, hand, handsAndDiscards, gameState.Score], dim: -1);
    }

    Tensor ProcessUseHandTensors(UseHandTensors move)
    {
        Tensor remainingHand = ProcessCardSet(move.RemainingHand);
        Tensor playedHand = ProcessCardSet(move.PlayedHand);
        Tensor score = move.Score.unsqueeze(2);
        return cat([playedHand, remainingHand, score], dim: -1);
    }
    
    public Tensor GetPolicyLogits(UseHandTensors moveData, Tensor processedState)
    {
        Tensor processedUseHandTensors = ProcessUseHandTensors(moveData);
        Tensor expandedState = processedState.unsqueeze(1).expand(processedUseHandTensors.shape[0], processedUseHandTensors.shape[1], processedState.shape[1]);
        Tensor moveInput = cat([processedUseHandTensors, expandedState], dim: -1);
        Tensor playAndDiscardLogits = UseHandEvaluator.forward(moveInput);
        Tensor moveLogits = playAndDiscardLogits.view([playAndDiscardLogits.size(0), -1]);
        return moveLogits;
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
