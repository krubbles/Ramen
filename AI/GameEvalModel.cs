namespace Ramen.AI;

using System.Linq;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class GameEvalModel : Module
{
    public const int
        RemainingDeckEmbedWidth = 128,
        FullHandEmbedWidth = 128,
        RemainingHandEmbedWidth = 64,
        PlayedHandEmbedWidth = 64,
        StateOtherStateWidth = 13,
        MoveOtherStateWidth = 13,
        Tiers = 1;

    public const int
        StateProcessorInputWidth = RemainingDeckEmbedWidth + FullHandEmbedWidth + StateOtherStateWidth,
        MoveEvaluatorInputWidth = RemainingHandEmbedWidth + PlayedHandEmbedWidth + MoveOtherStateWidth;

    public const int StateWidth = 64;
    public const int MoveWidth = 192;
    private readonly Embedding _embedRemainingDeck = Embedding(53, MoveWidth);
    private readonly Embedding _embedFullHand = Embedding(53, StateWidth);
    private readonly Embedding _embedRemainingHand = Embedding(53, MoveWidth);
    private readonly Embedding _embedPlayedHand = Embedding(53, MoveWidth);
    private readonly Embedding _embedHandsAndDiscardsMove = Embedding(25, MoveWidth);
    private readonly Embedding _embedHandsAndDiscardsState = Embedding(25, StateWidth);
    private readonly Linear _scoreUpscaleState = Linear(1, StateWidth, false);
    private readonly Linear _scoreUpscaleMove = Linear(1, MoveWidth, false);

    public readonly Sequential StateProcessor;
    public readonly Sequential ForcastPolicy;
    public readonly Sequential UseHandPolicy;
    public readonly Sequential UseHandMovePreProcessor;

    public GameEvalModel() : base(nameof(GameEvalModel))
    {

        StateProcessor = Sequential(
            new ResidualMLP(StateWidth, 3),
            Linear(StateWidth, MoveWidth)
        );

        UseHandMovePreProcessor = Sequential(
            Linear(MoveEvaluatorInputWidth, 128)
        );

        UseHandPolicy = Sequential(
            new Residual(new SwiGLUFeedForward(MoveWidth, 128)),
            new Residual(new SwiGLUFeedForward(MoveWidth, 128)),
            ReLU(),
            Linear(MoveWidth, 1)
        );

        ForcastPolicy = Sequential(
            Linear(128, 128),
            ReLU(),
            Linear(128, Tiers)
        );

        RegisterComponents();
    }

    public static Tensor EmbedCardSet(Embedding embedding, Tensor hand)
    {
        Tensor embeddedHand = embedding.forward(hand).sum(dim: hand.Dimensions - 1) / hand.size((int)hand.Dimensions - 1);
        return embeddedHand;
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
        Tensor hand = EmbedCardSet(_embedFullHand, gameState.FullHand);
        Tensor deck = EmbedCardSet(_embedRemainingDeck, gameState.RemainingDeck);
        Tensor score = _scoreUpscaleState.forward(gameState.Score);
        Tensor handsAndDiscards = _embedHandsAndDiscardsState.forward(gameState.HandsAndDiscards);
        return score + hand + handsAndDiscards;
    }

    Tensor EmbedMove(MoveTensors move)
    {
        Tensor remainingHand = EmbedCardSet(_embedRemainingHand, move.RemainingHand);
        Tensor playedHand = EmbedCardSet(_embedPlayedHand, move.PlayedHand);
        Tensor score = _scoreUpscaleMove.forward(move.Score.unsqueeze(2));
        Tensor handsAndDiscards = _embedHandsAndDiscardsMove.forward(move.HandsAndDiscards);
        return score + remainingHand;
    }

    public Tensor ProcessMove(Tensor embeddedMove, Tensor processedState)
    {
        return UseHandPolicy.forward(embeddedMove);
    }

    public Tensor ProcessMove(MoveTensors move, Tensor processedState)
    {
        Tensor embeddedMove = EmbedMove(move);
        return UseHandPolicy.forward(embeddedMove + processedState.unsqueeze(1).expand(embeddedMove.shape));
    }

    public Tensor GetForcastLogits(Tensor processedState)
    {
        return ForcastPolicy.forward(processedState);
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
    private readonly Linear w3; // Down projection (optional, usually follows SwiGLU)

    public SwiGLUFeedForward(long inputDim, long hiddenDim) : base(nameof(SwiGLUFeedForward))
    {
        // w1 and w2 are the two projections for the GLU
        w1 = Linear(inputDim, hiddenDim, hasBias: false);
        w2 = Linear(inputDim, hiddenDim, hasBias: false);
        w3 = Linear(hiddenDim, inputDim, hasBias: false);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        var swishGate = functional.silu(w1.forward(x));
        var intermediate = swishGate * w2.forward(x);
        return w3.forward(intermediate);
    }
}