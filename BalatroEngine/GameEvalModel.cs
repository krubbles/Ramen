namespace Ramen.AI;

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
        StateOtherStateWidth = 12,
        MoveOtherStateWidth = 12;

    public const int
        StateProcessorInputWidth = RemainingDeckEmbedWidth + FullHandEmbedWidth + StateOtherStateWidth,
        MoveEvaluatorInputWidth = RemainingHandEmbedWidth + PlayedHandEmbedWidth + MoveOtherStateWidth;
    


    private readonly Embedding _embedRemainingDeck = Embedding(53, RemainingDeckEmbedWidth);
    private readonly Embedding _embedFullHand = Embedding(53, FullHandEmbedWidth);
    private readonly Embedding _embedRemainingHand = Embedding(53, RemainingHandEmbedWidth);
    private readonly Embedding _embedPlayedHand = Embedding(53, PlayedHandEmbedWidth);

    public readonly Sequential StateProcessor;
    public readonly Sequential MoveEvaluator;
    public readonly Sequential MovePreProcessor;

    public GameEvalModel() : base(nameof(GameEvalModel))
    {

        StateProcessor = Sequential(
            Linear(StateProcessorInputWidth, 256),
            ReLU(),
            Linear(256, 128)
        );

        MovePreProcessor = Sequential(
            Linear(MoveEvaluatorInputWidth, 128)
        );

        MoveEvaluator = Sequential(
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 1)
        );


        RegisterComponents();
    }

    public static Tensor EmbedCardSet(Embedding embedding, Tensor hand)
    {
        Tensor embeddedHand = embedding.forward(hand).sum(dim: hand.Dimensions - 1);
        Tensor result = embeddedHand.relu_();
        return result;
    }

    Tensor ProcessState(Tensor processedHand, Tensor processedDeck, Tensor otherState)
    {
        Tensor input = concat([processedHand, processedDeck, otherState], dim: otherState.Dimensions - 1);
        Tensor output = StateProcessor.forward(input);
        return output;
    }

    public Tensor ProcessState(GameStateTensors gameState)
    {
        Tensor processedHand = EmbedCardSet(_embedFullHand, gameState.FullHand);
        Tensor processedDeck = EmbedCardSet(_embedRemainingDeck, gameState.RemainingDeck);
        return ProcessState(processedHand, processedDeck, gameState.OtherState);
    }

    public Tensor PreProcessMoves(MoveTensors moves)
    {
        Tensor processedRemainingHand = EmbedCardSet(_embedRemainingHand, moves.RemainingHand);
        Tensor processedPlayedHand = EmbedCardSet(_embedPlayedHand, moves.PlayedHand);
        Tensor input = concat([processedRemainingHand, processedPlayedHand, moves.OtherState], dim: moves.OtherState.Dimensions - 1);
        return MovePreProcessor.forward(input);
    }

    public Tensor GetMoveLogits(Tensor processedState, MoveTensors moves)
    {
        Tensor preProcessedMoves = PreProcessMoves(moves);
        Tensor logits = MoveEvaluator.forward(preProcessedMoves + processedState.unsqueeze(1).expand(preProcessedMoves.shape)).squeeze_(2);
        return logits;
    }
}


class ResidualMLP : Module<Tensor, Tensor>
{
    private ModuleList<Linear> upLayers = new();
    private ModuleList<Linear> downLayers = new();

    private ModuleList<LayerNorm> norms = new();

    private ModuleList<GELU> activationsA = new();
    private ModuleList<GELU> activationsB = new();

    public ResidualMLP(int size, int depth) : base("ResidualMLP")
    {
        for (int i = 0; i < depth; ++i)
        {
            int factor = 1;
            upLayers.append(Linear(size, size * factor));
            downLayers.append(Linear(size * factor, size));
            activationsA.append(GELU());
            activationsB.append(GELU());
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

public class SwiGLUFeedForward : Module<Tensor, Tensor>
{
    private readonly Linear w1; // Gate projection
    private readonly Linear w2; // Up projection
    private readonly Linear w3; // Down projection (optional, usually follows SwiGLU)

    public SwiGLUFeedForward(string name, long inputDim, long hiddenDim) : base(name)
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