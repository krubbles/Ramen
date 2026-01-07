namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class GameEvalModel : Module
{
    private readonly Embedding _embedCard;

    public readonly Sequential HandProcessor;
    public readonly Sequential OtherStateProcessor;
    public readonly Sequential FinalNetwork;

    public const int EmbeddedCardWidth = 64;
    public const int FinalNetworkWidth = EmbeddedCardWidth * 3 + OtherStateWidth;
    public const int OtherStateWidth = 14;

    public GameEvalModel() : base(nameof(GameEvalModel))
    {
        _embedCard = Embedding(53, EmbeddedCardWidth);

        FinalNetwork = Sequential(
            Linear(OtherStateWidth, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 1)
        );

        RegisterComponents();
    }

    public Tensor ProcessHand(Tensor hand)
    {
        return null;
        Tensor embeddedHand = _embedCard.forward(hand).sum(dim: hand.Dimensions - 1);
        Tensor result = embeddedHand.relu_();
        return result;
    }

    public Tensor GetPredictedRewardDistribution(Tensor processedHand, Tensor processedDeck, Tensor processedFullHand, Tensor otherState)
    {
        // Tensor input = concat([processedHand, processedDeck, processedFullHand, otherState], dim: otherState.Dimensions - 1);
        Tensor output = FinalNetwork.forward(otherState);
        return output;
    }

    static Tensor RemapVariance(Tensor output)
    {
        output = (output + sqrt(output.square() + 1)).square() * 0.25f;
        return output;
    }

    public Tensor forward(GameStateTensors gameState)
    {
        Tensor processedHand = ProcessHand(gameState.Hand);
        Tensor processedFullHand = ProcessHand(gameState.FullHand);
        Tensor processedDeck = ProcessHand(gameState.RemainingDeck);
        Tensor output = GetPredictedRewardDistribution(processedHand, processedDeck, processedFullHand, gameState.OtherState);
        return output;
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