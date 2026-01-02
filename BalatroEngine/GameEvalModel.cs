namespace BalatroAI
{
    using System.Collections.Generic;
    using System.Linq;
    using TorchSharp;
    using TorchSharp.Modules;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;


    public class GameEvalModel : Module
    {
        private readonly Embedding _embedCard;

        public readonly Sequential HandProcessor;
        public readonly Sequential OtherStateProcessor;
        public readonly Sequential FinalNetwork;

        public const int EmbeddedCardWidth = 128 - OtherStateWidth;
        public const int FinalNetworkWidth = EmbeddedCardWidth + OtherStateWidth;
        public const int OtherStateWidth = 12;

        public GameEvalModel() : base(nameof(GameEvalModel))
        {
            _embedCard = Embedding(53, EmbeddedCardWidth);

            FinalNetwork = Sequential(
                Linear(FinalNetworkWidth, 128),
                ReLU(),
                Linear(128, 16),
                ReLU(),
                Linear(16, 1)
            );

            RegisterComponents();
        }

        public Tensor ProcessHand(Tensor hand)
        {
            Tensor embeddedHand = _embedCard.forward(hand).sum(dim: 1);
            Tensor result = embeddedHand.relu_();
            return result;
        }

        public Tensor GetPredictedRewardDistribution(Tensor processedHand, Tensor otherState)
        {
            Tensor input = concat([processedHand, otherState], dim: 1);
            Tensor output = FinalNetwork.forward(input);
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
            Tensor output = GetPredictedRewardDistribution(processedHand, gameState.OtherState);
            return output;
        }
    }


    class ResidualMLP : Module<Tensor, Tensor>
    {
        private ModuleList<Linear> upLayers = new();
        private ModuleList<Linear> downLayers = new();

        private ModuleList<LayerNorm> norms = new();
        private ModuleList<GELU> activations = new();

        public ResidualMLP(int size, int depth) : base("ResidualMLP")
        {
            for (int i = 0; i < depth; ++i)
            {
                int factor = 2;
                upLayers.append(Linear(size, size * factor));
                downLayers.append(Linear(size * factor, size));
                activations.append(GELU());
                norms.append(LayerNorm(size));
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            for (int i = 0; i < upLayers.Count; i++)
            {
                Tensor normed = norms[i].forward(x);
                Tensor up = upLayers[i].forward(x);
                Tensor activated = activations[i].forward(up);
                Tensor down = downLayers[i].forward(activated);
                x = x / upLayers.Count + down;
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
}
