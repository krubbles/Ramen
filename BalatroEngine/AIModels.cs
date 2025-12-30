namespace BalatroAI
{
    using System.Collections.Generic;
    using System.Linq;
    using TorchSharp;
    using TorchSharp.Modules;
    using static TorchSharp.torch;
    using static TorchSharp.torch.nn;

    // Scaffolding container for the collection of AI models that make up an AI.
    // Currently only contains the MoveSelectorModule but provides delegation methods
    // so existing code can interact with the models through this single object.
    public class AIModels
    {
        public BestMovePredictor Policy { get; }
        public EvaluationModule Evaluation { get; }

        public AIModels()
        {
            Policy = new BestMovePredictor();
            Evaluation = new EvaluationModule(Policy);
        }

        public IEnumerable<Parameter> parameters()
        {
            // return parameters from both modules
            return Policy.parameters().Concat(Evaluation.parameters());
        }

        public IEnumerable<(string, Parameter)> named_parameters()
        {
            return Policy.named_parameters().Concat(Evaluation.named_parameters());
        }

        public Tensor EmbedCards(Tensor cards) => Policy.EmbedCards(cards);

        public Tensor GetCardUseRewards(Tensor fullHand, Tensor otherState, Tensor inUseMask)
            => Policy.GetCardUseRewards(fullHand, otherState, inUseMask);

        public Tensor GetExpectedReward(Tensor fullHand, Tensor otherState, Tensor inUseMask)
            => Evaluation.forward(fullHand, otherState, inUseMask);
    }

    public class EvaluationModule : Module
    {
        private readonly Embedding _embedCard;

        private readonly Sequential _mlp;
        private readonly Sequential _fullHandProcessor;
        private readonly Sequential _workingHandProcessor;

        public EvaluationModule(BestMovePredictor moveSelector) : base("EvaluationModule")
        {
            _embedCard = Embedding(53, BestMovePredictor.EmbeddedCardWidth);

            _fullHandProcessor = Sequential(
                ReLU(),
                new ResidualMLP(BestMovePredictor.EmbeddedCardWidth, 1));
            
            _workingHandProcessor = Sequential(
                ReLU(),
                new ResidualMLP(BestMovePredictor.EmbeddedCardWidth, 1));


            _mlp = Sequential(
                ReLU(),
                new ResidualMLP(BestMovePredictor.EmbeddedCardWidth + BestMovePredictor.OtherStateWidth, 1),
                ReLU(),
                Linear(BestMovePredictor.EmbeddedCardWidth + BestMovePredictor.OtherStateWidth, 32),
                ReLU(),
                Linear(32, 1)
            );

            RegisterComponents();
        }

        public Tensor forward(Tensor hand, Tensor otherState, Tensor inUseMask)
        {
            Tensor embeddedHand = _embedCard.forward(hand);
            Tensor compressedHand = _fullHandProcessor.forward(embeddedHand.sum(dim: 1));
            Tensor compressedWorkingHand = _workingHandProcessor.forward(embeddedHand.mul(inUseMask.unsqueeze(2)).sum(dim: 1));
            Tensor output = _mlp.forward(concat([compressedHand - compressedWorkingHand, otherState], dim: 1));
            return output;
        }

        public Tensor ProcessHand(Tensor hand)
        {
            return hand + _fullHandProcessor.forward(hand);
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
                upLayers.append(Linear(size, size));
                downLayers.append(Linear(size, size));
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
