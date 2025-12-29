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
        private readonly BestMovePredictor _moveSelector;
        private readonly Sequential _mlp;

        public EvaluationModule(BestMovePredictor moveSelector) : base("EvaluationModule")
        {
            _moveSelector = moveSelector;

            // input width = EmbeddedCardWidth + OtherStateWidth
            int inputWidth = BestMovePredictor.EmbeddedCardWidth * 2 + BestMovePredictor.OtherStateWidth;
            int hidden = 128;

            _mlp = Sequential(
                Linear(inputWidth, hidden), ReLU(),
                Linear(hidden, 1)
            );

            RegisterComponents();
        }

        public Tensor forward(Tensor fullHand, Tensor otherState, Tensor inUseMask)
        {
            Tensor compressedHand = fullHand.sum(dim: 1);
            Tensor compressedWorkingHand = fullHand.mul(inUseMask.unsqueeze(2)).sum(dim: 1);
            Tensor output = _mlp.forward(concat([compressedHand, compressedWorkingHand, otherState], dim: 1));
            return output;
        }
    }
}
