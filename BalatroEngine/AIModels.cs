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
        public MoveSelectorModule MoveSelector { get; }
        public EvaluationModule Evaluation { get; }

        public AIModels()
        {
            MoveSelector = new MoveSelectorModule();
            Evaluation = new EvaluationModule(MoveSelector);
        }

        public IEnumerable<Parameter> parameters()
        {
            // return parameters from both modules
            return MoveSelector.parameters().Concat(Evaluation.parameters());
        }

        public IEnumerable<(string, Parameter)> named_parameters()
        {
            return MoveSelector.named_parameters().Concat(Evaluation.named_parameters());
        }

        public Tensor EmbedCards(Tensor cards) => MoveSelector.EmbedCards(cards);

        public Tensor GetCardUseRewards(Tensor fullHand, Tensor otherState, Tensor inUseMask)
            => MoveSelector.GetCardUseRewards(fullHand, otherState, inUseMask);

        public Tensor GetExpectedReward(Tensor fullHand, Tensor otherState)
            => Evaluation.forward(fullHand, otherState);
    }

    // Small evaluation module that predicts expected final reward from a GameState (batched)
    public class EvaluationModule : Module<Tensor, Tensor>
    {
        private readonly MoveSelectorModule _moveSelector;
        private readonly Sequential _mlp;

        public EvaluationModule(MoveSelectorModule moveSelector) : base("EvaluationModule")
        {
            _moveSelector = moveSelector;

            // input width = EmbeddedCardWidth + OtherStateWidth
            int inputWidth = MoveSelectorModule.EmbeddedCardWidth + MoveSelectorModule.OtherStateWidth;
            int hidden = 128;

            _mlp = Sequential(
                Linear(inputWidth, hidden), ReLU(),
                Linear(hidden, hidden), ReLU(),
                Linear(hidden, 1)
            );

            RegisterComponents();
        }

        public IEnumerable<Parameter> parameters()
        {
            return _mlp.parameters();
        }

        public IEnumerable<(string, Parameter)> named_parameters()
        {
            return _mlp.named_parameters();
        }

        // Forward accepts the raw FullHand tensor (N, cards, CardInputWidth) and OtherState (N, otherWidth)
        public override Tensor forward(Tensor x)
        {
            throw new System.NotImplementedException("Use forward(fullHand, otherState) overload");
        }

        public Tensor forward(Tensor fullHand, Tensor otherState)
        {
            // fullHand: (N, cards, CardInputWidth)
            // otherState: (N, OtherStateWidth)
            using var scope = NewDisposeScope();
            Tensor cardEmb = _moveSelector.EmbedCards(fullHand); // (N, cards, EmbeddedCardWidth)
            Tensor handVec = cardEmb.sum(1); // (N, EmbeddedCardWidth)
            handVec = handVec.relu(); // non-linearity as requested
            Tensor inputVec = cat([handVec, otherState], dim: 1);
            Tensor output = _mlp.forward(inputVec);
            return output;
        }
    }
}
