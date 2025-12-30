namespace BalatroAI;

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

public class BestMovePredictor : Module
{
    public const int OtherStateWidth = 3;
    public const int CardInputWidth = 53;

    public const int EmbeddedCardWidth = 64;
    public const int MoveVectorWidth = 128;

    Embedding mainEmbedder;
    Embedding moveVecEmbedder;
    Sequential moveQueryGenerator;

    Sequential fullHandProcessor, workingHandProcessor;

    public BestMovePredictor() : base(nameof(BestMovePredictor))
    {
        mainEmbedder = Embedding(53, EmbeddedCardWidth);
        moveVecEmbedder = Embedding(53, MoveVectorWidth);

        fullHandProcessor = Sequential(
            ReLU(),
            new ResidualMLP(EmbeddedCardWidth, 1));

        workingHandProcessor = Sequential(
            ReLU(),
            new ResidualMLP(EmbeddedCardWidth, 1));

        moveQueryGenerator = Sequential(
            ReLU(),
            new ResidualMLP(EmbeddedCardWidth + OtherStateWidth, 1),
            Linear(EmbeddedCardWidth + OtherStateWidth, MoveVectorWidth)
            );

        RegisterComponents();
    }

    public Tensor EmbedCards(Tensor cards)
    {
        return mainEmbedder.forward(cards);
    }


    public Tensor GetCardUseRewards(Tensor hand, Tensor otherState, Tensor inUseMask)
    {
        Tensor embeddedHand = mainEmbedder.forward(hand);

        Tensor fullHand = embeddedHand.sum(dim: 1);
        Tensor processedFullHand = fullHandProcessor.forward(fullHand);

        Tensor workingHand = embeddedHand.mul(inUseMask.unsqueeze(2)).sum(dim: 1);
        Tensor processedWorkingHand = workingHandProcessor.forward(workingHand);

        Tensor moveQuery = moveQueryGenerator.forward(concat([processedFullHand - processedWorkingHand, otherState], dim: 1));
        Tensor moveQueryExpanded = moveQuery.unsqueeze(1).expand([hand.size(0), hand.size(1), moveQuery.size(1)]);

        Tensor moveKeys = moveVecEmbedder.forward(hand);

        Tensor output = mul(moveKeys, moveQueryExpanded).sum(dim: 2) / MathF.Sqrt(MoveVectorWidth);
        return output;
    }
}
