namespace BalatroAI;

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;

public class BestMovePredictor : Module
{
    public const int OtherStateWidth = 3;
    public const int CardInputWidth = 53;
    public const int EmbeddedCardWidth = 128;
    public const int MoveEvalHandHW = 128;
    public const int MoveEvalCardHW = 128;

    public const int MoveVectorWidth = 128;

    Sequential cardProcessor;
    Sequential moveEvalHand;
    Sequential moveEvalCard;

    public BestMovePredictor() : base(nameof(BestMovePredictor))
    {
        cardProcessor = Sequential(
            Linear(CardInputWidth, EmbeddedCardWidth));

        // input: hand (sum of processed cards)
        moveEvalHand = Sequential(
            ReLU(),
            Linear(EmbeddedCardWidth * 2 + OtherStateWidth, MoveEvalHandHW),
            ReLU(),
            Linear(MoveEvalHandHW, MoveVectorWidth)
            );

        // input: full hand, full hand extra info, current working hand, potential card to add
        moveEvalCard = Sequential(
            ReLU(),
            Linear(EmbeddedCardWidth, MoveEvalCardHW),
            ReLU(),
            Linear(MoveEvalCardHW, MoveVectorWidth)
            );

        RegisterComponents();
    }

    public Tensor EmbedCards(Tensor cards)
    {
        return cardProcessor.forward(cards);
    }


    public Tensor GetCardUseRewards(Tensor fullHand, Tensor otherState, Tensor inUseMask)
    {
        Tensor compressedHand = fullHand.sum(dim: 1);
        Tensor compressedWorkingHand = fullHand.mul(inUseMask.unsqueeze(2)).sum(dim: 1);
        Tensor handVec = moveEvalHand.forward(concat([compressedHand, compressedWorkingHand, otherState], dim: 1));
        Tensor cardVecs = moveEvalCard.forward(fullHand);
        Tensor handVecExpanded = handVec.unsqueeze(1).expand([fullHand.size(0), fullHand.size(1), handVec.size(1)]);
        Tensor output = mul(handVecExpanded, cardVecs).sum(dim: 2) / MathF.Sqrt(MoveVectorWidth);
        return output;
    }
}
