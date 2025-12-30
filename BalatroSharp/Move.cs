namespace BalatroAI;

public interface IMove
{
    public bool Optional { get; }

    public void Apply(GameState gameState);

    public void Revert(GameState gameState);
}

public class UseHandMove : IMove
{
    public readonly bool IsDiscard;
    public readonly int[] CardIndices;
    public readonly Card[] Cards;

    public UseHandMove(GameState gameState, bool isDiscard, params int[] cardIndices)
    {
        IsDiscard = isDiscard;
        CardIndices = cardIndices;
        Cards = new Card[cardIndices.Length];
        for (int i = 0; i < cardIndices.Length; ++i)
            Cards[i] = gameState.HandState.Hand[cardIndices[i]];
    }

    public bool Optional => true;

    public void Apply(GameState gameState)
    {
        if (IsDiscard)
        {
            gameState.HandState.DiscardHand(CardIndices);
        }
        else
        {
            gameState.HandState.PlayHand(CardIndices);
        }
    }

    public void Revert(GameState gameState)
    {
        gameState.HandState.RevertLastHandAction();
    }
}