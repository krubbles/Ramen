namespace Ramen.Game;

public partial class Joker // Joker Definition
{
    public string Name;
    public Rarity Rarity;
    public int BasePrice;

    public Action<GameState, JokerInstance>
        OnAdd,
        OnRemove,
        OnPlayHand,
        OnDiscardHand,
        OnJokerTrigger;

    public Action<GameState, JokerInstance, Card>
        OnBeginScoringCard,
        OnScoreCard,
        OnDiscardCard;
}

public class JokerInstance
{
    public readonly Joker JokerData;
    public int State = 0;

    public JokerInstance(Joker jokerData)
    {
        JokerData = jokerData;
    }
}

public enum Rarity
{
    Common,
    Uncommon,
    Rare,
    Legendary
}
