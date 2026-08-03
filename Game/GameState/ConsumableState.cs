namespace Ramen.Game;

public enum ConsumableType : byte
{
    Null,
    Tarot,
    Planet
}

public readonly record struct ConsumableCard
{
    public readonly ConsumableType ConsumableType;
    public readonly HandType PlanetType;
    public readonly TarotCardType TarotType;
    public bool IsNegative { get; init; }
    public int SellValue { get; init; }

    public ConsumableCard(TarotCardType tarot)
    {
        ConsumableType = ConsumableType.Tarot;
        TarotType = tarot;
    }

    public ConsumableCard(HandType planet)
    {
        ConsumableType = ConsumableType.Planet;
        PlanetType = planet;
    }
}

public sealed class ConsumableState
{
    public readonly List<ConsumableCard> Cards = [];
    public int Capacity = 2;
    public TarotCardType LastUsedTarotCard { get; internal set; }
}
