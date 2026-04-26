namespace Ramen.Game;

/// <summary>
/// State related to the shop in Balatro.
/// </summary>
public sealed class ShopState
{
    public readonly GameState GameState;

    /// <summary>
    /// The amount of money the player has.
    /// </summary>
    public int Money { get; internal set; }

    /// <summary>
    /// The number of offerings you get when the store is re-rolled.
    /// </summary>
    public int ShopSize { get; internal set; } = 2;

    /// <summary>
    /// Current purchasable jokers/consumables in the shop. Does not include packs or vouchers.
    /// </summary>
    public readonly List<JokerInstance> ShopOfferings = [];

    /// <summary>
    /// The starting cost of a reroll each time the player enters the store.
    /// </summary>
    public int StartingRerollCost { get; internal set; } = 5;

    /// <summary>
    /// The number of rerolls used this round.
    /// </summary>
    public int RerollsThisRoundCount { get; internal set; }

    /// <summary>
    /// The number of remaining free rerolls.
    /// </summary>
    public int FreeRerollsRemaining { get; internal set; }

    /// <summary>
    /// How much it would cost to reroll the store right now.
    /// </summary>
    public int CurrentRerollCost => FreeRerollsRemaining > 0 ? 0 : StartingRerollCost + RerollsThisRoundCount;

    public ShopState(GameState gameState)
    {
        GameState = gameState;
    }

    /// <summary>
    /// Returns the price of the shop offering at the given index, or
    /// <see cref="int.MaxValue"/> if that offering has already been purchased.
    /// </summary>
    public int GetShopOfferingPrice(int index)
    {
        return ShopOfferings[index]?.JokerData.BasePrice ?? int.MaxValue;
    }

    internal void AppendLegalMoves(List<Move> moves)
    {
        for (int offeringIndex = 0; offeringIndex < ShopOfferings.Count; ++offeringIndex)
        {
            if (GetShopOfferingPrice(offeringIndex) <= Money)
                moves.Add(new BuyShopOfferMove(offeringIndex));
        }
        if (CurrentRerollCost <= Money)
            moves.Add(new RerollMove());
        moves.Add(new ExitShopMove());
    }

    internal void EnterShop()
    {
        GameState.AssertIsStage(StageOfGame.EnterShop);
        RerollsThisRoundCount = 0;
        FreeRerollsRemaining = 1;
        Reroll();
        GameState.Stage = StageOfGame.InShop;
    }

    internal void ExitShop()
    {
        GameState.Stage = StageOfGame.BeginRound;
    }

    /// <summary>
    /// Rerolls the store.
    /// </summary>
    internal void Reroll()
    {
        if (FreeRerollsRemaining > 0)
            FreeRerollsRemaining--;
        else
        {
            if (Money < CurrentRerollCost)
                throw new InvalidOperationException($"Not enough money to reroll the store. Current cost: {CurrentRerollCost}, Money: {Money}");
            Money -= CurrentRerollCost;
            RerollsThisRoundCount++;
        }
        ShopOfferings.Clear();
        while (ShopOfferings.Count < ShopSize)
            ShopOfferings.Add(GenerateShopOffering());
    }

    JokerInstance GenerateShopOffering()
    {
        return new(GameState.Random.NextPick(GameState.GameData.Jokers));
    }

    internal JokerInstance BuyShopOffering(int index)
    {
        GameState.AssertIsStage(StageOfGame.InShop);
        int price = GetShopOfferingPrice(index);
        if (Money < price)
            throw new InvalidOperationException($"Not enough money to buy shop offering. Price: {price}, Money: {Money}");

        Money -= price;
        JokerInstance joker = ShopOfferings[index];
        GameState.JokerState.AddJoker(joker);
        ShopOfferings[index] = null;
        return joker;
    }
}
