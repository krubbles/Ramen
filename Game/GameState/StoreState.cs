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
    public int ShopSize { get; internal set; }

    /// <summary>
    /// Current purchasable jokers/consumables in the shop. Does not include packs or vouchers.
    /// </summary>
    public readonly List<Joker> ShopOfferings = [];

    /// <summary>
    /// The starting cost of a reroll each time the player enters the store.
    /// </summary>
    public int StartingRerollCost { get; internal set; }

    /// <summary>
    /// The current cost of a reroll. Note that if <see cref="FreeRerollsRemaining"/> > 0, then you do not have to pay this cost.
    /// </summary>
    public int CurrentRerollCost { get; internal set; }

    /// <summary>
    /// The number of remaining free rerolls
    /// </summary>
    public int FreeRerollsRemaining { get; internal set; }

    /// <summary>
    /// Returns the price of the shop offering at the given index, or
    /// <see cref="int.MaxValue"/> if that offering has already been purchased.
    /// </summary>
    public int GetShopOfferingPrice(int index)
    {
        return ShopOfferings[index]?.BasePrice ?? int.MaxValue;
    }

    internal void EnterShop()
    {
        GameState.AssertIsStage(StageOfGame.EnterShop);
        CurrentRerollCost = StartingRerollCost;
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
        ShopOfferings.Clear();
        while (ShopOfferings.Count < ShopSize)
            ShopOfferings.Add(GenerateShopOffering());
    }

    Joker GenerateShopOffering()
    {
        return GameState.Random.NextPick(GameState.GameData.Jokers);
    }
}
