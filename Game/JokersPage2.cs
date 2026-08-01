namespace Ramen.Game;

public partial class Joker // Page 2 Jokers
{
    // Jokers 16-30 of the collection. Effects follow the Balatro wiki's joker list.
    //
    // Several of these need engine features that do not exist yet. Those are declared
    // as null so the names still resolve, and are left out of Page2Jokers. Each one
    // documents the specific capability it is waiting on.

    public static readonly Joker HalfJoker = new()
    {
        Name = "HalfJoker",
        Rarity = Rarity.Common,
        BasePrice = 5,
        OnJokerTrigger = (gameState, joker) =>
        {
            if (gameState.HandState.ActiveHandPatterns.CardCount <= 3)
                gameState.ScoringState.AddMultToCurrentHand(20);
        }
    };

    /// <summary>
    /// X1 Mult per empty joker slot. Every Joker Stencil counts as an empty slot,
    /// including this one and any other copies.
    /// </summary>
    public static readonly Joker JokerStencil = new()
    {
        Name = "JokerStencil",
        Rarity = Rarity.Uncommon,
        BasePrice = 8,
        OnJokerTrigger = (gameState, joker) =>
        {
            List<JokerInstance> jokers = gameState.JokerState.Jokers;
            int emptySlots = gameState.GameData.MaxJokers - jokers.Count;
            int stencilCount = 0;
            for (int i = 0; i < jokers.Count; ++i)
            {
                if (jokers[i].JokerData == JokerStencil)
                    stencilCount++;
            }
            gameState.ScoringState.CurrentHandMult *= emptySlots + stencilCount;
        }
    };

    public static readonly Joker FourFingers = new()
    {
        Name = "FourFingers",
        Rarity = Rarity.Uncommon,
        BasePrice = 7,
        OnAdd = (gameState, joker) =>
        {
            gameState.PatternMatchingState.FlushSize = 4;
            gameState.PatternMatchingState.StraightSize = 4;
        },
        OnRemove = (gameState, joker) =>
        {
            gameState.PatternMatchingState.FlushSize = 5;
            gameState.PatternMatchingState.StraightSize = 5;
        }
    };

    // Retrigger all card held in hand abilities.
    // Needs a held-in-hand scoring hook; the engine only triggers jokers for played
    // cards (OnBeginScoringCard / OnScoreCard) and has no held-in-hand phase.
    public static readonly Joker Mime = null;

    /// <summary>
    /// How far into debt a single Credit Card lets the player go.
    /// </summary>
    public const int CreditCardDebtAllowance = 20;

    /// <summary>
    /// Go up to -$20 in debt. Copies do not stack; the floor is -$20 no matter how many
    /// Credit Cards are held.
    /// </summary>
    public static readonly Joker CreditCard = new()
    {
        Name = "CreditCard",
        Rarity = Rarity.Common,
        BasePrice = 1,
        OnAdd = (gameState, joker) => RecomputeMinimumMoney(gameState),
        OnRemove = (gameState, joker) => RecomputeMinimumMoney(gameState)
    };

    /// <summary>
    /// Recomputes the debt floor from the jokers currently held. Both joker hooks run
    /// after the list has been updated, so this stays correct however copies are added
    /// or removed.
    /// </summary>
    static void RecomputeMinimumMoney(GameState gameState)
    {
        List<JokerInstance> jokers = gameState.JokerState.Jokers;
        int minimumMoney = 0;
        for (int i = 0; i < jokers.Count; ++i)
        {
            if (jokers[i].JokerData == CreditCard)
            {
                minimumMoney = -CreditCardDebtAllowance;
                break;
            }
        }
        gameState.ShopState.MinimumMoney = minimumMoney;
    }

    // When Blind is selected, destroy the joker to the right and permanently add double
    // its sell value to this joker's mult.
    // Needs a blind-selected hook, joker sell values, and rollback-safe per-joker state.
    public static readonly Joker CeremonialDagger = null;

    public static readonly Joker Banner = new()
    {
        Name = "Banner",
        Rarity = Rarity.Common,
        BasePrice = 5,
        OnJokerTrigger = (gameState, joker) =>
        {
            gameState.ScoringState.AddChipsToCurrentHand(30 * gameState.HandState.RemainingDiscards);
        }
    };

    public static readonly Joker MysticSummit = new()
    {
        Name = "MysticSummit",
        Rarity = Rarity.Common,
        BasePrice = 5,
        OnJokerTrigger = (gameState, joker) =>
        {
            if (gameState.HandState.RemainingDiscards == 0)
                gameState.ScoringState.AddMultToCurrentHand(15);
        }
    };

    // Adds one Stone card to the deck when Blind is selected.
    // Needs a blind-selected hook and a way to add cards to the deck mid-run.
    public static readonly Joker MarbleJoker = null;

    /// <summary>
    /// X4 Mult every 6 hands played. The hand counter lives in
    /// <see cref="JokerInstance.State"/> and is advanced through a
    /// <see cref="SetJokerStateMove"/> so it rolls back with the hand that advanced it.
    /// </summary>
    public static readonly Joker LoyaltyCard = new()
    {
        Name = "LoyaltyCard",
        Rarity = Rarity.Uncommon,
        BasePrice = 5,
        OnJokerTrigger = (gameState, joker) =>
        {
            int handsPlayed = joker.State + 1;
            if (handsPlayed >= 6)
            {
                gameState.ScoringState.CurrentHandMult *= 4;
                handsPlayed = 0;
            }
            new SetJokerStateMove(joker, handsPlayed).Apply(gameState);
        }
    };

    // 1 in 4 chance for each played 8 to create a Tarot card when scored.
    // Needs consumables; the engine has no tarot cards or consumable slots.
    public static readonly Joker EightBall = null;

    /// <summary>
    /// +0-23 Mult, rolled per hand. Safe to roll inline: <see cref="Move.Apply"/> records
    /// the RNG state and <see cref="Move.Revert"/> restores it, so a reverted and
    /// replayed hand rolls the same value.
    /// </summary>
    public static readonly Joker Misprint = new()
    {
        Name = "Misprint",
        Rarity = Rarity.Common,
        BasePrice = 4,
        OnJokerTrigger = (gameState, joker) =>
        {
            gameState.ScoringState.AddMultToCurrentHand(gameState.Random.Next(24));
        }
    };

    /// <summary>
    /// Retriggers every scoring card on the last hand of the round.
    /// </summary>
    public static readonly Joker Dusk = new()
    {
        Name = "Dusk",
        Rarity = Rarity.Uncommon,
        BasePrice = 5,
        OnBeginScoringCard = (gameState, joker, card) =>
        {
            // RemainingHands is decremented before scoring, so zero means this is the
            // final hand of the round.
            if (gameState.HandState.RemainingHands == 0)
                gameState.ScoringState.CurrentScoringCardTriggerCount++;
        }
    };

    /// <summary>
    /// Adds double the rank of the lowest ranked card held in hand to mult.
    /// </summary>
    public static readonly Joker RaisedFist = new()
    {
        Name = "RaisedFist",
        Rarity = Rarity.Common,
        BasePrice = 5,
        OnJokerTrigger = (gameState, joker) =>
        {
            ReadOnlySpan<Card> hand = gameState.HandState.Hand;
            if (hand.Length == 0)
                return;

            int lowestRankValue = int.MaxValue;
            for (int i = 0; i < hand.Length; ++i)
            {
                int rankValue = GameData.BaseChipsForCardRank(hand[i].Rank);
                if (rankValue < lowestRankValue)
                    lowestRankValue = rankValue;
            }
            gameState.ScoringState.AddMultToCurrentHand(2 * lowestRankValue);
        }
    };

    // 1 free reroll per shop.
    // Needs a shop-entry hook. ShopState.EnterShop hardcodes FreeRerollsRemaining, so a
    // joker has no point at which it could grant an extra one.
    public static readonly Joker ChaosTheClown = null;

    public static readonly Joker[] Page2Jokers =
    [
        HalfJoker,
        JokerStencil,
        FourFingers,
        CreditCard,
        Banner,
        MysticSummit,
        LoyaltyCard,
        Misprint,
        Dusk,
        RaisedFist,
    ];
}
