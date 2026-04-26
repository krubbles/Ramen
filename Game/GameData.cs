namespace Ramen.Game;

public sealed class GameData
{
    public const int MaxPlayedHandSize = 5;
    public const int HandSize = 8;

    public int Seed;

    public int Hands = 4, Discards = 3;

    public Action<GameState> InitStartingDeck = InitStandardStartingDeck;

    // When true, GameState should choose a random seed instead of using the Seed field.
    public bool RandomizeSeed = true;

    public readonly Joker[] Jokers = Joker.Page1Jokers;

    public (int chips, int mult)[] StartingHandBaseScore =
    [
        (0, 0), // null
        (5, 1), // high card
        (10, 2), // pair
        (20, 2), // two pair
        (30, 3), // 3oak
        (30, 4), // straight
        (35, 4), // flush
        (40, 4), // full house
        (60, 7), // 4oak
        (100, 8), // straight flush
        (120, 12), // 5oak
        (140, 14), // flush house
        (160, 16), // flush five
    ];

    public (int chips, int mult)[] PlanetScores =
    [
        (0, 0), // null
        (10, 1), // high card
        (15, 1), // pair
        (20, 1), // two pair
        (20, 2), // 3oak
        (30, 3), // straight
        (15, 2), // flush
        (25, 2), // full house
        (30, 3), // 4oak
        (40, 4), // straight flush
        (35, 3), // 5oak
        (40, 4), // flush house
        (50, 3), // flush five
    ];

    public int[] RoundScoreThresholds =
    [
        0, // round zero doesn't exist
        300,
        450,
        600,
    ];

    public static int BaseChipsForCardRank(int rank)
    {
        if (rank <= 10)
            return rank;
        else return rank == 14 ? 11 : 10;
    }

    public (int chips, int mult) GetHandBaseScore(HandType handType, int level)
    {
        (int chips, int mult) = StartingHandBaseScore[(int)handType];
        (int planetChips, int planetMult) = PlanetScores[(int)handType];
        return (chips + planetChips * (level - 1), mult + planetMult * (level - 1));
    }

    public static void InitStandardStartingDeck(GameState gameState)
    {
        for (int rank = 2; rank <= 14; ++rank)
        {
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Spade));
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Heart));
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Club));
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Diamond));
        }
    }

    public static void InitCheckeredStartingDeck(GameState gameState)
    {
        for (int rank = 2; rank <= 14; ++rank)
        {
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Spade));
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Heart));
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Spade));
            gameState.DeckState.AddCardToFullDeck(new(rank, Suit.Heart));
        }
    }

    public static void InitErraticStartingDeck(GameState gameState)
    {
        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
        {
            byte rank = (byte)(gameState.Random.Next(13) + 2);
            Suit suit = (Suit)(gameState.Random.Next(4) + 1);
            gameState.DeckState.AddCardToFullDeck(new(rank, suit));
        }
    }

    public static readonly GameData Default = new();
}
