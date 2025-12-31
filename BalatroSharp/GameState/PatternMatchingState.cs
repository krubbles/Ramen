namespace BalatroAI;

/// <summary>
/// A type of poker hand, like high card or flush.
/// </summary>
public enum HandType
{
    None,
    HighCard,
    Pair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
    FiveOfAKind,
    FlushHouse,
    FlushFive,
}


/// <summary>
/// Stores all the pattern matching results for a given hand, like whether it contains a flush or straight.
/// </summary>
public struct HandPatterns
{
    /// <summary>
    /// Number of cards in the hand.
    /// </summary>
    public int CardCount;

    /// <summary>
    /// The resolved hand type of the hand. This is the highest tier hand type that can be made with the cards in the hand.
    /// </summary>
    public HandType HandType;

    /// <summary>
    /// Whether the hand contains a flush. 
    /// </summary>
    public bool ContainsFlush;

    /// <summary>
    /// Whether the hand contains a straight.
    /// </summary>
    public bool ContainsStraight;

    /// <summary>
    /// The number of copies of the most common rank in the hand.
    /// </summary>
    public int MaxNOfAKind;

    /// <summary>
    /// The number of copies of the second most common rank in the hand. Used to identify two pair and full house hands.
    /// </summary>
    public int SecondMaxNOfAKind;

    /// <summary>
    /// A bit mask that specifies which cards in the hand were used to make the matched pattern. This determines what cards get triggered during scoring.
    /// </summary>
    public int PlayedCardsMask;


    /// NOTE: These properties need to be reworked, since they include cards that aren't in the played cards mask, which isn't the correct behaviour for page 1 jokers.

    /// <summary>
    /// Whether the hand contains a pair. May be part of a higher tier hand.
    /// </summary>
    public readonly bool ContainsPair => MaxNOfAKind >= 2;

    /// <summary>
    /// Whether the hand contains a 3-of-a-kind. May be part of a higher tier hand.
    /// </summary>
    public readonly bool Contains3OAK => MaxNOfAKind >= 3;

    /// <summary>
    /// Whether the hand contains at least a two-pair. May be part of a higher tier hand.
    /// </summary>
    public readonly bool ContainsTwoPair => SecondMaxNOfAKind >= 2;
}

/// <summary>
/// The state object for pattern matching operations on hands. Can be modified by jokers to change how patterns are matched.
/// </summary>
public class PatternMatchingState
{
    public readonly GameState GameState;
    readonly  GameData _gameData;

    public int FlushSize = 5;
    public int StraightSize = 5;
    public int StraightSkipDist = 1;
    public bool SmearSuits;

    public PatternMatchingState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
    }

    /// <summary>
    /// Calculates the pattern matching results for a given hand.
    /// </summary>
    public void MatchHand(ReadOnlySpan<Card> hand, out HandPatterns results) 
    {
        results.CardCount = hand.Length;
        results.ContainsFlush = MatchFlush(hand, out int flushPlayedCardsMask, out Suit flushSuit);
        results.ContainsStraight = MatchStraight(hand, out int straightPlayedCardsMask);
        MatchNOfAKind(hand, out int nOfAKindPlayedCardsMask, out _, out results.MaxNOfAKind, out _, out results.SecondMaxNOfAKind);
        if (!results.ContainsFlush && !results.ContainsStraight)
        {
            results.PlayedCardsMask = nOfAKindPlayedCardsMask;
            if (results.SecondMaxNOfAKind == 2)
            {
                results.HandType = results.MaxNOfAKind == 3 ? HandType.FullHouse : HandType.TwoPair;
            }
            else
            {
                results.HandType = results.MaxNOfAKind switch
                {
                    5 => HandType.FiveOfAKind,
                    4 => HandType.FourOfAKind,
                    3 => HandType.ThreeOfAKind,
                    2 => HandType.Pair,
                    1 => HandType.HighCard,
                    _ => throw new Exception($"Unexpected max N-Of-A-Kind in hand matching code: {results.MaxNOfAKind}")
                };
            }
        }
        else
        {
            if (results.MaxNOfAKind == 5)
            {
                results.HandType = HandType.FlushFive;
                results.PlayedCardsMask = nOfAKindPlayedCardsMask | flushPlayedCardsMask;
            }
            else if (results.MaxNOfAKind == 3 && results.SecondMaxNOfAKind == 2)
            {
                results.HandType = HandType.FlushHouse;
                results.PlayedCardsMask = nOfAKindPlayedCardsMask | flushPlayedCardsMask;
            }
            else if (results.ContainsStraight && results.ContainsFlush)
            {
                results.HandType = HandType.StraightFlush;
                results.PlayedCardsMask = flushPlayedCardsMask | straightPlayedCardsMask;
            }
            else if (results.MaxNOfAKind == 4)
            {
                results.HandType = HandType.FourOfAKind;
                results.PlayedCardsMask = nOfAKindPlayedCardsMask;
            }
            else if (results.ContainsFlush)
            {
                results.HandType = HandType.Flush;
                results.PlayedCardsMask = flushPlayedCardsMask;
            }
            else
            {
                results.HandType = HandType.Straight;
                results.PlayedCardsMask = straightPlayedCardsMask;
            }
        }
    }

    internal void MatchNOfAKind(ReadOnlySpan<Card> hand, out int playedCardsMask, out int rank1, out int rank1Count, out int rank2, out int rank2Count)
    {
        if (hand.Length == 1) // optimization for special case of single-card hand
        {
            playedCardsMask = 1;
            rank1 = hand[0].Rank;
            rank1Count = 1;
            rank2 = 0;
            rank2Count = 0;
            return;
        }

        Span<int> countByRank = stackalloc int[15];
        for (int i = 0; i < hand.Length; ++i)
            countByRank[hand[i].Rank]++;

        rank1 = 0;
        rank2 = 0;
        rank1Count = 0;
        rank2Count = 0;
        for (int rank = 2; rank < 15; ++rank)
        {
            int count = countByRank[rank];
            if (count > rank1Count)
            {
                rank2Count = rank1Count;
                rank2 = rank1;
                rank1Count = count;
                rank1 = rank;
            }
            else if (count > rank2Count)
            {
                rank2Count = count;
                rank2 = rank;
            }
        }

        if (rank1Count == 1) // high card 
        {
            int indexOfHighCard = 0;
            for (int i = 0; i < hand.Length; ++i)
            {
                int rank = hand[i].Rank;
                if (rank >= rank1)
                {
                    rank1 = rank;
                    indexOfHighCard = i;
                }
            }
            playedCardsMask = 1 << indexOfHighCard;
        }
        else // not high card
        {
            playedCardsMask = 0;
            if (rank2Count <= 1)
            {
                for (int i = 0; i < hand.Length; ++i)
                    playedCardsMask |= (RanksMatch(hand[i].Rank, rank1) ? 1 : 0) << i;

            }
            else // two pair, full house, flush house
            {
                for (int i = 0; i < hand.Length; ++i)
                    playedCardsMask |= ((RanksMatch(hand[i].Rank, rank1) | RanksMatch(hand[i].Rank, rank2)) ? 1 : 0) << i;
            }
        }
    }

    internal bool MatchFlush(ReadOnlySpan<Card> hand, out int playedCardsMask, out Suit suit)
    {
        playedCardsMask = 0;
        suit = Suit.None;

        if (hand.Length < FlushSize)
            return false;

        Span<int> countBySuit = stackalloc int[6];
        for (int i = 0; i < hand.Length; ++i)
            countBySuit[(int)hand[i].Suit]++;

        int wildcards = countBySuit[(int)Suit.All];   
        for (suit = Suit.None + 1; suit < Suit.All; ++suit)
        {
            if (countBySuit[(int)suit] + wildcards >= FlushSize)
                break;
        }

        if (suit == Suit.All)
            return false;

        for (int i = 0; i < hand.Length; ++i)
            playedCardsMask |= (SuitsMatch(hand[i].Suit, suit) ? 1 : 0) << i;

        return true;
    }

    internal bool MatchStraight(ReadOnlySpan<Card> hand, out int playedCardsMask)
    {
        playedCardsMask = 0;

        if (hand.Length < StraightSize)
            return false;

        Span<int> handRanks = stackalloc int[hand.Length];
        for (int i = 0; i < handRanks.Length; ++i)
        {
            handRanks[i] = hand[i].Rank;
        }
        handRanks.Sort();

        int currentRank = handRanks[0];
        int runLength = 1;
        int runStartRank = currentRank;
        int maxRunLength = 1;
        int maxRunStartRank = 0;
        int maxRunEndRank = 0;

        for (int i = 1; i < handRanks.Length; ++i)
        {
            int nextRank = handRanks[i];
            int skipAmount = nextRank - currentRank;
            if (skipAmount > 0 && skipAmount <= StraightSkipDist)
            {
                runLength++;
                if (runLength > maxRunLength)
                {
                    maxRunLength = runLength;
                    maxRunStartRank = runStartRank;
                    maxRunEndRank = nextRank;
                }
            }
            else
            {
                runLength = 1;
                runStartRank = nextRank;
            }
            currentRank = nextRank;
        }

        if (maxRunStartRank - 1 <= StraightSkipDist && handRanks[^1] == 14) // adding low ace to straight if applicable
        {
            maxRunStartRank = 1;
            maxRunLength += 1;
        }

        if (maxRunLength < StraightSize)
            return false;

        for (int i = 0; i < hand.Length; ++i)
        {
            int rank = hand[i].Rank;
            bool played = rank >= maxRunStartRank && rank <= maxRunEndRank;
            played |= maxRunStartRank == 1 && rank == 14; // low ace       
            playedCardsMask |= (played ? 1 : 0) << i;
        }

        return true;
    }

    /// <summary>
    /// Determines if two ranks match. In vanilla Balatro this isn't modified by jokers, but use this utility in case of future changes.
    /// </summary>
    public bool RanksMatch(int a, int b)
    {
        return a == b;
    }

    /// <summary>
    /// Determines if two suits match. Can be modified by jokers.
    /// </summary>
    public bool SuitsMatch(Suit a, Suit b)
    {
        if (SmearSuits)
        {
            a = CardUtils.SmearedSuit(a);
            b = CardUtils.SmearedSuit(b);
        }
        if (a == Suit.All || b == Suit.All)
            return true;
        return a == b;
    }
}