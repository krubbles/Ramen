namespace BalatroAI;

/// <summary>
/// Holds the state of the player's hand including remaining plays and discards.
/// </summary>
public class HandState 
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    readonly Card[] _handBuffer = new Card[30];

    public int CardsInHand {get; private set; }

    public Span<Card> Hand => _handBuffer.AsSpan(0, CardsInHand);

    /// <summary>
    /// The number of hands the player can still play this round.
    /// </summary>
    public int RemainingHands;
    /// <summary>
    /// The number of discards the player can still make this round.
    /// </summary>
    public int RemainingDiscards;

    /// <summary>
    /// Pattern matching results for the currently active (played/discarded) hand. Used by jokers and scoring.
    /// </summary>
    public HandPatternResults ActiveHandPatternResults;

    /// <summary>
    /// If the player has less then this number of cards in their hand after playing/discarding, they will draw up to this number.
    /// </summary>
    public int HandSize = 8;

    /// <summary>
    /// The number of hands the player can play each round.
    /// </summary>
    public int HandsPerRound = 4;

    /// <summary>
    /// The number of discards the player can use each round.
    /// </summary>
    public int DiscardsPerRound = 3;

    public HandState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
    }

    public void CloneFrom(HandState other)
    {
        RemainingHands = other.RemainingHands;
        RemainingDiscards = other.RemainingDiscards;
        other.Hand.CopyTo(_handBuffer);
        CardsInHand = other.CardsInHand;

        ActiveHandPatternResults = other.ActiveHandPatternResults;
        HandSize = other.HandSize;
        HandsPerRound = other.HandsPerRound;
        DiscardsPerRound = other.DiscardsPerRound;
    }


    /// <summary>
    /// Adds a card to the players hand. This card is not added to the deck and ceases to exist after being played.
    /// </summary>
    public void AddCardToHand(Card card) 
    {
        _handBuffer[CardsInHand++] = card;
    }

    /// <summary>
    /// If the player has less then <see cref="HandSize"/> cards in their hand, draw cards from the deck until they have <see cref="HandSize"/> cards.
    /// </summary>
    public void DrawToHandSize()
    {
        while (CardsInHand < HandSize && GameState.DeckState.TryDraw(out Card card))
            AddCardToHand(card);
    }

    /// <summary>
    /// Plays a hand.
    /// </summary>
    public double PlayHand(ReadOnlySpan<int> cardIndices)
    {
        if (RemainingHands < 1)
            throw new Exception("Cannot play hand, out of hands.");
        RemainingHands--;

        Span<Card> playedHand = stackalloc Card[cardIndices.Length];
        for (int i = 0; i < cardIndices.Length; i++)
        {
            playedHand[i] = _handBuffer[cardIndices[i]];
            _handBuffer[cardIndices[i]] = Card.Null;
        }

        GameState.PatternMatchingState.MatchHand(playedHand, out ActiveHandPatternResults);

        // remove played cards
        int writeIndex = 0;
        for (int i = 0; i < CardsInHand; ++i)
        {
            Card card = _handBuffer[i];
            if (!card.IsNull)
                _handBuffer[writeIndex++] = card;
        }
        CardsInHand = writeIndex;

        GameState.JokerState.OnBeforePlayHand();

        double score = GameState.ScoringState.ScoreHand(playedHand);

        DrawToHandSize();

        return score;
    }

    /// <summary>
    /// Discards a hand.
    /// </summary>
    public void DiscardHand(ReadOnlySpan<int> cardIndices)
    {
        if (RemainingDiscards < 1)
            throw new Exception("Cannot discard, out of discards");
        RemainingDiscards--;

        GameState.JokerState.OnDiscardHand();
        for (int i = 0; i < cardIndices.Length; i++)
        {
            GameState.JokerState.OnDiscardCard(_handBuffer[cardIndices[i]]);
            _handBuffer[cardIndices[i]] = Card.Null;
        } 

        // remove discarded cards
        int writeIndex = 0;
        for (int i = 0; i < CardsInHand; ++i)
        {
            Card card = _handBuffer[i];
            if (!card.IsNull)
                _handBuffer[writeIndex++] = card;
        }
        CardsInHand = writeIndex;

        
        DrawToHandSize();
    }

    /// <summary>
    /// Resets the remaining hands and discards to the per-round values.
    /// </summary>
    public void ResetRemainingHandsAndDiscards()
    {
        RemainingHands = HandsPerRound;
        RemainingDiscards = DiscardsPerRound;
    }
}