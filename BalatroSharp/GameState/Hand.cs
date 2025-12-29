namespace BalatroAI;

public class HandState // Hand
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    readonly Card[] _handBuffer = new Card[30];

    public int CardsInHand {get; private set; }

    public Span<Card> Hand => _handBuffer.AsSpan(0, CardsInHand);


    public int RemainingHands, RemainingDiscards;

    public HandPatternResults ActiveHandPatternResults;
    
    // Settings
    public int MaxHandSize = 8;
    public int HandsPerRound = 4, DiscardsPerRound = 3;

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
        MaxHandSize = other.MaxHandSize;
        HandsPerRound = other.HandsPerRound;
        DiscardsPerRound = other.DiscardsPerRound;
    }


    public void AddCardToHand(Card card) 
    {
        _handBuffer[CardsInHand++] = card;
    }

    public void RemoveCardFromHand(int index)
    {
        _handBuffer[index] = _handBuffer[--CardsInHand];
    }

    public void DrawToHandSize()
    {
        while (CardsInHand < MaxHandSize && GameState.DeckState.TryDraw(out Card card))
            AddCardToHand(card);
    }


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

        GameState.JokerState.OnPlayHand();

        double score = GameState.ScoringState.ScoreHand(playedHand);

        DrawToHandSize();

        return score;
    }

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

    public void ResetRemainingHandsAndDiscards()
    {
        RemainingHands = HandsPerRound;
        RemainingDiscards = DiscardsPerRound;
    }
}