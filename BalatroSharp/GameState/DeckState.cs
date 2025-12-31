namespace BalatroAI;

public sealed class DeckState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    readonly Card[] _deckBuffer = new Card[100], _fullDeckBuffer = new Card[100];

    public int RemainingDeckCardCount { get; private set; }
    public int FullDeckCardCount { get; private set; }

    /// <summary>
    /// The cards remaining in the deck for the current round.
    /// </summary>
    public ReadOnlySpan<Card> Deck => _deckBuffer.AsSpan(0, RemainingDeckCardCount);

    /// <summary>
    /// All cards in the player's deck.
    /// </summary>
    public ReadOnlySpan<Card> FullDeck => _fullDeckBuffer.AsSpan(0, FullDeckCardCount);

    public Action OnRemainingDeckChanged;
    public Action OnFullDeckChanged;

    public DeckState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
    }

    public override int GetHashCode()
    {
        return
            463507903 +
            Card.HashCardSetOrdered(Deck) * 210384047 +
            Card.HashCardSetOrdered(FullDeck);
    }

    internal void CloneFrom(DeckState other)
    {
        other.Deck.CopyTo(_deckBuffer);
        other.FullDeck.CopyTo(_fullDeckBuffer);
        RemainingDeckCardCount = other.RemainingDeckCardCount;
        FullDeckCardCount = other.FullDeckCardCount;
    }

    internal void AddCardToFullDeck(Card card)
    {
        _fullDeckBuffer[FullDeckCardCount++] = card;
        GameState.MoveState.ScheduleCallback(OnFullDeckChanged);
    }

    internal void ResetDeck()
    {
        FullDeck.CopyTo(_deckBuffer);
        RemainingDeckCardCount = FullDeckCardCount;
        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    internal void Draw(Span<Card> cards)
    {
        int count = cards.Length;
        Span<int> indices = stackalloc int[count];
        for (int i = 0; i < count; ++i)
            indices[i] = GameState.Random.Next(RemainingDeckCardCount - i);
        for (int i = 0; i < count; ++i)
            cards[i] = Draw(indices[i]);

        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    internal void UnDraw(ReadOnlySpan<Card> cards)
    {
        int count = cards.Length;
        Span<int> indices = stackalloc int[count];
        for (int i = 0; i < count; ++i)
            indices[i] = GameState.Random.Next(RemainingDeckCardCount - i);

        for (int i = indices.Length - 1; i >= 0; --i)
            UnDraw(indices[i], cards[i]);

        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    // these functions don't call events, so must be private.

    Card Draw(int index)
    {
        if (index < 0 || index >= RemainingDeckCardCount)
            throw new IndexOutOfRangeException($"Index {index} out of range [0, {RemainingDeckCardCount})");

        Card card = _deckBuffer[index];
        _deckBuffer[index] = _deckBuffer[--RemainingDeckCardCount];
        return card;
    }

    void UnDraw(int index, Card card)
    {
        if (index < 0 || index >= RemainingDeckCardCount)
            throw new IndexOutOfRangeException($"Index {index} out of range [0, {RemainingDeckCardCount})");

        _deckBuffer[RemainingDeckCardCount++] = _deckBuffer[index];
        _deckBuffer[index] = card;
    }
}