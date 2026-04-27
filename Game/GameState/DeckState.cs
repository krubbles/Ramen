namespace Ramen.Game;

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
    public ReadOnlySpan<Card> RemainingDeck => _deckBuffer.AsSpan(0, RemainingDeckCardCount);

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
            Card.HashCardSetOrdered(RemainingDeck) * 210384047 +
            Card.HashCardSetOrdered(FullDeck);
    }

    internal void CloneFrom(DeckState other)
    {
        other.RemainingDeck.CopyTo(_deckBuffer);
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
        ShuffleDeck();
        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    internal void SetRemainingDeck(ReadOnlySpan<Card> cards)
    {
        cards.CopyTo(_deckBuffer);
        RemainingDeckCardCount = cards.Length;
        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    /// <summary>
    /// Randomly shuffles <see cref="RemainingDeck"/>.
    /// </summary>
    internal void ShuffleDeck()
    {
        Span<int> indices = stackalloc int[RemainingDeckCardCount];
        for (int i = 0; i < indices.Length; ++i)
            indices[i] = GameState.Random.NextInRange(i, RemainingDeckCardCount);
        for (int i = 0; i < indices.Length; ++i)
            (_deckBuffer[i], _deckBuffer[indices[i]]) = (_deckBuffer[indices[i]], _deckBuffer[i]);
    }

    /// <summary>
    /// Inverts the shuffle produced by <see cref="ShuffleDeck"/> on <see cref="RemainingDeck"/> assuming the <see cref="GameState.Random"/> starts at the same state.
    /// </summary>
    internal void UnshuffleDeck()
    {
        Span<int> indices = stackalloc int[RemainingDeckCardCount];
        for (int i = 0; i < indices.Length; ++i)
            indices[i] = GameState.Random.NextInRange(i, RemainingDeckCardCount);
        for (int i = indices.Length - 1; i >= 0; --i)
            (_deckBuffer[i], _deckBuffer[indices[i]]) = (_deckBuffer[indices[i]], _deckBuffer[i]);
    }

    internal void Draw(Span<Card> cards)
    {
        int count = cards.Length;
        for (int i = 0; i < count; ++i)
            cards[i] = Draw();

        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    internal void UnDraw(ReadOnlySpan<Card> cards)
    {
        int count = cards.Length;
        for (int i = cards.Length - 1; i >= 0; --i)
            UnDraw(cards[i]);

        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    internal int RemoveCardFromRemainingDeck(Card card)
    {
        int index = Array.IndexOf(_deckBuffer, card, 0, RemainingDeckCardCount);
        if (index < 0)
            throw new ArgumentException($"Card {card} not found in remaining deck.");
        RemainingDeckCardCount--;
        for (int i = index; i < RemainingDeckCardCount; ++i)
            _deckBuffer[i] = _deckBuffer[i + 1];
        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
        return index;
    }

    internal void InsertCardIntoRemainingDeck(Card card, int index)
    {
        if (index < 0 || index > RemainingDeckCardCount)
            throw new ArgumentOutOfRangeException(nameof(index), index, "Index is outside of the remaining deck range.");
        for (int i = RemainingDeckCardCount; i > index; --i)
            _deckBuffer[i] = _deckBuffer[i - 1];
        _deckBuffer[index] = card;
        RemainingDeckCardCount++;
        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
    }

    // these functions don't call events, so must be private.

    Card Draw()
    {
        return _deckBuffer[--RemainingDeckCardCount];
    }

    void UnDraw(Card card)
    {
        _deckBuffer[RemainingDeckCardCount++] = card;
    }
}
