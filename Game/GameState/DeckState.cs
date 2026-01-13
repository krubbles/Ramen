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
        Span<Card> cards = stackalloc Card[FullDeckCardCount];
        for (int i = 0; i < cards.Length; ++i)
            cards[i] = _fullDeckBuffer[i];
        int remainingDeckIndex = 0;
        while (cards.Length > 0)
        {
            int index = GameState.Random.Next(cards.Length);
            Card card = cards[index];
            _deckBuffer[remainingDeckIndex++] = card;
            cards[index] = cards[^1];
            cards = cards[0..(cards.Length - 1)];
        }
        RemainingDeckCardCount = FullDeckCardCount;
        GameState.MoveState.ScheduleCallback(OnRemainingDeckChanged);
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