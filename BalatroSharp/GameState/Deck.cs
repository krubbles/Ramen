namespace BalatroAI;

public class DeckState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    FastRandom _shuffleRandom;


    readonly Card[] _deckBuffer = new Card[100], _fullDeckBuffer = new Card[100];

    public int DeckSize { get; private set; }
    public int FullDeckSize { get; private set; }

    public Span<Card> Deck => _deckBuffer.AsSpan(0, DeckSize);
    public Span<Card> FullDeck => _fullDeckBuffer.AsSpan(0, FullDeckSize);

    public DeckState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
        _shuffleRandom = new(_gameData.Seed);
    }

    public void CloneFrom(DeckState other)
    {
        _shuffleRandom = new(other._shuffleRandom);
        other.Deck.CopyTo(_deckBuffer);
        other.FullDeck.CopyTo(_fullDeckBuffer);
        DeckSize = other.DeckSize;
        FullDeckSize = other.FullDeckSize;
    }

    public void AddCardToFullDeck(Card card)
    {
        _fullDeckBuffer[FullDeckSize++] = card;
    }

    public void AddCardToDeck(Card card)
    {
        int lastIndex = DeckSize++;
        _deckBuffer[lastIndex] = card;
        int index = _shuffleRandom.Next(Deck.Length);
        (_deckBuffer[index], _deckBuffer[lastIndex]) = (_deckBuffer[lastIndex], _deckBuffer[index]);
    }

    public void ResetAndShuffleDeck()
    {
        FullDeck.CopyTo(_deckBuffer);
        DeckSize = FullDeckSize;
        ShuffleDeck();
    }

    public void ShuffleDeck()
    {
        Span<Card> deck = Deck;
        for (int i = 0; i < deck.Length; ++i)
        {
            int index = _shuffleRandom.NextInRange(i, deck.Length);
            (_deckBuffer[i], _deckBuffer[index]) = (_deckBuffer[index], _deckBuffer[i]);
        }
    }

    public bool TryDraw(out Card card)
    {
        if (DeckSize > 0)
        {
            DeckSize--;
            card = _deckBuffer[DeckSize];
            return true;
        }
        else
        {
            card = default;
            return false;
        }
    }

    public void Reseed(int seed)
    {
        _shuffleRandom = new(seed);
        ShuffleDeck();
    }
}