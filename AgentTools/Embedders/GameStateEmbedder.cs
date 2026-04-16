namespace Ramen.AgentTools;

using Ramen.Game;
using static TorchSharp.torch;

/// <summary>
/// An embedded GameState.
/// </summary>
public class GameStateTensors : ITensorGroup
{
    /// <summary>
    /// Vector of cards in <see cref="HandState.Hand"/> encoded with <see cref="Card.ToIndex"/>.
    /// </summary>
    public Tensor FullHand;

    /// <summary>
    /// Vector of cards in <see cref="DeckState.RemainingDeck"/> encoded with <see cref="Card.ToIndex"/>.
    /// </summary>
    public Tensor RemainingDeck;

    /// <summary>
    /// <see cref="ScoringState.CurrentRoundTotalChips"/>
    /// </summary>
    public Tensor Score;

    /// <summary>
    /// <see cref="HandState.RemainingHands"/> * 5 + <see cref="HandState.RemainingDiscards"/>
    /// </summary>
    public Tensor HandsAndDiscards;
}

/// <summary>
/// An embedded Move.
/// </summary>
public class UseHandTensors : ITensorGroup
{
    /// <summary>
    /// <see cref="ScoringState.CurrentRoundTotalChips"/> after hand is played.
    /// </summary>
    public Tensor Score;
}

public class GameStateEmbedder
{
    readonly long[,] _fullHand;
    readonly long[,] _remainingDeck;
    readonly long[] _handsAndDiscards;
    readonly float[] _score;

    int _addedGameStateCount;

    public GameStateEmbedder(int gameStateCount)
    {
        _fullHand = new long[gameStateCount, GameData.HandSize];
        _remainingDeck = new long[gameStateCount, 52];
        _handsAndDiscards = new long[gameStateCount];
        _score = new float[gameStateCount];
    }


    public void AddGameState(GameState gameState)
    {
        if (_addedGameStateCount >= _handsAndDiscards.Length)
            throw new InvalidOperationException("GameStateEmbedder is already full.");

        ReadOnlySpan<Card> hand = gameState.HandState.Hand;
        for (int cardIndex = 0; cardIndex < hand.Length; ++cardIndex)
            _fullHand[_addedGameStateCount, cardIndex] = hand[cardIndex].ToIndex();

        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int cardIndex = 0; cardIndex < deck.Length; ++cardIndex)
            _remainingDeck[_addedGameStateCount, cardIndex] = deck[cardIndex].ToIndex();

        _handsAndDiscards[_addedGameStateCount] = GetHandsAndDiscardsValue(gameState);
        _score[_addedGameStateCount] = GetScoreValue(gameState);
        _addedGameStateCount++;
    }


    public GameStateTensors ToTensors()
    {
        return ToTensors(CPU);
    }


    public GameStateTensors ToTensors(Device device)
    {
        long[,] fullHand = _fullHand;
        long[,] remainingDeck = _remainingDeck;
        long[] handsAndDiscards = _handsAndDiscards;
        float[,] score2D = new float[_addedGameStateCount, 1];
        if (_addedGameStateCount != _handsAndDiscards.Length)
        {
            fullHand = new long[_addedGameStateCount, GameData.HandSize];
            remainingDeck = new long[_addedGameStateCount, 52];
            handsAndDiscards = new long[_addedGameStateCount];

            for (int stateIndex = 0; stateIndex < _addedGameStateCount; ++stateIndex)
            {
                for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                    fullHand[stateIndex, cardIndex] = _fullHand[stateIndex, cardIndex];

                for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                    remainingDeck[stateIndex, cardIndex] = _remainingDeck[stateIndex, cardIndex];

                handsAndDiscards[stateIndex] = _handsAndDiscards[stateIndex];
                score2D[stateIndex, 0] = _score[stateIndex];
            }
        }
        else
        {
            for (int stateIndex = 0; stateIndex < _addedGameStateCount; ++stateIndex)
                score2D[stateIndex, 0] = _score[stateIndex];
        }

        return new()
        {
            FullHand = tensor(fullHand, dtype: ScalarType.Int64, device: device),
            RemainingDeck = tensor(remainingDeck, dtype: ScalarType.Int64, device: device),
            HandsAndDiscards = tensor(handsAndDiscards, dtype: ScalarType.Int64, device: device),
            Score = tensor(score2D, dtype: ScalarType.Float32, device: device),
        };
    }


    static float GetScoreValue(GameState gameState)
    {
        return (float)gameState.ScoringState.CurrentRoundTotalChips / 300f;
    }


    static int GetHandsAndDiscardsValue(GameState gameState)
    {
        return gameState.HandState.RemainingHands * 5 + gameState.HandState.RemainingDiscards;
    }
}
