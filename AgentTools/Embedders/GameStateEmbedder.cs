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
    /// Optional tensor of per-play hand score deltas in the standard hand ordering.
    /// Shape: (batch, <see cref="GameStateEmbedder.PlayHandScoreCount"/>).
    /// </summary>
    public Tensor PlayHandScores;

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
    public static readonly int PlayHandScoreCount = Combinatorics.CalculateCombinationCount(
        setSize: GameData.HandSize,
        maxSubsetSize: 5,
        minSubsetSize: 1);

    readonly long[,] _fullHand;
    readonly long[,] _remainingDeck;
    readonly long[] _handsAndDiscards;
    readonly float[,] _playHandScores;
    readonly float[] _score;

    int _addedGameStateCount;

    public GameStateEmbedder(int gameStateCount)
    {
        _fullHand = new long[gameStateCount, GameData.HandSize];
        _remainingDeck = new long[gameStateCount, 52];
        _handsAndDiscards = new long[gameStateCount];
        _playHandScores = new float[gameStateCount, PlayHandScoreCount];
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
        WritePlayHandScores(gameState, _addedGameStateCount);
        _score[_addedGameStateCount] = GetScoreValue(gameState);
        _addedGameStateCount++;
    }


    public GameStateTensors ToTensors()
    {
        return ToTensors(CPU, includePlayHandScores: false);
    }


    public GameStateTensors ToTensors(bool includePlayHandScores)
    {
        return ToTensors(CPU, includePlayHandScores);
    }


    public GameStateTensors ToTensors(Device device, bool includePlayHandScores = false)
    {
        long[,] fullHand = new long[_addedGameStateCount, GameData.HandSize];
        long[,] remainingDeck = new long[_addedGameStateCount, 52];
        long[] handsAndDiscards = new long[_addedGameStateCount];
        float[,] score2D = new float[_addedGameStateCount, 1];
        float[,] playHandScores = includePlayHandScores ? new float[_addedGameStateCount, PlayHandScoreCount] : null;

        for (int stateIndex = 0; stateIndex < _addedGameStateCount; ++stateIndex)
        {
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                fullHand[stateIndex, cardIndex] = _fullHand[stateIndex, cardIndex];

            for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                remainingDeck[stateIndex, cardIndex] = _remainingDeck[stateIndex, cardIndex];

            handsAndDiscards[stateIndex] = _handsAndDiscards[stateIndex];
            score2D[stateIndex, 0] = _score[stateIndex];

            if (!includePlayHandScores)
                continue;

            for (int handIndex = 0; handIndex < PlayHandScoreCount; ++handIndex)
                playHandScores[stateIndex, handIndex] = _playHandScores[stateIndex, handIndex];
        }

        return new()
        {
            FullHand = tensor(fullHand, dtype: ScalarType.Int64, device: device),
            RemainingDeck = tensor(remainingDeck, dtype: ScalarType.Int64, device: device),
            HandsAndDiscards = tensor(handsAndDiscards, dtype: ScalarType.Int64, device: device),
            PlayHandScores = includePlayHandScores ? tensor(playHandScores, dtype: ScalarType.Float32, device: device) : null,
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


    void WritePlayHandScores(GameState gameState, int stateIndex)
    {
        int[][] playHandOptions = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        float roundScoreBefore = (float)gameState.ScoringState.CurrentRoundTotalChips;
        int handCardCount = gameState.HandState.HandCardCount;

        for (int handIndex = 0; handIndex < playHandOptions.Length; ++handIndex)
        {
            int[] cardIndices = playHandOptions[handIndex];
            if (cardIndices[^1] >= handCardCount)
            {
                _playHandScores[stateIndex, handIndex] = 0f;
                continue;
            }

            UseHandMove useHandMove = new(false, cardIndices);
            useHandMove.Apply(gameState);
            float roundScoreAfter = (float)gameState.ScoringState.CurrentRoundTotalChips;
            _playHandScores[stateIndex, handIndex] = (roundScoreAfter - roundScoreBefore) / 300f;
            useHandMove.Revert(gameState);
        }
    }
}
