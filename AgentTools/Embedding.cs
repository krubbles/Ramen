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

/// <summary>
/// Methods for embedding GameState and Move data into tensors for use in neural networks.
/// </summary>
public static class Embedding
{
    /// <summary>
    /// Embeds a batch of GameStates into tensors.
    /// </summary>
    public static GameStateTensors EmbedGameStates(ReadOnlySpan<GameState> gameStates)
    {
        using var pscope = ProfileScope.New(nameof(EmbedGameStates));

        // Build the full 3D card tensors in single allocations.
        Tensor fullHand = EmbedHands(gameStates, cardCount: GameData.HandSize);
        Tensor remainingDeck = TensorizeRemainingDeckBatch(gameStates, cardCount: 52);

        // Build scalar tensors in single allocations.
        Tensor handsAndDiscards = TensorizeHandsAndDiscardsBatch(gameStates);
        Tensor score = TensorizeScalarBatch(gameStates, GetScoreValue, 1);

        GameStateTensors gameStateTensors = new()
        {
            FullHand = fullHand,
            RemainingDeck = remainingDeck,
            HandsAndDiscards = handsAndDiscards,
            Score = score,
        };

        return gameStateTensors;
    }

    static Tensor EmbedHands(ReadOnlySpan<GameState> gameStates, int cardCount)
    {
        using var pscope = ProfileScope.New(nameof(EmbedHands));

        long[,] cards = new long[gameStates.Length, cardCount];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            ReadOnlySpan<Card> hand = gameStates[stateIndex].HandState.Hand;
            for (int i = 0; i < cardCount; ++i)
            {
                if (i < hand.Length)
                    cards[stateIndex, i] = hand[i].ToIndex();
            }
        }

        return tensor(cards);
    }

    static Tensor TensorizeRemainingDeckBatch(ReadOnlySpan<GameState> gameStates, int cardCount)
    {
        long[,] cards = new long[gameStates.Length, cardCount];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            ReadOnlySpan<Card> deck = gameStates[stateIndex].DeckState.RemainingDeck;
            for (int i = 0; i < cardCount; ++i)
            {
                if (i < deck.Length)
                    cards[stateIndex, i] = deck[i].ToIndex();
            }
        }

        return tensor(cards);
    }

    static Tensor TensorizeScalarBatch(ReadOnlySpan<GameState> gameStates, Func<GameState, float> selector, int width)
    {
        float[,] values = new float[gameStates.Length, width];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
            values[stateIndex, 0] = selector(gameStates[stateIndex]);
        return tensor(values);
    }

    static Tensor TensorizeHandsAndDiscardsBatch(ReadOnlySpan<GameState> gameStates)
    {
        long[] values = new long[gameStates.Length];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
            values[stateIndex] = GetHandsAndDiscardsValue(gameStates[stateIndex]);
        return tensor(values, ScalarType.Int64);
    }

    static float GetScoreValue(GameState gameState)
    {
        return (float)gameState.ScoringState.CurrentRoundTotalChips;
    }

    static int GetHandsAndDiscardsValue(GameState gameState)
    {
        return gameState.HandState.RemainingHands * 5 + gameState.HandState.RemainingDiscards;
    }
}
