namespace Ramen.AI;

using System;
using System.Collections.Generic;
using Ramen.Game;
using static TorchSharp.torch;

/// <summary>
/// Batch-first AI agent that plays Balatro across multiple game states.
/// </summary>
public class NewAgent
{
    /// <summary>
    /// The Balatro game states this AI agent is attached to.
    /// </summary>
    public readonly GameState[] GameStates;

    /// <summary>
    /// A reference to the policy network used by this agent.
    /// </summary>
    public readonly IPolicyModel Model;

    /// <summary>
    /// The PS-RNG used by this agent to make decisions like which move to play.
    /// </summary>
    public readonly FastRandom Random;

    public NewAgent(GameState[] gameStates, IPolicyModel model)
    {
        GameStates = gameStates;
        Model = model;
        Random = FastRandom.SeededByClock();
    }

    /// <summary>
    /// Returns whether or not the game is complete from the agent's perspective.
    /// </summary>
    public bool[] GameIsDone()
    {
        bool[] results = new bool[GameStates.Length];
        for (int i = 0; i < GameStates.Length; ++i)
            results[i] = IsGameDone(GameStates[i]);
        return results;
    }

    /// <summary>
    /// Returns the agent's reward at the current game states.
    /// </summary>
    public float[] GetCurrentReward()
    {
        float[] results = new float[GameStates.Length];
        for (int i = 0; i < GameStates.Length; ++i)
            results[i] = GetReward(GameStates[i]);
        return results;
    }

    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalChips >= 300)
            return 1f + gameState.HandState.RemainingHands * 0.2f;
        return (float)gameState.ScoringState.CurrentRoundTotalChips / 1000f;
    }

    /// <summary>
    /// Makes a move based on the policy model's predicted probability distribution.
    /// If <paramref name="annotatePolicy"/> is true, the policy distribution used to make the move
    /// is saved to the move history using <see cref="AnnotatingDataMove"/>.
    /// </summary>
    public void MakeMove(float temp, bool annotatePolicy = false)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        // Advance all states to the next player choice and collect active indices.
        List<int> activeIndices = new();
        for (int i = 0; i < GameStates.Length; ++i)
        {
            GameState gameState = GameStates[i];
            gameState.AdvanceToNextPlayerChoice();
            if (!IsGameDone(gameState))
                activeIndices.Add(i);
        }

        if (activeIndices.Count == 0)
            return;

        // Build the active state batch.
        GameState[] activeStates = new GameState[activeIndices.Count];
        for (int i = 0; i < activeStates.Length; ++i)
            activeStates[i] = GameStates[activeIndices[i]];

        // Evaluate the policy in a single batch.
        (UseHandTensors moveTensors, Tensor probs) = GetPolicyProbDist(temp, activeStates);

        // Sample and apply one move per active state.
        Tensor indexTensor = multinomial(probs, num_samples: 1);
        long[] indices = indexTensor.data<long>().ToArray();

        for (int i = 0; i < activeStates.Length; ++i)
        {
            int gameStateIndex = activeIndices[i];
            int chosenIndex = (int)indices[i];

            UseHandMove move = MoveForIndex(gameStateIndex, chosenIndex);
            move.Apply(GameStates[gameStateIndex]);

            if (annotatePolicy)
                AnnotatePolicy(GameStates[gameStateIndex], probs, i);
        }
    }

    /// <summary>
    /// Returns the move for the given move index in a specific game state.
    /// </summary>
    public UseHandMove MoveForIndex(int gameStateIndex, int index)
    {
        GameState gameState = GameStates[gameStateIndex];
        int[][] useHandOptions = Combinatorics.GetCombinations(
            setSize: gameState.HandState.HandCardCount,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        return new UseHandMove(index % 2 == 1, useHandOptions[index / 2]);
    }

    /// <summary>
    /// Returns the policy model's probability distributions for all game states.
    /// </summary>
    public float[][] GetPolicyProbDistManaged(float temp)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        (UseHandTensors moveTensors, Tensor probs) = GetPolicyProbDist(temp, GameStates);

        int batchSize = (int)probs.size(0);
        int moveCount = (int)probs.size(1);
        float[] flat = probs.data<float>().ToArray();

        float[][] results = new float[batchSize][];
        for (int b = 0; b < batchSize; ++b)
        {
            float[] row = new float[moveCount];
            Array.Copy(flat, b * moveCount, row, 0, moveCount);
            results[b] = row;
        }
        return results;
    }

    /// <summary>
    /// Returns the policy model's predicted probability distribution for the best next move.
    /// Returned probs is a batch x N tensor where N is the number of moves.
    /// </summary>
    public (UseHandTensors moveTensors, Tensor probs) GetPolicyProbDist(float temp)
    {
        return GetPolicyProbDist(temp, GameStates);
    }

    /// <summary>
    /// Embeds a list of moves into tensors for all game states.
    /// </summary>
    public (UseHandTensors useHandTensors, int moveCount) CreateUseHandTensors()
    {
        return CreateUseHandTensors(GameStates);
    }

    (UseHandTensors useHandTensors, int moveCount) CreateUseHandTensors(GameState[] gameStates)
    {
        // Precompute combination count (assumes hand size is consistent across the batch).
        int useHandCount = Combinatorics.CalculateCombinationCount(
            setSize: gameStates[0].HandState.HandCardCount,
            minSubsetSize: 1,
            maxSubsetSize: 5);

        // Build score tensor in a single allocation.
        float[,] scores = new float[gameStates.Length, useHandCount];

        // Populate scores per state by simulating each use-hand move.
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            int[][] cardIndicesEnumerator = Combinatorics.GetCombinations(
                setSize: gameState.HandState.HandCardCount,
                minSubsetSize: 1,
                maxSubsetSize: 5);

            int move = 0;
            for (int i = 0; i < cardIndicesEnumerator.Length; ++i)
            {
                UseHandMove useHandMove = new(false, cardIndicesEnumerator[i]);
                useHandMove.Apply(gameState);
                scores[stateIndex, move++] = (float)gameState.ScoringState.CurrentRoundTotalChips / 300f;
                useHandMove.Revert(gameState);
            }
        }

        UseHandTensors useHandTensors = new()
        {
            Score = tensor(scores),
        };

        return (useHandTensors, useHandCount * 2);
    }

    (UseHandTensors moveTensors, Tensor probs) GetPolicyProbDist(float temp, GameState[] gameStates)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameStateTensors gameStateTensors = CreateGameStateTensors(gameStates);
        (UseHandTensors useHandTensors, int moveCount) = CreateUseHandTensors(gameStates);

        useHandTensors.MoveToOuterDisposeScope();

        Tensor logits = Model.GetPolicyLogits(gameStateTensors, useHandTensors);

        Tensor discardMask = BuildDiscardMask(gameStates, moveCount, logits.device);
        logits += discardMask;

        Tensor probs = (logits / MathF.Max(temp, 0.0001f)).softmax(1).MoveToOuterDisposeScope();
        return (useHandTensors, probs);
    }

    static GameStateTensors CreateGameStateTensors(GameState[] gameStates)
    {
        // Build the full 3D card tensors in single allocations.
        Tensor fullHand = TensorizeHandBatch(gameStates, cardCount: GameData.HandSize);
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

    static Tensor TensorizeHandBatch(GameState[] gameStates, int cardCount)
    {
        long[,,] cards = new long[gameStates.Length, cardCount, 2];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            ReadOnlySpan<Card> hand = gameStates[stateIndex].HandState.Hand;
            for (int i = 0; i < cardCount; ++i)
            {
                if (i < hand.Length)
                {
                    Card card = hand[i];
                    cards[stateIndex, i, 0] = card.Rank - 2;
                    cards[stateIndex, i, 1] = (int)card.Suit;
                }
            }
        }

        return tensor(cards);
    }

    static Tensor TensorizeRemainingDeckBatch(GameState[] gameStates, int cardCount)
    {
        long[,,] cards = new long[gameStates.Length, cardCount, 2];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            ReadOnlySpan<Card> deck = gameStates[stateIndex].DeckState.RemainingDeck;
            for (int i = 0; i < cardCount; ++i)
            {
                if (i < deck.Length)
                {
                    Card card = deck[i];
                    cards[stateIndex, i, 0] = card.Rank - 2;
                    cards[stateIndex, i, 1] = (int)card.Suit;
                }
            }
        }

        return tensor(cards);
    }

    static Tensor TensorizeScalarBatch(GameState[] gameStates, Func<GameState, float> selector, int width)
    {
        float[,] values = new float[gameStates.Length, width];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
            values[stateIndex, 0] = selector(gameStates[stateIndex]);
        return tensor(values);
    }

    static Tensor TensorizeHandsAndDiscardsBatch(GameState[] gameStates)
    {
        long[] values = new long[gameStates.Length];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
            values[stateIndex] = GetHandsAndDiscardsValue(gameStates[stateIndex]);
        return tensor(values, ScalarType.Int64);
    }

    static Tensor BuildDiscardMask(GameState[] gameStates, int moveCount, Device device)
    {
        float[,] mask = new float[gameStates.Length, moveCount];
        int useHandCount = moveCount / 2;
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            if (gameStates[stateIndex].HandState.RemainingDiscards != 0)
                continue;

            for (int handIndex = 0; handIndex < useHandCount; ++handIndex)
                mask[stateIndex, handIndex * 2 + 1] = -1e8f;
        }

        return tensor(mask).to(device);
    }

    static int GetHandsAndDiscardsValue(GameState gameState)
    {
        return gameState.HandState.RemainingHands * 5 + gameState.HandState.RemainingDiscards;
    }

    static float GetScoreValue(GameState gameState)
    {
        return (float)gameState.ScoringState.CurrentRoundTotalChips;
    }

    static bool IsGameDone(GameState gameState)
    {
        return gameState.GameIsDone;
    }

    static void AnnotatePolicy(GameState gameState, Tensor probs, int batchIndex)
    {
        Tensor row = probs[batchIndex];
        float[] probDist = row.data<float>().ToArray();
        ushort[] encodedProbs = new ushort[probDist.Length];
        for (int i = 0; i < probDist.Length; i++)
            encodedProbs[i] = AnnotatingDataMove.EncodeProb(probDist[i]);
        AnnotatingDataMove annotation = AnnotatingDataMove.FromArray(encodedProbs);
        annotation.Apply(gameState);
    }
}
