namespace Ramen.AI;

using System;
using System.Collections.Generic;
using Ramen.Agents;
using Ramen.Game;
using static TorchSharp.torch;

/// <summary>
/// Batch-first AI agent that plays Balatro across multiple game states.
/// </summary>
public class PolicyOnlyAgent : IAgent, IDisposable
{
    readonly bool _ownsModel;

    /// <summary>
    /// A reference to the policy network used by this agent.
    /// </summary>
    public readonly IPolicyModel Model;

    /// <summary>
    /// The PS-RNG used by this agent to make decisions like which move to play.
    /// </summary>
    public readonly FastRandom Random;

    public PolicyOnlyAgent(IPolicyModel model, bool ownsModel = false)
    {
        Model = model;
        _ownsModel = ownsModel;
        Random = FastRandom.SeededByClock();
    }

    /// <summary>
    /// Returns true if the game is done from the perspective of this agent. Is sometimes but not always equivalent to <see cref="GameState.GameIsDone"/>.
    /// </summary>
    public bool IsGameDone(GameState gameState) => gameState.GameIsDone;

    public void Dispose()
    {
        if (_ownsModel && Model is IDisposable disposableModel)
            disposableModel.Dispose();
    }

    /// <summary>
    /// Makes a move based on the policy model's predicted probability distribution.
    /// If <paramref name="annotatePolicy"/> is true, the policy distribution used to make the move
    /// is saved to the move history using <see cref="AnnotatingDataMove"/>.
    /// </summary>
    public void MakeMove(float temp, bool annotatePolicy, params ReadOnlySpan<GameState> states)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        const int MaxActiveBatchSize = 32;

        // Advance all states to the next player choice and collect active indices.
        List<int> activeIndices = new();
        for (int i = 0; i < states.Length; ++i)
        {
            GameState gameState = states[i];
            gameState.AdvanceToNextPlayerChoice();
            if (!IsGameDone(gameState))
                activeIndices.Add(i);
        }

        if (activeIndices.Count == 0)
            return;

        // Process active states in chunks when the active set is large.
        for (int batchStart = 0; batchStart < activeIndices.Count; batchStart += MaxActiveBatchSize)
        {
            int batchSize = Math.Min(MaxActiveBatchSize, activeIndices.Count - batchStart);

            // Build the active state batch.
            GameState[] activeStates = new GameState[batchSize];
            for (int i = 0; i < activeStates.Length; ++i)
                activeStates[i] = states[activeIndices[batchStart + i]];

            // Evaluate the policy for the current active-state chunk.
            (GameStateTensors _, UseHandTensors _, Tensor probs) = GetPolicyProbDist(temp, activeStates);

            // Sample and apply one move per active state in this chunk.
            Tensor indexTensor = multinomial(probs, num_samples: 1);
            long[] indices = indexTensor.data<long>().ToArray();

            for (int i = 0; i < activeStates.Length; ++i)
            {
                GameState state = activeStates[i];
                int chosenIndex = (int)indices[i];

                UseHandMove move = MoveForIndex(state, chosenIndex);
                move.Apply(state);

                if (annotatePolicy)
                    AnnotatePolicy(state, probs, i);
            }
        }
    }

    /// <summary>
    /// Returns the policy model's probability distributions for all game states.
    /// </summary>
    public float[][] GetPolicy(float temp, params ReadOnlySpan<GameState> states)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        (GameStateTensors _, UseHandTensors _, Tensor probs) = GetPolicyProbDist(temp, states);

        probs = probs.to(CPU);
        
        int batchSize = (int)probs.size(0);
        int moveCount = (int)probs.size(1);
        float[] flat = probs.data<float>().ToArray();

        float[][] results = new float[batchSize][];
        for (int b = 0; b < batchSize; ++b)
        {
            if (IsGameDone(states[b]))
                continue;
            float[] row = new float[moveCount];
            Array.Copy(flat, b * moveCount, row, 0, moveCount);
            results[b] = row;
        }
        return results;
    }

    (UseHandTensors useHandTensors, int moveCount) CreateUseHandTensors(ReadOnlySpan<GameState> gameStates)
    {
        using var p_funcScope = ProfileScope.New(nameof(CreateUseHandTensors));

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

    /// <summary>
    /// Returns the policy model's probability distributions for all game states and the generated use-hand tensors. 
    /// Returned tensors are on the GPU. 
    /// </summary>
    public (GameStateTensors gameStateTensors, UseHandTensors moveTensors, Tensor probs) GetPolicyProbDist(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        using var dscope = NewDisposeScope();
        using var gscope = no_grad();
        using var pscope = ProfileScope.New(nameof(GetPolicyProbDist));

        GameStateTensors gameStateTensors = CreateGameStateTensors(gameStates);
        (UseHandTensors useHandTensors, int moveCount) = CreateUseHandTensors(gameStates);
        Tensor discardMask = BuildDiscardMask(gameStates, moveCount);

        Profiling.Enter("GetPolicyLogits");
        Tensor logits = Model.GetPolicyLogits(gameStateTensors, useHandTensors); // interface so it doesn't have a profile scope internally
        Profiling.Exit("GetPolicyLogits");

        logits += discardMask;

        Tensor probs = (logits / MathF.Max(temp, 0.0001f)).softmax(1).ToOuterScope();

        useHandTensors.ToOuterScope();
        gameStateTensors.ToOuterScope();
        return (gameStateTensors, useHandTensors, probs);
    }

    static GameStateTensors CreateGameStateTensors(ReadOnlySpan<GameState> gameStates)
    {
        using var pscope = ProfileScope.New(nameof(CreateGameStateTensors));

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

    /// <summary>
    /// Returns the move for the given move index in a specific game state.
    /// </summary>
    public static UseHandMove MoveForIndex(GameState state, int index)
    {
        int[][] useHandOptions = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        return new UseHandMove(state.HandState.RemainingDiscards >= 1 && index % 2 == 1, useHandOptions[index / 2]);
    }

    static Tensor TensorizeHandBatch(ReadOnlySpan<GameState> gameStates, int cardCount)
    {
        using var pscope = ProfileScope.New(nameof(TensorizeHandBatch));

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

    static Tensor BuildDiscardMask(ReadOnlySpan<GameState> gameStates, int moveCount)
    {
        using var dscope = NewDisposeScope();
        using var pscope = ProfileScope.New(nameof(BuildDiscardMask));
        
        Profiling.Enter("BuildManaged");
        float[,] mask = new float[gameStates.Length, 2];
        int useHandCount = moveCount / 2;
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            if (gameStates[stateIndex].HandState.RemainingDiscards == 0)
                mask[stateIndex, 1] = -1e6f;
        }
        Profiling.Exit("BuildManaged");

        Profiling.Enter("BuildTensor");
        Tensor maskNoRepeat = tensor(mask, device: CPU);
        Profiling.Exit("BuildTensor");

        Profiling.Enter("ToMPS");
        maskNoRepeat = maskNoRepeat.to(MPS);
        Profiling.Exit("ToMPS");

        Profiling.Enter("Repeat");
        Tensor result = maskNoRepeat.repeat([1, moveCount / 2]).ToOuterScope();
        Profiling.Exit("Repeat");

        return result;
    }

    static int GetHandsAndDiscardsValue(GameState gameState)
    {
        return gameState.HandState.RemainingHands * 5 + gameState.HandState.RemainingDiscards;
    }

    static float GetScoreValue(GameState gameState)
    {
        return (float)gameState.ScoringState.CurrentRoundTotalChips;
    }

    static void AnnotatePolicy(GameState gameState, Tensor probs, int batchIndex)
    {
        Tensor row = probs[batchIndex];
        float[] probDist = row.data<float>().ToArray();
        AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation(probDist);
        annotation.Apply(gameState);
    }
}
