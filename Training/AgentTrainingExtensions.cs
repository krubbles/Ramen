namespace Ramen.Training;

using System;
using System.Collections.Generic;
using Ramen.AI;
using Ramen.Game;
using Tensorboard;
using static TorchSharp.torch;

/// <summary>
/// Extension methods for RamenAgent that provide training-related functionality.
/// </summary>
public static class AgentTrainingExtensions
{
    /// <summary>
    /// Makes a move based on the policy model's predicted probability distribution.
    /// Also generates a GRPO training sample with <paramref name="sampleCount"/> sampled moves.
    /// </summary>
    public static PolicyTrainingSample MakeMoveAndTrainingSample(this RamenAgent agent, int sampleCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        agent.GameState.AdvanceToNextPlayerChoice();

        if (agent.GameIsDone())
            return null;

        (UseHandTensors useHandTensors, Tensor probs) = agent.GetPolicyProbDist(temp: 1f);
        int moveCount = (int)probs.size(1);
        int clampedSampleCount = Math.Clamp(sampleCount, 1, moveCount);

        Tensor indices = multinomial(probs.view([-1]), clampedSampleCount, replacement: false);
        long[] indicesArray = [.. indices.data<long>()];
        int chosenIndex = (int)indicesArray[0];

        Tensor sampledProbs = probs.index_select(dim: 1, indices);

        PolicyTrainingSample sample = new()
        {
            SamplingProb = sampledProbs,
            StateTensors = agent.GameStateTensors.Clone(),
            UseHandTensors = useHandTensors,
            MoveIndices = indices.unsqueeze(0),
            ChosenMoveNLProb = -MathF.Log(probs[0, chosenIndex].item<float>() + 1e-9f),
        };
        sample.DetachFromScope();

        UseHandMove move = agent.MoveForIndex(chosenIndex);
        move.Apply(agent.GameState);

        return sample;
    }

    /// <summary>
    /// Makes moves based on the policy model's predicted probability distribution for a batch of game states.
    /// Also generates a GRPO training sample with <paramref name="sampleCount"/> sampled moves.
    /// </summary>
    public static PolicyTrainingSample[] MakeMoveAndTrainingSample(this PolicyOnlyAgent agent, GameState[] gameStates, int sampleCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var p_funcScope = ProfileScope.New(nameof(MakeMoveAndTrainingSample));

        // go through each state, advance them to the next player choice, and build a list of states that aren't done. 
        PolicyTrainingSample[] outputSamples = new PolicyTrainingSample[gameStates.Length];
        List<int> activeIndices = new();
        for (int i = 0; i < gameStates.Length; ++i)
        {
            GameState gameState = gameStates[i];
            gameState.AdvanceToNextPlayerChoice();
            if (!agent.IsGameDone(gameState))
                activeIndices.Add(i);
        }

        // all states are done, nothing to do
        if (activeIndices.Count == 0)
            return outputSamples;

        // Build active state batch and evaluate policy once.
        GameState[] activeStates = new GameState[activeIndices.Count];
        for (int i = 0; i < activeStates.Length; ++i)
            activeStates[i] = gameStates[activeIndices[i]];

        // get a batch of policy probabilities for each active state as a single batched tensor
        (GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor probs) = agent.GetPolicyProbDist(temp: 1f, activeStates); // returned tensors are on the gpu.


        int moveCount = (int)probs.size(1);
        int clampedSampleCount = Math.Clamp(sampleCount, 1, moveCount);

        // sample move indices using the policy. Index 0 for each item in the batch is the chosen move, 
        // the remaining are used for sampled softmax training
        Tensor sampleIndices = multinomial(probs, clampedSampleCount, replacement: false);

        Tensor chosenMoveIndices = sampleIndices.select(dim: 1, index: 0);
        Tensor sampledProbs = probs.gather(dim: 1, sampleIndices);
        Tensor chosenProbs = sampledProbs.select(dim: 1, index: 0);

        // readback the chosen move index and prob, so the move can be made and nl prob can be stored in the sample
        Profiling.Enter("ChosenMoveReadback");
        long[] chosenMoveIndicesManaged = [.. chosenMoveIndices.to(CPU).data<long>()];
        float[] chosenProbsManaged = [.. chosenProbs.to(CPU).data<float>()];
        Profiling.Exit("ChosenMoveReadback");
        
        Profiling.Enter("BuildSamples");
        // Sample and apply one move per active state while creating one sample per state.
        for (int activeIndex = 0; activeIndex < activeStates.Length; ++activeIndex)
        {
            // Clone per-sample slices so each sample owns compact tensors instead of views to full step batches.
            Tensor sampleSamplingProb = sampledProbs[activeIndex..(activeIndex + 1)].clone();
            GameStateTensors sampleStateTensors = gameStateTensors.GetBatch(activeIndex, activeIndex + 1).Clone();
            UseHandTensors sampleUseHandTensors = useHandTensors.GetBatch(activeIndex, activeIndex + 1).Clone();
            Tensor sampleMoveIndices = sampleIndices[activeIndex..(activeIndex + 1)].clone();

            PolicyTrainingSample sample = new()
            {
                SamplingProb = sampleSamplingProb,
                StateTensors = sampleStateTensors,
                UseHandTensors = sampleUseHandTensors,
                MoveIndices = sampleMoveIndices,
                ChosenMoveNLProb = -MathF.Log(chosenProbsManaged[activeIndex] + 1e-9f),
            };
            sample.DetachFromScope();

            // make the chosen move
            GameState state = activeStates[activeIndex];
            UseHandMove move = PolicyOnlyAgent.MoveForIndex(state, (int)chosenMoveIndicesManaged[activeIndex]);
            move.Apply(state);

            // save sample in the output buffer
            outputSamples[activeIndices[activeIndex]] = sample;
        }
        Profiling.Exit("BuildSamples"); 
        return outputSamples;
    }

    /// <summary>
    /// Creates a policy training sample from move tensors and probability distribution.
    /// </summary>
    public static PolicyTrainingSample CreatePolicyTrainingSample(this RamenAgent agent, UseHandTensors useHandTensors, Tensor probs, Tensor target, Tensor moveIndices)
    {
        PolicyTrainingSample sample = new()
        {
            SamplingProb = probs,
            StateTensors = agent.GameStateTensors.Clone(),
            UseHandTensors = useHandTensors,
            Target = target,
            MoveIndices = moveIndices,
        };
        sample.DetachFromScope();
        return sample;
    }

    #region NotInUse
    /// <summary>
    /// Samples <paramref name="sampleCount"/> moves based on the policy model's prediction,
    /// then plays <paramref name="continuationCount"/> continuations after each move.
    /// Calculates the average reward for all continuations from each move, and makes the move with the best average.
    /// </summary>
    /// <returns>An array containing the indices to all sampled moves, with the highest average reward move at index 0.</returns>
    public static MoveSampleAnnotationData[] MakeMoveMonteCarlo(this RamenAgent agent, float temp, int sampleCount, int continuationCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        agent.GameState.AdvanceToNextPlayerChoice();

        if (agent.GameIsDone())
            return null;

        (UseHandTensors moveTensors, Tensor probs) = agent.GetPolicyProbDist(temp: 1f);
        Tensor indices = multinomial(probs, sampleCount);
        long[] indicesArray = indices.data<long>().ToArray();

        float[] avgRewards = new float[sampleCount];
        int initialStep = agent.GameState.MoveState.MoveStep;

        for (int i = 0; i < sampleCount; i++)
        {
            Move candidateMove = agent.MoveForIndex((int)indicesArray[i]);

            candidateMove.Apply(agent.GameState);
            int afterCandidateStep = agent.GameState.MoveState.MoveStep;

            float totalReward = 0f;
            for (int c = 0; c < continuationCount; c++)
            {
                agent.GameState.Reseed();

                while (!agent.GameIsDone())
                {
                    agent.GameState.AdvanceToNextPlayerChoice();
                    agent.MakeMove(temp);
                }

                totalReward += agent.GetCurrentReward();

                agent.GameState.MoveState.RevertToStep(afterCandidateStep);
            }

            avgRewards[i] = totalReward / continuationCount;

            agent.GameState.MoveState.RevertToStep(initialStep);
        }

        // Find the best move based on average rewards
        int bestIndexIndex = 0;
        float bestReward = avgRewards[0];
        for (int i = 1; i < sampleCount; i++)
        {
            if (avgRewards[i] > bestReward)
            {
                bestReward = avgRewards[i];
                bestIndexIndex = i;
            }
        }

        // Swap the best move to index 0
        if (bestIndexIndex != 0)
        {
            (indicesArray[0], indicesArray[bestIndexIndex]) = (indicesArray[bestIndexIndex], indicesArray[0]);
        }

        // Apply the best move (now at index 0)
        agent.MoveForIndex((int)indicesArray[0]).Apply(agent.GameState);

        float[] sampledProbs = probs.index_select(dim: 1, indices.squeeze(0)).data<float>().ToArray();
        MoveSampleAnnotationData[] annotationData = new MoveSampleAnnotationData[indicesArray.Length];
        for (int i = 0; i < annotationData.Length; ++i)
        {
            annotationData[i].MoveIndex = (ushort)indicesArray[i];
            annotationData[i].NLProbTimes1K = (ushort)Math.Clamp(-MathF.Log(sampledProbs[i]) * 1000 + 0.5f, 0, ushort.MaxValue); // encoding for low-bit-depth
        }
        return annotationData;
    }
    /// <summary>
    /// Creates a Monte Carlo training sample for the given move indices.
    /// </summary>
    public static PolicyTrainingSample CreateMonteCarloTrainingSample(this RamenAgent agent, MoveSampleAnnotationData[] moveIndices)
    {
        if (!agent.GameState.IsPlayerChoice)
                throw new ArgumentException("Cannot create a training sample because gamestate is not at a player choice.");
        (UseHandTensors useHandTensors, int moveCount) = agent.CreateUseHandTensors();
        int[] indices = new int[moveIndices.Length];
        float[] probs = new float[moveIndices.Length];
        for (int i = 0; i < moveIndices.Length; ++i)
        {
            indices[i] = moveIndices[i].MoveIndex;
            probs[i] = (float)Math.Exp(-moveIndices[i].NLProbTimes1K / 1000.0);
        }
        Tensor moveIndexTensor = tensor(indices, ScalarType.Int64).unsqueeze(0);
        Tensor probsTensor = tensor(probs).unsqueeze(0);
        Tensor target = zeros(1, moveIndices.Length);
        target[0, 0] = 1f;
        return agent.CreatePolicyTrainingSample(useHandTensors, probsTensor, target, moveIndexTensor);
    }
    #endregion

    static GameStateTensors CreateGameStateTensors(GameState gameState)
    {
        GameStateTensors gameStateTensors = new()
        {
            FullHand = TensorizeHand(gameState, cardCount: GameData.HandSize),
            RemainingDeck = TensorizeRemainingDeck(gameState, cardCount: 52),
            HandsAndDiscards = TensorizeHandsAndDiscards(gameState),
            Score = TensorizeScore(gameState),
        };

        return gameStateTensors;
    }

    static Tensor TensorizeHand(GameState gameState, int cardCount)
    {
        long[,,] cards = new long[1, cardCount, 2];
        ReadOnlySpan<Card> hand = gameState.HandState.Hand;
        for (int i = 0; i < cardCount; ++i)
        {
            if (i < hand.Length)
            {
                Card card = hand[i];
                cards[0, i, 0] = card.Rank - 2;
                cards[0, i, 1] = (int)card.Suit;
            }
        }

        return tensor(cards);
    }

    static Tensor TensorizeRemainingDeck(GameState gameState, int cardCount)
    {
        long[,,] cards = new long[1, cardCount, 2];
        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int i = 0; i < cardCount; ++i)
        {
            if (i < deck.Length)
            {
                Card card = deck[i];
                cards[0, i, 0] = card.Rank - 2;
                cards[0, i, 1] = (int)card.Suit;
            }
        }

        return tensor(cards);
    }

    static Tensor TensorizeHandsAndDiscards(GameState gameState)
    {
        int handsAndDiscards = gameState.HandState.RemainingHands * 5 + gameState.HandState.RemainingDiscards;
        long[] values = [handsAndDiscards];
        return tensor(values, ScalarType.Int64);
    }

    static Tensor TensorizeScore(GameState gameState)
    {
        float score = (float)gameState.ScoringState.CurrentRoundTotalChips;
        float[,] values = new float[1, 1];
        values[0, 0] = score;
        return tensor(values);
    }

}
