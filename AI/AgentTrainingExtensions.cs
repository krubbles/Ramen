namespace Ramen.AI;

using Ramen.Game;
using static TorchSharp.torch;

/// <summary>
/// Extension methods for RamenAgent that provide training-related functionality.
/// </summary>
public static class AgentTrainingExtensions
{
    /// <summary>
    /// Makes a move based on the policy model's predicted probability distribution.
    /// Also generates a training sample with the chosen move and <paramref name="sampleCount"/> other moves.
    /// Intended to create PPO/GRPO training data.
    /// </summary>
    public static PolicyTrainingSample MakeMoveAndTrainingSample(this RamenAgent agent, float temp)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        agent.GameState.AdvanceToNextPlayerChoice();

        if (agent.GameIsDone())
            return null;

        (UseHandTensors moveTensors, Tensor probs) = agent.GetPolicyProbDist(temp);
        
        return null; // not currently in use
    }

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
    /// Creates a policy training sample from move tensors and probability distribution.
    /// </summary>
    public static PolicyTrainingSample CreatePolicyTrainingSample(this RamenAgent agent, UseHandTensors useHandTensors, Tensor probs, Tensor target, Tensor mask)
    {
        PolicyTrainingSample sample = new()
        {
            SamplingProb = probs.DetachFromDisposeScope(),
            StateTensors = agent.GameStateTensors.Clone().DetachFromDisposeScope(),
            UseHandTensors = useHandTensors.DetachFromDisposeScope(),
            Target = target.DetachFromDisposeScope(),
            Mask = mask.DetachFromDisposeScope(),
        };
        return sample;
    }

    /// <summary>
    /// Creates a Monte Carlo training sample for the given move indices.
    /// </summary>
    public static PolicyTrainingSample CreateMonteCarloTrainingSample(this RamenAgent agent, MoveSampleAnnotationData[] moveIndices)
    {
        if (!agent.GameState.IsPlayerChoice)
                throw new ArgumentException("Cannot create a training sample because gamestate is not at a player choice.");
        (UseHandTensors useHandTensors, int moveCount) = agent.CreateUseHandTensors();
        Tensor target = zeros(1, moveCount);
        target[moveIndices[0].MoveIndex] = 1f;
        Tensor mask = zeros(1, moveCount);
        Tensor probs = ones(1, moveCount);
        for (int i = 0; i < moveIndices.Length; ++i)
        {
            mask[moveIndices[i].MoveIndex] = 1f;
            probs[0, moveIndices[i].MoveIndex] = (float)Math.Exp(-moveIndices[i].NLProbTimes1K / 1000.0);
        }
        return agent.CreatePolicyTrainingSample(useHandTensors, probs, target, mask);
    }

    /// <summary>
    /// Samples moves based on the policy model's prediction for training data generation.
    /// </summary>
    private static (Move[] sampledMoves, UseHandTensors sampledMoveTensors, Tensor sampledProbs) SampleMoves(this RamenAgent agent, Move[] moves, UseHandTensors moveTensors, Tensor probs, int sampleCount)
    {
        Tensor indices = multinomial(probs.view([-1]), sampleCount, replacement: false);
        Tensor sampledProbs = probs.index_select(dim: 1, indices);
        UseHandTensors sampledMoveTensors = moveTensors.IndexSelect(dim: 1, indices);

        long[] indicesArray = indices.data<long>().ToArray();
        Move[] sampledMoves = new Move[sampleCount];
        for (int i = 0; i < sampledMoves.Length; ++i)
            sampledMoves[i] = moves[indicesArray[i]];

        indices.Dispose();

        return (sampledMoves, sampledMoveTensors, sampledProbs);
    }
}