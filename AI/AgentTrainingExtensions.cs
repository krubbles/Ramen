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
    public static EvaluationTrainingSample MakeMoveAndTrainingSample(this RamenAgent agent, float temp, int sampleCount = 20)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        agent.GameState.AdvanceToNextPlayerChoice();

        if (agent.GameIsDone())
            return null;

        (Move[] moves, MoveTensors moveTensors, Tensor probs) = agent.GetPolicyProbDist(temp);
        (Move[] sampledMoves, MoveTensors sampledMoveTensors, Tensor sampledProbs) = SampleMoves(agent, moves, moveTensors, probs, sampleCount);
        Tensor indices = multinomial(probs.view([-1]), sampleCount, replacement: false);

        moves[(int)indices[0].item<long>()].Apply(agent.GameState);

        return CreateEvaluationTrainingSample(agent, sampledMoveTensors, sampledProbs);
    }

    /// <summary>
    /// Samples <paramref name="sampleCount"/> moves based on the policy model's prediction,
    /// then plays <paramref name="continuationCount"/> continuations after each move.
    /// Calculates the average reward for all continuations from each move, and makes the move with the best average.
    /// </summary>
    /// <returns>An array containing the indices to all sampled moves, with the highest average reward move at index 0.</returns>
    public static ushort[] MakeMoveMonteCarlo(this RamenAgent agent, float temp, int sampleCount, int continuationCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        agent.GameState.AdvanceToNextPlayerChoice();

        if (agent.GameIsDone())
            return null;

        (Move[] moves, MoveTensors moveTensors, Tensor probs) = agent.GetPolicyProbDist(temp);

        long[] indices = multinomial(probs, sampleCount).data<long>().ToArray();

        float[] avgRewards = new float[sampleCount];
        int initialStep = agent.GameState.MoveState.MoveStep;

        for (int i = 0; i < sampleCount; i++)
        {
            Move candidateMove = moves[indices[i]];
            float totalReward = 0f;

            candidateMove.Apply(agent.GameState);
            int afterCandidateStep = agent.GameState.MoveState.MoveStep;

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
            (indices[0], indices[bestIndexIndex]) = (indices[bestIndexIndex], indices[0]);
        }

        // Apply the best move (now at index 0)
       moves[indices[bestIndexIndex]].Apply(agent.GameState);

        ushort[] compressedIndices = new ushort[indices.Length];
        for (int i = 0; i < compressedIndices.Length; ++i)
        {
            compressedIndices[i] = (ushort)indices[i];
        }
        return compressedIndices;
    }

    /// <summary>
    /// Creates an evaluation training sample from move tensors and probability distribution.
    /// </summary>
    public static EvaluationTrainingSample CreateEvaluationTrainingSample(this RamenAgent agent, MoveTensors moveTensors, Tensor probs)
    {
        EvaluationTrainingSample sample = new()
        {
            MoveProbDist = probs.DetachFromDisposeScope(),
            State = agent.Tensors.Clone().DetachFromDisposeScope(),
            Moves = moveTensors.DetachFromDisposeScope(),
            ChosenMoveNLProb = -MathF.Log(Math.Max(probs[0, 0].item<float>(), 1e-9f)),
        };
        return sample;
    }

    /// <summary>
    /// Creates a Monte Carlo training sample for the given move indices.
    /// </summary>
    public static EvaluationTrainingSample CreateMonteCarloTrainingSample(this RamenAgent agent, ushort[] moveIndices, float temp)
    {
        Move[] moves = agent.GameState.GetMoveOptions();
        Move[] sampledMoves = new Move[moveIndices.Length];
        for (int i = 0; i < moveIndices.Length; ++i)
            sampledMoves[i] = moves[moveIndices[i]];
        (MoveTensors sampledMoveTensors, Tensor sampledProbs) = agent.GetPolicyProbDistForMoves(temp, sampledMoves);
        return CreateEvaluationTrainingSample(agent, sampledMoveTensors, sampledProbs);
    }

    /// <summary>
    /// Samples moves based on the policy model's prediction for training data generation.
    /// </summary>
    private static (Move[] sampledMoves, MoveTensors sampledMoveTensors, Tensor sampledProbs) SampleMoves(this RamenAgent agent, Move[] moves, MoveTensors moveTensors, Tensor probs, int sampleCount)
    {
        Tensor indices = multinomial(probs.view([-1]), sampleCount, replacement: false);
        Tensor sampledProbs = probs.index_select(dim: 1, indices);
        MoveTensors sampledMoveTensors = moveTensors.IndexSelect(dim: 1, indices);

        long[] indicesArray = indices.data<long>().ToArray();
        Move[] sampledMoves = new Move[sampleCount];
        for (int i = 0; i < sampledMoves.Length; ++i)
            sampledMoves[i] = moves[indicesArray[i]];

        indices.Dispose();

        return (sampledMoves, sampledMoveTensors, sampledProbs);
    }
}