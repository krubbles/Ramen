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
    public static PolicyTrainingSample MakeMoveAndTrainingSample(this RamenAgent agent, float temp, int sampleCount = 20)
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

        return CreatePolicyTrainingSample(agent, sampledMoveTensors, sampledProbs);
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

        (Move[] moves, MoveTensors moveTensors, Tensor probs) = agent.GetPolicyProbDist(temp: 1f);
        Tensor indices = multinomial(probs, sampleCount);
        long[] indicesArray = indices.data<long>().ToArray();

        float[] avgRewards = new float[sampleCount];
        int initialStep = agent.GameState.MoveState.MoveStep;

        for (int i = 0; i < sampleCount; i++)
        {
            Move candidateMove = moves[indicesArray[i]];

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
        moves[indicesArray[bestIndexIndex]].Apply(agent.GameState);

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
    /// Creates an evaluation training sample from move tensors and probability distribution.
    /// </summary>
    public static PolicyTrainingSample CreatePolicyTrainingSample(this RamenAgent agent, MoveTensors moveTensors, Tensor probs)
    {
        PolicyTrainingSample sample = new()
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
    public static PolicyTrainingSample CreateMonteCarloTrainingSample(this RamenAgent agent, MoveSampleAnnotationData[] moveIndices)
    {
        Move[] moves = agent.GameState.GetMoveOptions();
        Move[] sampledMoves = new Move[moveIndices.Length];
        for (int i = 0; i < moveIndices.Length; ++i)
            sampledMoves[i] = moves[moveIndices[i].MoveIndex];
        MoveTensors sampledMoveTensors = agent.CreateMoveTensors(sampledMoves);
        float[] sampledProbs = new float[moveIndices.Length];
        for (int i = 0; i < moveIndices.Length; ++i)
            sampledProbs[i] = MathF.Exp(moveIndices[i].NLProbTimes1K / -1000f); // it's fixed precision so the encoding is needed
        return CreatePolicyTrainingSample(agent, sampledMoveTensors, tensor(sampledProbs).unsqueeze_(0));
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