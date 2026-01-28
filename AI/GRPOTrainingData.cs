namespace Ramen.AI;

using System;
using System.Collections.Generic;
using Ramen.Game;
using static TorchSharp.torch;

public static class GRPOTrainingData
{
    public static TrainingDataStats GenerateTrainingData(PolicyModel model, int games, int sampleCount, int groupSize = 128)
    {
        TrainingDataStats stats = new();
        GameState gameState = new(new());
        RamenAgent agent = new(gameState, model);

        using var scope = NewDisposeScope();

        while (stats.GamesCount < games)
        {
            int batchSize = Math.Min(groupSize, games - stats.GamesCount);
            List<PolicyTrainingSample>[] groupSamples = new List<PolicyTrainingSample>[batchSize];
            float[] groupRewards = new float[batchSize];

            int startingMoveCount = gameState.MoveState.MoveHistory.Count;
            using (no_grad())
            {
                for (int group = 0; group < batchSize; ++group)
                {
                    gameState.Reseed();
                    List<PolicyTrainingSample> gameSamples = new();
                    groupSamples[group] = gameSamples;

                    while (!agent.GameIsDone())
                    {
                        PolicyTrainingSample sample = agent.MakeMoveAndTrainingSample(sampleCount);
                        if (sample != null)
                            gameSamples.Add(sample);
                    }

                    groupRewards[group] = agent.GetCurrentReward();

                    while (gameState.MoveState.MoveHistory.Count > startingMoveCount)
                        gameState.MoveState.RevertLastMove();
                }
            }

            ApplyGroupStatsAndSamples(stats, groupRewards, groupSamples);
        }

        return stats;
    }

    static void ApplyGroupStatsAndSamples(TrainingDataStats stats, float[] groupRewards, List<PolicyTrainingSample>[] groupSamples)
    {
        float sum = 0f;
        float sqSum = 0f;

        for (int group = 0; group < groupRewards.Length; ++group)
        {
            sum += groupRewards[group];
            sqSum += groupRewards[group] * groupRewards[group];
        }

        stats.TotalReward += sum;
        stats.TotalSquaredReward += sqSum;
        stats.GamesCount += groupRewards.Length;

        float mean = sum / groupRewards.Length;
        float ss = sqSum - sum * mean;
        float stdDev = MathF.Sqrt(ss / Math.Max(1, groupRewards.Length - 1));

        Array.Sort(groupRewards, groupSamples);

        lock (TrainingData.PolicyData)
        {
            for (int group = 0; group < groupSamples.Length; ++group)
            {
                float advantage = (groupRewards[group] - mean) / MathF.Max(stdDev, 1e-8f);
                if (float.IsNaN(advantage) || float.IsInfinity(advantage))
                    advantage = 0f;

                List<PolicyTrainingSample> nodes = groupSamples[group];
                for (int depth = 0; depth < nodes.Count; ++depth)
                {
                    PolicyTrainingSample node = nodes[depth];
                    stats.NodesCount++;
                    node.Advantage = tensor(advantage).unsqueeze(0).DetachFromDisposeScope();
                    TrainingData.PolicyData.Add(node);

                    if (depth < TrainingDataStats.MaxDepth)
                    {
                        stats.TotalNLProbByDepth[depth] += node.ChosenMoveNLProb;
                        stats.CountByDepth[depth] += 1;
                    }
                }
            }
        }
    }
}
