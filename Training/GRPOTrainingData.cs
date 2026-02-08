namespace Ramen.Training;

using System;
using System.Collections.Generic;
using Ramen.AI;
using Ramen.Game;
using static TorchSharp.torch;

public static class GRPOTrainingData
{
    public static TrainingDataStats GenerateTrainingData(IPolicyModel model, int games, int sampleCount, int groupSize = 128)
    {
        TrainingDataStats stats = new();
        GameState gameState = new(new());
        RamenAgent agent = new(gameState, model);

        using var scope = NewDisposeScope();

        // Generate grouped rollouts.
        while (stats.GamesCount < games)
        {
            // Prepare batch containers.
            int batchSize = Math.Min(groupSize, games - stats.GamesCount);
            List<PolicyTrainingSample>[] groupSamples = new List<PolicyTrainingSample>[batchSize];
            float[] groupRewards = new float[batchSize];

            int startingMoveCount = gameState.MoveState.MoveHistory.Count;
            using (no_grad())
            {
                // Play each game in the batch.
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
                    FillEntropyScalars(gameSamples);

                    while (gameState.MoveState.MoveHistory.Count > startingMoveCount)
                        gameState.MoveState.RevertLastMove();
                }
            }
        }

        return stats;
    }

    static void FillEntropyScalars(List<PolicyTrainingSample> samples)
    {
        float totalNlProbAfterwards = 0f;
        for (int i = samples.Count - 1; i >= 0; i--)
        {
            PolicyTrainingSample sample = samples[i];
            float entropyScalar = totalNlProbAfterwards;
            sample.EntropyScalar = tensor(entropyScalar).unsqueeze(0).DetachFromDisposeScope();
            totalNlProbAfterwards += sample.ChosenMoveNLProb;
        }
    }
}
