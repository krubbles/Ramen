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
        if (games <= 0 || groupSize <= 0)
            return stats;

        PolicyOnlyAgent agent = new(model);
        int batchSize = Math.Min(groupSize, games);

        GameState[] activeStates = new GameState[batchSize];
        List<PolicyTrainingSample>[] activeSamples = new List<PolicyTrainingSample>[batchSize];
        bool[] slotIsActive = new bool[batchSize];

        int gamesStarted = 0;
        int gamesFinished = 0;
        Dictionary<PolicyTrainingSample, float> rewardBySample = new();
        float totalReward = 0f;
        float totalSquaredReward = 0f;

        // Seed the initial active slots. 
        // active slots are used to track which slots aren't in use after all games have been started but only some are completed. 
        for (int slot = 0; slot < batchSize; ++slot)
        {
            activeStates[slot] = CreateGameState();
            activeSamples[slot] = new();
            slotIsActive[slot] = true;
            gamesStarted++;
        }

        using var scope = NewDisposeScope();
        using (no_grad())
        {
            while (gamesFinished < games)
            {
                // Build active slot map for this step.
                List<int> activeSlotIndices = new(batchSize);
                for (int slot = 0; slot < batchSize; ++slot)
                {
                    if (slotIsActive[slot])
                        activeSlotIndices.Add(slot);
                }

                GameState[] stepStates = new GameState[activeSlotIndices.Count];
                for (int i = 0; i < activeSlotIndices.Count; ++i)
                    stepStates[i] = activeStates[activeSlotIndices[i]];

                // Advance all active games by one move in a single batched policy forward pass.
                PolicyTrainingSample[] stepSamples = agent.MakeMoveTrainingSample(stepStates, sampleCount);

                // Finalize completed games and refill slots immediately for maximum parallelism.
                for (int i = 0; i < activeSlotIndices.Count; ++i)
                {
                    int slot = activeSlotIndices[i];
                    PolicyTrainingSample sample = stepSamples[i];
                    if (sample != null)
                        activeSamples[slot].Add(sample);

                    if (!agent.IsGameDone(activeStates[slot]))
                        continue;

                    List<PolicyTrainingSample> samples = activeSamples[slot];
                    FillEntropyScalars(samples);

                    float reward = GetReward(activeStates[slot]);
                    totalReward += reward;
                    totalSquaredReward += reward * reward;
                    for (int sampleIndex = 0; sampleIndex < samples.Count; ++sampleIndex)
                        rewardBySample[samples[sampleIndex]] = reward;

                    lock (TrainingData.PolicyData)
                    {
                        TrainingData.PolicyData.AddRange(samples);
                    }

                    gamesFinished++;
                    stats.GamesCount = gamesFinished;

                    if (gamesStarted < games)
                    {
                        activeStates[slot] = CreateGameState();
                        activeSamples[slot] = new();
                        slotIsActive[slot] = true;
                        gamesStarted++;
                        continue;
                    }

                    slotIsActive[slot] = false;
                }
            }
        }

        // Normalize rewards into advantages after all rollouts are complete.
        float meanReward = totalReward / Math.Max(1, gamesFinished);
        float centeredSquares = totalSquaredReward - totalReward * meanReward;
        float stdDev = MathF.Sqrt(MathF.Max(0f, centeredSquares / Math.Max(1, gamesFinished - 1)));

        foreach (KeyValuePair<PolicyTrainingSample, float> pair in rewardBySample)
        {
            PolicyTrainingSample sample = pair.Key;
            float advantage = (pair.Value - meanReward) / MathF.Max(stdDev, 1e-8f);
            sample.Advantage?.Dispose();
            sample.Advantage = tensor(advantage).unsqueeze(0).DetachFromDisposeScope();
        }

        stats.TotalReward = totalReward;
        stats.TotalSquaredReward = totalSquaredReward;

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


    static GameState CreateGameState()
    {
        return new(GameData.Default);
    }


    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalChips >= 300)
            return 1f + gameState.HandState.RemainingHands * 0.2f;
        return (float)gameState.ScoringState.CurrentRoundTotalChips / 1000f;
    }
}
