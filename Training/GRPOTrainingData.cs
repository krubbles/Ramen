namespace Ramen.Training;

using System;
using System.Collections.Generic;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public static class GRPOTrainingData
{
    public static List<PolicyTrainingSample> GenerateTrainingData(IPolicyModel model, int trainingSampleCount, int sampledSoftmaxCount, int groupSize = 32)
    {
        List<PolicyTrainingSample> outputSamples = [];
        
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        PolicyOnlyAgent agent = new(model);

        GameState[] groupStates = new GameState[groupSize];
        List<PolicyTrainingSample>[] activeSamples = new List<PolicyTrainingSample>[groupSize];

        Dictionary<PolicyTrainingSample, float> rewardBySample = new();
        float totalReward = 0f;
        float totalSquaredReward = 0f;
        int gamesCompleted = 0;

        // Seed the initial active slots. 
        // active slots are used to track which slots aren't in use after all games have been started but only some are completed. 
        for (int slot = 0; slot < groupSize; ++slot)
        {
            groupStates[slot] = CreateGameState();
            activeSamples[slot] = [];
        }

        while (outputSamples.Count < trainingSampleCount)
        {
            // Advance all active games by one move in a single batched policy forward pass.
            PolicyTrainingSample[] stepSamples = agent.MakeMoveAndTrainingSample(groupStates, sampledSoftmaxCount);

            for (int slot = 0; slot < groupSize; ++slot)
            {
                PolicyTrainingSample sample = stepSamples[slot];
                if (sample != null)
                    activeSamples[slot].Add(sample);

                if (!agent.IsGameDone(groupStates[slot]))
                    continue;

                // Game is done, calculate reward and fill out training data for all moves in the game.

                List<PolicyTrainingSample> gameSamples = activeSamples[slot];
                FillEntropyScalars(gameSamples);

                gamesCompleted++;
                float reward = GetReward(groupStates[slot]);
                totalReward += reward;
                totalSquaredReward += reward * reward;
                for (int sampleIndex = 0; sampleIndex < gameSamples.Count; ++sampleIndex)
                    rewardBySample[gameSamples[sampleIndex]] = reward;

                foreach (PolicyTrainingSample s in gameSamples)
                {
                    if (outputSamples.Count >= trainingSampleCount)
                        goto fillAdvantagesAndReturn;
                    outputSamples.Add(s);
                }

                groupStates[slot] = CreateGameState();
                activeSamples[slot] = [];
                continue;
            }
        }

        fillAdvantagesAndReturn:

        // Normalize rewards into advantages after all rollouts are complete.
        float meanReward = totalReward / gamesCompleted;
        float centeredSquares = totalSquaredReward - totalReward * meanReward;
        float stdDev = MathF.Sqrt(MathF.Max(0f, centeredSquares / gamesCompleted));

        foreach (KeyValuePair<PolicyTrainingSample, float> pair in rewardBySample)
        {
            PolicyTrainingSample sample = pair.Key;
            float advantage = (pair.Value - meanReward) / MathF.Max(stdDev, 1e-8f);
            sample.Advantage?.Dispose();
            sample.Advantage = tensor([advantage], device: CPU).DetachFromScope();
        }   

        return outputSamples;
    }


    static void FillEntropyScalars(List<PolicyTrainingSample> samples)
    {
        float totalNlProbAfterwards = 0f;
        for (int i = samples.Count - 1; i >= 0; i--)
        {
            PolicyTrainingSample sample = samples[i];
            float entropyScalar = totalNlProbAfterwards;
            sample.EntropyScalar?.Dispose();
            sample.EntropyScalar = tensor([entropyScalar], device: CPU).DetachFromScope();
            totalNlProbAfterwards += sample.ChosenMoveNLProb;
        }
    }


    static GameState CreateGameState()
    {
        return new(GameData.Default);
    }


    public static float GetReward(GameState gameState)
    {
        float roundsSurvived = gameState.Round / 3f;
        return roundsSurvived * roundsSurvived;
    }
}

public class PolicyTrainingSample : Ramen.AgentTools.ITensorGroup
{
    public GameStateTensors StateTensors;
    public UseHandTensors UseHandTensors;
    public Tensor MoveIndices;
    public Tensor Target;
    public Tensor SamplingProb;

    /// <summary>
    /// The negative natural log probability of the chosen move.
    /// </summary>
    public float ChosenMoveNLProb;
    public Tensor Advantage;
    public Tensor EntropyScalar;
}
