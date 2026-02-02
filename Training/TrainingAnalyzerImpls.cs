namespace Ramen.Training;

using System;
using System.Collections.Generic;
using Ramen.AI;
using Ramen.Game;

/// <summary>
/// Calculates mean reward and reward standard deviation.
/// </summary>
public sealed class RewardStatsTrainingRunAnalyzer : ITrainingRunAnalyzer
{
    public void Analyze(PolicyModel model, IEnumerable<GameState> games, CSVBuilder output)
    {
        double sum = 0;
        double sqSum = 0;
        int count = 0;

        foreach (GameState game in games)
        {
            RamenAgent agent = new(game, model);
            float reward = agent.GetCurrentReward();
            sum += reward;
            sqSum += reward * reward;
            count++;
        }

        if (count == 0)
        {
            output.SetCell("reward_mean", 0f);
            output.SetCell("reward_stddev", 0f);
            return;
        }

        double mean = sum / count;
        double variance = 0;
        if (count > 1)
            variance = (sqSum - sum * mean) / (count - 1);
        if (variance < 0)
            variance = 0;

        double stdDev = Math.Sqrt(variance);

        output.SetCell("reward_mean", mean);
        output.SetCell("reward_stddev", stdDev);
    }
}
