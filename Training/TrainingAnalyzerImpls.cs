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

/// <summary>
/// Calculates the average policy entropy across all annotated moves.
/// </summary>
public sealed class PolicyEntropyTrainingRunAnalyzer : ITrainingRunAnalyzer
{
    public void Analyze(PolicyModel model, IEnumerable<GameState> games, CSVBuilder output)
    {
        double totalEntropy = 0;
        int distributionCount = 0;

        // Gather entropy from each annotated policy distribution.
        foreach (GameState game in games)
        {
            List<Move> moveHistory = game.MoveState.MoveHistory;
            for (int moveIndex = 0; moveIndex < moveHistory.Count; moveIndex++)
            {
                Move move = moveHistory[moveIndex];
                if (move is not AnnotatingDataMove annotation)
                    continue;

                ushort[] encodedProbs = annotation.ToArray<ushort>();
                if (encodedProbs.Length == 0)
                    continue;

                double entropy = 0;
                for (int i = 0; i < encodedProbs.Length; i++)
                {
                    float nlProb = encodedProbs[i] / 3000f;
                    float prob = MathF.Exp(-nlProb);
                    entropy += prob * nlProb;
                }

                totalEntropy += entropy;
                distributionCount++;
            }
        }

        // Emit the mean entropy across all distributions (or 0 if none were found).
        if (distributionCount == 0)
        {
            output.SetCell("policy_entropy_mean", 0f);
            return;
        }

        double meanEntropy = totalEntropy / distributionCount;
        output.SetCell("policy_entropy_mean", meanEntropy);
    }
}
