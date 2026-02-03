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
/// Calculates the models average policy entropy. Supports multiple columns, each of which has a filter that determines which GameStates to average over.
/// 
/// </summary>
public sealed class PolicyEntropyTrainingRunAnalyzer : ITrainingRunAnalyzer
{
    readonly (string colName, Func<GameState, Move, bool> filter)[] _filters;

    /// <summary>
    /// Initializes a new instance of the <see cref="PolicyEntropyTrainingRunAnalyzer"/> class with a default filter that includes all GameStates.
    /// This means it calculates the mean policy entropy across all moves.
    /// </summary>
    public PolicyEntropyTrainingRunAnalyzer() : this(( "policy_entropy_mean", (_, _) => true ))
    {
    }

    public PolicyEntropyTrainingRunAnalyzer(params (string colName, Func<GameState, Move, bool> filter)[] filters)
    {
        _filters = filters ?? [];
    }

    public void Analyze(PolicyModel model, IEnumerable<GameState> games, CSVBuilder output)
    {
        if (_filters.Length == 0)
            return;

        float[] totalEntropy = new float[_filters.Length];
        int[] distributionCount = new int[_filters.Length];

        // gather entropy from each annotated policy distribution.
        foreach (GameState game in games)
        {
            // iterate through move history in reverse
            List<Move> moveHistory = game.MoveState.MoveHistory;
            for (int moveIndex = moveHistory.Count - 1; moveIndex >= 0; moveIndex--)
            {
                Move move = moveHistory[moveIndex];
                if (move is not AnnotatingDataMove annotation)
                    continue;

                ushort[] encodedProbs = annotation.ToArray<ushort>();
                if (encodedProbs.Length == 0)
                    continue;

                // revert annotation, then revert the move to get the state before the move was applied.
                annotation.Revert(game); // removes this move and all subsequent moves from the history
                moveIndex--;
                if (game.MoveState.MoveHistory.Count == 0)
                    break;
                game.MoveState.MoveHistory[moveIndex].Revert(game);

                // calculate the entropy for this move's policy distribution 
                float entropy = 0;
                for (int i = 0; i < encodedProbs.Length; i++)
                {
                    float prob = AnnotatingDataMove.DecodeProb(encodedProbs[i]);
                    float nlProb = -MathF.Log(MathF.Max(prob, 1e-9f));
                    entropy += prob * nlProb;
                }

                // add the entropy to the appropriate filtered entropy columns
                for (int filterIndex = 0; filterIndex < _filters.Length; filterIndex++)
                {
                    if (!_filters[filterIndex].filter(game, move))
                        continue;

                    totalEntropy[filterIndex] += entropy;
                    distributionCount[filterIndex] += 1;
                }
            }
        }

        for (int filterIndex = 0; filterIndex < _filters.Length; filterIndex++)
        {
            string colName = _filters[filterIndex].colName;
            if (distributionCount[filterIndex] == 0)
            {
                output.SetCell(colName, 0f);
                continue;
            }

            double meanEntropy = totalEntropy[filterIndex] / distributionCount[filterIndex];
            output.SetCell(colName, meanEntropy);
        }
    }
}
