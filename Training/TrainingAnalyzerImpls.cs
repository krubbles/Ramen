namespace Ramen.Training;

using System;
using System.Collections.Generic;
using Ramen.AI;
using Ramen.Game;

public static class TrainingRunAnalyzerUtils
{
    public readonly struct MoveContext
    {
        public MoveContext(GameState gameState, Move move, AnnotatingDataMove annotation, int moveIndex)
        {
            GameState = gameState;
            Move = move;
            Annotation = annotation;
            MoveIndex = moveIndex;
        }

        public readonly GameState GameState;

        public readonly Move Move;

        public readonly AnnotatingDataMove Annotation;

        public readonly int MoveIndex;
    }


    public static void ForeachMove(this GameState gameState, Action<MoveContext> action, bool preState)
    {
        // Save move buffer.
        Move[] moveBuffer = gameState.MoveState.MoveHistory.ToArray();

        // Roll back to the beginning.
        gameState.MoveState.RevertToStep(0);

        try
        {
            // Replay moves and invoke action on each player move.
            int playerMoveIndex = 0;
            for (int moveIndex = 0; moveIndex < moveBuffer.Length; moveIndex++)
            {
                Move move = moveBuffer[moveIndex];
                if (move is AnnotatingDataMove)
                {
                    move.Apply(gameState);
                    continue;
                }

                AnnotatingDataMove annotation = null;

                if (moveIndex + 1 < moveBuffer.Length && moveBuffer[moveIndex + 1] is AnnotatingDataMove nextAnnotation)
                    annotation = nextAnnotation;

                if (annotation == null)
                {
                    move.Apply(gameState);
                    continue;
                }

                if (preState)
                    action(new MoveContext(gameState, move, annotation, playerMoveIndex));

                move.Apply(gameState);
                annotation.Apply(gameState);
                moveIndex++;

                if (!preState)
                    action(new MoveContext(gameState, move, annotation, playerMoveIndex));

                playerMoveIndex++;
            }
        }
        finally
        {
            // Restore if the replay did not complete.
            if (gameState.MoveState.MoveHistory.Count != moveBuffer.Length)
            {
                gameState.MoveState.RevertToStep(0);
                for (int moveIndex = 0; moveIndex < moveBuffer.Length; moveIndex++)
                    moveBuffer[moveIndex].Apply(gameState);
            }
        }
    }
}

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
            game.ForeachMove(context =>
            {
                AnnotatingDataMove annotation = context.Annotation;
                ushort[] encodedProbs = annotation.ToArray<ushort>();
                if (encodedProbs.Length == 0)
                    return;

                Move move = context.Move;

                float entropy = 0;
                for (int i = 0; i < encodedProbs.Length; i++)
                {
                    float prob = AnnotatingDataMove.DecodeProb(encodedProbs[i]);
                    float nlProb = -MathF.Log(MathF.Max(prob, 1e-9f));
                    entropy += prob * nlProb;
                }

                for (int filterIndex = 0; filterIndex < _filters.Length; filterIndex++)
                {
                    if (!_filters[filterIndex].filter(game, move))
                        continue;

                    totalEntropy[filterIndex] += entropy;
                    distributionCount[filterIndex] += 1;
                }
            }, preState: true);
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
