namespace Ramen.Agents;

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
    public void Analyze(IEnumerable<GameState> games, CSVBuilder output)
    {
        float sum = 0f;
        float sqSum = 0f;
        int count = 0;

        foreach (GameState game in games)
        {
            float reward = GetReward(game);
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

        float mean = sum / count;
        float variance = 0f;
        if (count > 1)
            variance = (sqSum - sum * mean) / (count - 1);
        if (variance < 0f)
            variance = 0f;

        float stdDev = MathF.Sqrt(variance);

        output.SetCell("reward_mean", mean);
        output.SetCell("reward_stddev", stdDev);
    }

    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalChips >= 300)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalChips / 1000f;
    }
}

/// <summary>
/// Calculates the models average policy entropy. 
/// Only works for agents that add policy annotations using <see cref="AnnotationDataUtils"/> 
/// Supports multiple columns, each of which has a filter that determines which GameStates to average over.
/// </summary>
public sealed class PolicyEntropyTrainingRunAnalyzer : ITrainingRunAnalyzer
{
    readonly (string colName, Func<GameState, Move, bool> filter)[] _filters;

    /// <summary>
    /// Initializes a new instance of the <see cref="PolicyEntropyTrainingRunAnalyzer"/> class with a default filter that includes all GameStates.
    /// This means it calculates the mean policy entropy across all moves.
    /// </summary>
    public PolicyEntropyTrainingRunAnalyzer() : this(("policy_entropy_mean", (_, _) => true))
    {
    }

    public PolicyEntropyTrainingRunAnalyzer(params (string colName, Func<GameState, Move, bool> filter)[] filters)
    {
        _filters = filters ?? [];
    }

    public void Analyze(IEnumerable<GameState> games, CSVBuilder output)
    {
        if (_filters.Length == 0)
            return;

        float[] totalEntropy = new float[_filters.Length];
        int[] distributionCount = new int[_filters.Length];

        // Gather entropy from each annotated policy distribution.
        foreach (GameState game in games)
        {
            game.ForeachMove(context =>
            {
                AnnotatingDataMove annotation = context.Annotation;
                if (!AnnotationDataUtils.TryDecodePolicyAnnotation(annotation, out float[] policy) || policy.Length == 0)
                    return;

                Move move = context.Move;

                float entropy = 0f;
                for (int i = 0; i < policy.Length; i++)
                {
                    float prob = policy[i];
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

            float meanEntropy = totalEntropy[filterIndex] / distributionCount[filterIndex];
            output.SetCell(colName, meanEntropy);
        }
    }
}

/// <summary>
/// Calculates the fraction of games that played a flush, full house, or straight.
/// </summary>
public sealed class HandTypePresenceTrainingRunAnalyzer : ITrainingRunAnalyzer
{
    public void Analyze(IEnumerable<GameState> games, CSVBuilder output)
    {
        // Initialize counts.
        int gameCount = 0;
        int flushCount = 0;
        int fullHouseCount = 0;
        int straightCount = 0;

        // Scan each game for played hand types.
        foreach (GameState game in games)
        {
            gameCount++;
            bool sawFlush = false;
            bool sawFullHouse = false;
            bool sawStraight = false;

            game.ForeachMove(context =>
            {
                Move move = context.Move;
                if (move is not UseHandMove useHandMove || useHandMove.IsDiscard)
                    return;

                HandPatterns patterns = context.GameState.HandState.ActiveHandPatterns;
                if (patterns.ContainsFlush)
                    sawFlush = true;
                if (patterns.HandType == HandType.FullHouse || patterns.HandType == HandType.FlushHouse)
                    sawFullHouse = true;
                if (patterns.ContainsStraight)
                    sawStraight = true;
            }, preState: false);

            if (sawFlush)
                flushCount++;
            if (sawFullHouse)
                fullHouseCount++;
            if (sawStraight)
                straightCount++;
        }

        // Write results.
        float totalGames = Math.Max(gameCount, 1);
        output.SetCell("played_flush_frac", flushCount / totalGames);
        output.SetCell("played_full_house_frac", fullHouseCount / totalGames);
        output.SetCell("played_straight_frac", straightCount / totalGames);
    }
}

/// <summary>
/// Calculates the fraction of games that lose, or win with specific remaining hands.
/// </summary>
public sealed class EndStateHandCountTrainingRunAnalyzer : ITrainingRunAnalyzer
{
    public void Analyze(IEnumerable<GameState> games, CSVBuilder output)
    {
        // Initialize counts.
        int gameCount = 0;
        int loseCount = 0;
        int winHands0 = 0;
        int winHands1 = 0;
        int winHands2 = 0;
        int winHands3 = 0;

        // Classify each game's end state.
        foreach (GameState game in games)
        {
            gameCount++;

            if (game.ScoringState.CurrentRoundTotalChips < 300)
            {
                loseCount++;
                continue;
            }

            int remainingHands = game.HandState.RemainingHands;
            if (remainingHands == 0)
                winHands0++;
            else if (remainingHands == 1)
                winHands1++;
            else if (remainingHands == 2)
                winHands2++;
            else if (remainingHands == 3)
                winHands3++;
        }

        // Write results.
        float totalGames = Math.Max(gameCount, 1);
        output.SetCell("loss_frac", loseCount / totalGames);
        output.SetCell("win_hands_0_frac", winHands0 / totalGames);
        output.SetCell("win_hands_1_frac", winHands1 / totalGames);
        output.SetCell("win_hands_2_frac", winHands2 / totalGames);
        output.SetCell("win_hands_3_frac", winHands3 / totalGames);
    }
}
