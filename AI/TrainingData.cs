namespace Ramen.AI;

using Ramen.Game;
using static TorchSharp.torch;

public class TrainingDataStats
{
    public float TotalReward;
    public float TotalSquaredReward;
    public int GamesCount;

    public const int MaxDepth = 10;
    public int NodesCount = 0;
    public int[] CountByDepth = new int[MaxDepth];
    public float[] TotalNLProbByDepth = new float[MaxDepth];
    public float[] TotalAttributionByDepth = new float[MaxDepth];

    public float MeanReward => TotalReward / GamesCount;
    public float RewardStdDev => (TotalSquaredReward - TotalReward * MeanReward) / Math.Max(1, GamesCount - 1);

    public float AverageNLProb(int depth) => TotalNLProbByDepth[depth] / Math.Max(1, CountByDepth[depth]);
    public float AverageAttribution(int depth) => TotalAttributionByDepth[depth] / Math.Max(1, CountByDepth[depth]);
}

public static class TrainingData
{
    public static readonly List<PolicyTrainingSample> PolicyData = new();

    public const int PolicyOutputWidth = 9;


    class SN
    {
        public PolicyTrainingSample Sample;
        public int N;
        public Move Move;
        public float NLProb;
    }


    static void RunGroup(PolicyModel model, TrainingDataStats stats, int groupSize = 512)
    {
        GameState gameState = new(new());
        GenerateGRPOTrainingDataGroup(model, gameState, stats, groupSize);
    }

    static void GenerateGRPOTrainingDataGroup(PolicyModel model, GameState gameState, TrainingDataStats stats, int groupSize = 128)
    {
        RamenAgent agent = new(gameState, model);
        FastRandom random = FastRandom.SeededByClock();

        using var scope = NewDisposeScope();
        List<SN>[] groupGames = new List<SN>[groupSize];
        float[] groupRewards = new float[groupSize];

        int startingMoveCount = gameState.MoveState.MoveHistory.Count;
        using (no_grad())
        {
            for (int group = 0; group < groupSize; ++group)
            {
                gameState.Reseed();

                List<SN> gameSamples = new();

                groupGames[group] = gameSamples;

                gameState.AdvanceToNextPlayerChoice();
                while (!agent.GameIsDone())
                {
                    gameState.AdvanceToNextPlayerChoice();
                    PolicyTrainingSample sample = agent.MakeMoveAndTrainingSample(temp: 1f);
                    gameSamples.Add(new() { Sample = sample, N = 1, Move = gameState.MoveState.MoveHistory[^1], NLProb = sample.ChosenMoveNLProb });
                }
                groupRewards[group] = agent.GetCurrentReward();
                while (gameState.MoveState.MoveHistory.Count > startingMoveCount)
                    gameState.MoveState.RevertLastMove();

            }
        }

        float sum = 0;
        float sqSum = 0;

        for (int group = 0; group < groupSize; ++group)
        {
            sum += groupRewards[group];
            sqSum += groupRewards[group] * groupRewards[group];
        }

        stats.TotalReward += sum;
        stats.TotalSquaredReward += sqSum;
        stats.GamesCount += groupSize;

        float mean = sum / groupSize;
        float ss = sqSum - sum * mean;
        float stdDev = MathF.Sqrt(ss / (groupSize - 1));

        Array.Sort(groupRewards, groupGames);
        for (int group = 0; group < groupSize; ++group)
        {
            float percentile = (group + 0.5f) / groupSize;
            float advantage = (groupRewards[group] - mean) / MathF.Max(stdDev, 1e-8f);
            if (float.IsNaN(advantage) || float.IsInfinity(advantage))
                advantage = 0;
            List<SN> nodes = groupGames[group];
            for (int depth = 0; depth < nodes.Count; ++depth)
            {
                SN node = nodes[depth];
                stats.NodesCount++;
                node.Sample.Advantage = tensor(advantage).unsqueeze_(0).DetachFromDisposeScope();
                PolicyData.Add(node.Sample);

                stats.TotalNLProbByDepth[depth] += node.NLProb;
                stats.CountByDepth[depth] += 1;
            }
        }
    }
    public static TrainingDataStats GenerateGRPO(PolicyModel model, int samples)
    {
        Console.WriteLine();
        TrainingDataStats stats = new();
        while (stats.GamesCount < samples)
        {
            RunGroup(model, stats);
            Console.Write($"\rGenerated {stats.GamesCount}/{samples} games...");
        }
        return stats;
    }

    public static void GenerateTrainingDataFromGames(PolicyModel model, GameDatabase database)
    {
        int totalGames = 0;
        foreach (GameState game in database)
        {
            if (game.MoveState.MoveHistory.Count > 0)
                totalGames++;
        }

        int processed = 0;
        foreach (GameState game in database)
        {
            if (game.MoveState.MoveHistory.Count == 0)
                continue;

            processed++;
            int percent = (int)((float)processed / totalGames * 100);
            Console.Write($"\rProcessing: {percent}% ({processed}/{totalGames})");

            RamenAgent agent = new(game, model);

            for (int moveIndex = game.MoveState.MoveHistory.Count - 1; moveIndex >= 0; moveIndex--)
            {
                Move move = game.MoveState.MoveHistory[moveIndex];
                if (move is AnnotatingDataMove annotation)
                {
                    MoveSampleAnnotationData[] branchData = GetBranchMoveIndices(annotation);

                    move.Revert(game);
                    game.MoveState.RevertLastMove(); // we want to create the sample in the context of the state before the move was applied.
                    moveIndex--;
                    PolicyTrainingSample sample = agent.CreateMonteCarloTrainingSample(branchData);
                    PolicyData.Add(sample);
                }
            }
        }
        Console.WriteLine();
    }

    static unsafe MoveSampleAnnotationData[] GetBranchMoveIndices(AnnotatingDataMove annotation)
    {
        byte[] bytes = annotation.Data;
        MoveSampleAnnotationData[] data = new MoveSampleAnnotationData[bytes.Length / sizeof(MoveSampleAnnotationData)];
        fixed (byte* src = bytes)
        fixed (MoveSampleAnnotationData* dst = data)
        {
            Buffer.MemoryCopy(src, dst, bytes.Length, bytes.Length);
        }
        return data;
    }

    static unsafe AnnotatingDataMove GetMoveIndicesAnnotation(MoveSampleAnnotationData[] data)
    {
        byte[] bytes = new byte[data.Length * sizeof(MoveSampleAnnotationData)];
        fixed (MoveSampleAnnotationData* src = data)
        fixed (byte* dst = bytes)
        {
            Buffer.MemoryCopy(src, dst, bytes.Length, bytes.Length);
        }
        AnnotatingDataMove annotation = new(bytes);
        return annotation;
    }


    public static GameState PlayGame(PolicyModel model, float temp, CancellationToken cancel = default)
    {
        GameState gameState = new(GameData.Default);
        RamenAgent agent = new(gameState, model);
        while (!agent.GameIsDone())
        {
            cancel.ThrowIfCancellationRequested();
            gameState.AdvanceToNextPlayerChoice();
            agent.MakeMove(temp);
        }
        return gameState;
    }

    public static GameState PlayGameMonteCarlo(PolicyModel model, int branches, int continuations, bool log, float temp, CancellationToken cancel = default)
    {
        GameState gameState = new(GameData.Default);
        RamenAgent agent = new(gameState, model);
        List<int> moveSteps = new();
        while (!agent.GameIsDone())
        {
            cancel.ThrowIfCancellationRequested();
            gameState.AdvanceToNextPlayerChoice();
            moveSteps.Add(gameState.MoveState.MoveStep);
            agent.MakeMove(1f);
        }

        cancel.ThrowIfCancellationRequested();

        gameState.MoveState.RevertToStep(FastRandom.SeededByClock().NextPick(moveSteps));

        if (log)
            Console.WriteLine("\n" + gameState);
        MoveSampleAnnotationData[] branchData = agent.MakeMoveMonteCarlo(temp, branches, continuations);
        if (log)
            Console.WriteLine(gameState.MoveState.MoveHistory[^1]);
        AnnotatingDataMove annotation = GetMoveIndicesAnnotation(branchData);
        annotation.Apply(gameState);

        return gameState;
    }
}


public class GameStateTensors : ITensorGroup
{
    public Tensor FullHand;
    public Tensor RemainingDeck;
    public Tensor Score;
    public Tensor HandsAndDiscards;
}

public class PolicyTrainingSample : ITensorGroup
{
    public GameStateTensors State;
    public MoveTensors Moves;
    public Tensor MoveProbDist;
    public Tensor Advantage;

    /// <summary>
    /// The negative natural log probability of the chosen move.
    /// </summary>
    public float ChosenMoveNLProb;
}


public struct MoveSampleAnnotationData
{
    public ushort MoveIndex;
    public ushort NLProbTimes1K; // for log-q adjustment
}
