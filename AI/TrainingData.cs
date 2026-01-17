namespace Ramen.AI;

using Ramen.Game;
using System.Diagnostics;
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
    public float RewardStdDev => (TotalSquaredReward - TotalReward  * MeanReward) / Math.Max(1, GamesCount - 1);

    public float AverageNLProb(int depth) => TotalNLProbByDepth[depth] / Math.Max(1, CountByDepth[depth]);
    public float AverageAttribution(int depth) => TotalAttributionByDepth[depth] / Math.Max(1, CountByDepth[depth]);
}

public static class TrainingData
{
    public static readonly List<EvaluationTrainingSample> EvaluationTrainingData = new();

    public const int PolicyOutputWidth = 9;

    class SN
    {
        public EvaluationTrainingSample Sample;
        public int N;
        public Move Move;
        public float NLProb;
    }


    static void RunGroup(PolicyModel model, TrainingDataStats stats, int groupSize = 32)
    {
        GameState gameState = new(new());
        if (true)
        {
            GenerateGRPOTrainingDataGroup(model, gameState, stats, groupSize);
            return;
        }
    }

    static void GenerateGRPOTrainingDataGroup(PolicyModel model, GameState gameState, TrainingDataStats stats, int groupSize = 128)
    {
        RamenAgent agent = new(gameState, model);
        FastRandom random = FastRandom.SeededByClock();
        GameData gameData = new();
        using var scope = NewDisposeScope();
        List<SN>[] groupGames = new List<SN>[groupSize];
        float[] groupRewards = new float[groupSize];
        int baselineMoveCount = gameState.MoveState.MoveHistory.Count;
        using (no_grad())
        {
            for (int group = 0; group < groupSize; ++group)
            {
                gameState.Random.SetState((ulong)random.Next());
                
                List<SN> gameSamples = new();

                groupGames[group] = gameSamples;

                gameState.AdvanceToNextPlayerChoice();
                while (gameState.HandState.RemainingHands > 1 && gameState.ScoringState.CurrentRoundTotalChips < 300)
                {
                    gameState.AdvanceToNextPlayerChoice();
                    if (agent.GameIsDone())
                        break;
                    EvaluationTrainingSample sample = agent.MakeMoveAndTrainingSample(temp: 1f);
                    gameSamples.Add(new() { Sample = sample, N = 1, Move = gameState.MoveState.MoveHistory[^1], NLProb = sample.ChosenMoveNLProb });
                }
                if (gameState.ScoringState.CurrentRoundTotalChips < 300)
                {
                    gameState.AdvanceToNextPlayerChoice();
                    agent.MakeHighestScoringMove();
                }
                groupRewards[group] = agent.GetCurrentReward();
                while (gameState.MoveState.MoveHistory.Count > baselineMoveCount)
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

            List<SN> nodes = groupGames[group];
            for (int depth = 0; depth < nodes.Count; ++depth)
            {
                SN node = nodes[depth];
                stats.NodesCount++;
                node.Sample.Advantage = tensor(advantage).unsqueeze_(0).DetachFromDisposeScope();
                EvaluationTrainingData.Add(node.Sample);

                stats.TotalNLProbByDepth[depth] += node.NLProb;
                stats.CountByDepth[depth] += 1;
            }
        }
    }

    static void GenerateEvalTrainingDataJob(PolicyModel model, TrainingDataStats stats, int samples, float temp)
    {
        while (EvaluationTrainingData.Count < samples)
        {
            RunGroup(model, stats);
        }
    }

    public static TrainingDataStats GenerateEvaluationTrainingData(PolicyModel model, int samples, float temp)
    {
        Stopwatch watch = Stopwatch.StartNew();
        TrainingDataStats stats = new();
        if (true)
        {

            Task[] tasks = new Task[1];
            for (int i = 0; i < tasks.Length; ++i)
            {
                tasks[i] = Task.Run(() =>
                {
                    GenerateEvalTrainingDataJob(model, stats, samples, temp);
                });
            }

            while (EvaluationTrainingData.Count < samples)
            {
                Thread.Sleep(1000);
                Console.WriteLine($"Samples {EvaluationTrainingData.Count}, time {watch.Elapsed.TotalSeconds:F1}, rate {EvaluationTrainingData.Count / watch.Elapsed.TotalSeconds:F1}");
                foreach (Task task in tasks)
                {
                    if (task.Exception != null)
                        throw task.Exception;
                }
            }

            Task.WaitAll(tasks);
        }
        else
        {
            GenerateEvalTrainingDataJob(model, stats, samples, temp);
        }
        return stats;
    }
    
    public static void GenerateLastMoveTrainingData(PolicyModel model, GameDatabase database)
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

            Move lastMove = game.MoveState.MoveHistory[^1];
            RamenAgent agent = new RamenAgent(game, model);

            game.MoveState.MoveHistory[^1].Revert(game);

            if (agent.CreateTrainingSample(lastMove, 1.0f, out var sample, out _, 20))
            {
                EvaluationTrainingData.Add(sample);
            }
        }
        Console.WriteLine();
    }

    static ushort[] GetSampledMoveIndices(AnnotatingDataMove annotation) 
    {
        byte[] data = annotation.Data;
        ushort[] indices = new ushort[data.Length * 2];
        Buffer.BlockCopy(data, 0, indices, 0, data.Length);
        return indices;
    }

    static AnnotatingDataMove GetMoveIndicesAnnotation(ushort[] moveIndices)
    {
        byte[] data = new byte[moveIndices.Length / 2];
        Buffer.BlockCopy(moveIndices, 0, data, 0, data.Length);
        AnnotatingDataMove annotation = new(data);
        return annotation;
    }

    public static GameState PlayGameMonteCarlo(PolicyModel model, int branches, int samples, bool log, float temp)
    {
        GameState gameState = new(GameData.Default);
        RamenAgent agent = new(gameState, model);
        gameState.AdvanceToNextPlayerChoice();

        List<int> playerChoiceMoveSteps = new();

        while (!agent.GameIsDone())
        {
            gameState.AdvanceToNextPlayerChoice();
            playerChoiceMoveSteps.Add(gameState.MoveState.MoveStep);
            agent.MakeMove(1.0f);
        }

        int rollbackIndex = agent.Random.Next(playerChoiceMoveSteps.Count);
        int rollbackStep = playerChoiceMoveSteps[rollbackIndex];
        gameState.MoveState.RevertToStep(rollbackStep);

        var candidateMoves = agent.SampleMoves(1.0f, branches);
        Move bestMove = agent.SelectBestMoveMonteCarlo(candidateMoves, samples, temp);

        if (log)
        {
            Console.WriteLine($"GameState: {gameState}");
            // Note: We could optionally add logging of move evaluations here if needed
        }

        bestMove.Apply(gameState);
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

public class EvaluationTrainingSample : ITensorGroup
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