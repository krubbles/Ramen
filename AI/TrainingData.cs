namespace Ramen.AI;

using Ramen.Game;
using System.ComponentModel;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.Intrinsics.Arm;
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
                    EvaluationTrainingSample sample = agent.MakeMove(temp: 1f);
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

public interface ITensorGroup
{
}

public static class TensorGroupExtentions
{
    public static ITensorGroup Stack(IList<ITensorGroup> tensorGroups, bool disposeInputs, bool concat, int dim = 0)
    {
        ITensorGroup result = MakeNew(tensorGroups[0].GetType());
        FieldInfo[] fields = GetTensorFields(result.GetType());

        foreach (FieldInfo field in fields)
        {
            if (field.FieldType == typeof(Tensor))
            {
                Tensor[] tensors = new Tensor[tensorGroups.Count];
                for (int i = 0; i < tensorGroups.Count; ++i)
                    tensors[i] = field.GetValue(tensorGroups[i]) as Tensor;
                if (tensors[0] is not null)
                    field.SetValue(result, concat ? cat(tensors, dim) : stack(tensors, dim));
            }
            else if (typeof(ITensorGroup).IsAssignableFrom(field.FieldType))
            {
                ITensorGroup[] tensors = new ITensorGroup[tensorGroups.Count];
                for (int i = 0; i < tensorGroups.Count; ++i)
                    tensors[i] = field.GetValue(tensorGroups[i]) as ITensorGroup;
                field.SetValue(result, Stack(tensors, disposeInputs, concat, dim));
            }

            if (disposeInputs)
            {
                foreach (ITensorGroup tensorGroup in tensorGroups)
                    tensorGroup.Dispose();
            }
        }

        return result;
    }

    public static T Stack<T>(IList<T> tensorGroups, bool disposeInputs, bool concat, int dim = 0) where T : ITensorGroup
    {
        ITensorGroup[] genericGroups = new ITensorGroup[tensorGroups.Count];
        for (int i = 0; i < tensorGroups.Count; ++i)
            genericGroups[i] = tensorGroups[i];
        ITensorGroup result = Stack(genericGroups, disposeInputs, concat, dim);
        return (T)result;
    }

    public static ITensorGroup GetBatch(this ITensorGroup me, int start, int end)
    {
        ITensorGroup result = MakeNew(me.GetType());
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                field.SetValue(result, tensor[start..end]);
            else if (value is ITensorGroup group)
                field.SetValue(result, group.GetBatch(start, end));
        }
        return result;
    }

    public static T GetBatch<T>(this T me, int start, int end) where T : ITensorGroup => (T)GetBatch((ITensorGroup)me, start, end);

    public static ITensorGroup IndexSelect(this ITensorGroup me, int dim, Tensor indices)
    {
        ITensorGroup result = MakeNew(me.GetType());
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                field.SetValue(result, tensor.index_select(dim, indices));
            else if (value is ITensorGroup group)
                field.SetValue(result, group.IndexSelect(dim, indices));
        }
        return result;
    }

    public static T IndexSelect<T>(this T me, int dim, Tensor indices) where T : ITensorGroup => (T)IndexSelect((ITensorGroup)me, dim, indices);
    
    public static ITensorGroup Clone(this ITensorGroup me)
    {
        ITensorGroup result = MakeNew(me.GetType());
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                field.SetValue(result, tensor.clone());
            else if (value is ITensorGroup group)
                field.SetValue(result, group.Clone());
        }
        return result;
    }

    public static T Clone<T>(this T me) where T : ITensorGroup => (T)Clone((ITensorGroup)me);

    public static ITensorGroup DetachFromDisposeScope(this ITensorGroup me)
    {
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                tensor.DetachFromDisposeScope();
            else if (value is ITensorGroup group)
                group.DetachFromDisposeScope();
        }
        return me;
    }

    public static T DetachFromDisposeScope<T>(this T me) where T : ITensorGroup => (T)DetachFromDisposeScope((ITensorGroup)me);
    
    public static void Dispose(this ITensorGroup me)
    {
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                tensor.Dispose();
            if (value is ITensorGroup tensorGroup)
                tensorGroup.Dispose();
        }
    }

    static ITensorGroup MakeNew(Type type)
    {
        ConstructorInfo constructor = type.GetConstructor(Type.EmptyTypes);
        return (ITensorGroup)constructor.Invoke(null);
    }

    static FieldInfo[] GetTensorFields(Type type)
    {
        if (!_tensorFieldsByType.TryGetValue(type, out FieldInfo[] tensors))
        {
            List<FieldInfo> tList = new(5);
            foreach (FieldInfo field in type.GetFields())
            {
                if (field.FieldType == typeof(Tensor)|| (typeof(ITensorGroup)).IsAssignableFrom(field.FieldType))
                {
                    tList.Add(field);
                }
            }
            tensors = tList.ToArray();
            _tensorFieldsByType.Add(type, tensors);
        }
        return tensors;
    }

    static readonly Dictionary<Type, FieldInfo[]> _tensorFieldsByType = [];
}

public class MoveTensors : ITensorGroup
{
    public Tensor PlayedHand;
    public Tensor RemainingHand;
    public Tensor HandsAndDiscards;
    public Tensor Score;
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