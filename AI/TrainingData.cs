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
    public int[] CountByTier = new int[GameEvalModel.Tiers];
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


    static void RunGroup(GameEvalModel model, TrainingDataStats stats, int groupSize = 32)
    {
        GameState gameState = new(new());
        RamenAgent agent = new(gameState, model);
        FastRandom random = FastRandom.SeededByClock();
        if (true)
        {
            GenerateGRPOTrainingDataGroup(model, gameState, stats, groupSize);
            return;
        }
        List<float> nlProbs = new();
        List<Move> moves = new();
        while (true)
        {
            gameState.AdvanceToNextPlayerChoice();
            if (agent.GameIsDone())
                break;
            agent.MakeMove(1.2f, out _, out float nlProb, 1, false);
            nlProbs.Add(nlProb);
            moves.Add(gameState.MoveState.MoveHistory[^1]);
        }
        float totalNLProb = 0;
        float[] cumNLProbs = new float[nlProbs.Count];
        for (int i = 0; i < nlProbs.Count; i++)
        {
            totalNLProb += nlProbs[i];
            cumNLProbs[i] = totalNLProb;
        }
        float rand = (random.NextPortion() * (1 - 1e-6f)) * totalNLProb;
        for (int i = 0; i < nlProbs.Count; i++)
        {
            if (rand <= cumNLProbs[i])
            {
                moves[i].Revert(gameState);
                GenerateGRPOTrainingDataGroup(model, gameState, stats, groupSize);
                return;
            }    
        }
    }

    static void GenerateGRPOTrainingDataGroup(GameEvalModel model, GameState gameState, TrainingDataStats stats, int groupSize = 128)
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
                    if (!agent.MakeMove(1, out EvaluationTrainingSample sample, out float nlProb, 12, true))
                        break;
                    gameSamples.Add(new() { Sample = sample, N = 1, Move = gameState.MoveState.MoveHistory[^1], NLProb = nlProb });
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
#if true
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
                int tier = (int)(node.Sample.ForcastTier?.item<long>() ?? 0);
                float advantageRemapped = RemapA(advantage, tier switch
                {
                    2 => -1,
                    1 => -0.5f,
                    0 => 0f,
                    3 => 0.5f,
                    4 => 1f,
                });
                node.Sample.Advantage = tensor(advantage).unsqueeze_(0).DetachFromDisposeScope();
                EvaluationTrainingData.Add(node.Sample);

                stats.CountByTier[tier]++;
                stats.TotalNLProbByDepth[depth] += node.NLProb;
                stats.CountByDepth[depth] += 1;
            }

        }


        float RemapA(float a, float x)
        {
            return a * MathF.Exp(x) - (x - 1) * MathF.Exp(x) - 1;
        }
#else // Renormalizing advantage
        Array.Sort(groupRewards, groupGames);
        for (int group = 0; group < groupSize; ++group)
        {
            float percentile = (group + 0.5f) / groupSize;
            float advantage = (float)NormalCDFInverse(percentile);

            foreach (EvaluationTrainingSample sample in groupGames[group])
                EvaluationTrainingData.Add(sample with { Advantage = tensor(advantage).DetachFromDisposeScope() });
        }
#endif
    }

    static double RationalApproximation(double t)
    {
        // Abramowitz and Stegun formula 26.2.23.
        // The absolute value of the error should be less than 4.5 e-4.
        double[] c = { 2.515517, 0.802853, 0.010328 };
        double[] d = { 1.432788, 0.189269, 0.001308 };
        return t - ((c[2] * t + c[1]) * t + c[0]) /
                    (((d[2] * t + d[1]) * t + d[0]) * t + 1.0);
    }

    static double NormalCDFInverse(double p)
    {
        if (p <= 0.0 || p >= 1.0)
        {
            string msg = String.Format("Invalid input argument: {0}.", p);
            throw new ArgumentOutOfRangeException(msg);
        }

        // See article above for explanation of this section.
        if (p < 0.5)
        {
            // F^-1(p) = - G^-1(p)
            return -RationalApproximation(Math.Sqrt(-2.0 * Math.Log(p)));
        }
        else
        {
            // F^-1(p) = G^-1(1-p)
            return RationalApproximation(Math.Sqrt(-2.0 * Math.Log(1.0 - p)));
        }
    }


    static void GenerateEvalTrainingDataJob(GameEvalModel model, TrainingDataStats stats, int samples, float temp)
    {
        while (EvaluationTrainingData.Count < samples)
        {
            RunGroup(model, stats);
        }
    }

    public static TrainingDataStats GenerateEvaluationTrainingData(GameEvalModel model, int samples, float temp)
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

    public static void GenerateEvalTrainingDataOneShotBestHand(int samples)
    {
        using var scope = NewDisposeScope();
        int lastLogCount = 0;
        int startingSampleCount = EvaluationTrainingData.Count;
        FastRandom random = FastRandom.SeededByClock();
        GameData gameData = new();
        using (no_grad())
        {
            while (EvaluationTrainingData.Count < startingSampleCount + samples)
            {
                gameData.Seed = random.Next();
                GameState gameState = new(gameData);
                gameState.StartRound();

                float finalReward = Testing.GetMaxOneShotScore(gameState) / 300f;
                Tensor target = tensor(finalReward).unsqueeze(0).unsqueeze(0);

                lock (EvaluationTrainingData)
                {
                    EvaluationTrainingData.Add(new()
                    {
                        State =  default,
                        MoveProbDist = target.clone().DetachFromDisposeScope(),
                    });
                    if (EvaluationTrainingData.Count - startingSampleCount >= lastLogCount + 1000)
                    {
                        lastLogCount += 1000;
                        Console.WriteLine($"Generated {EvaluationTrainingData.Count - startingSampleCount} / {samples} evaluation training samples");
                    }
                }
            }
        }
    }

    public static void GenerateLastMoveTrainingData(GameEvalModel model, GameDatabase database)
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
    public Tensor ForcastProbDist;
    public Tensor MoveProbDist;
    public Tensor Advantage;
    public Tensor ForcastTier;
}