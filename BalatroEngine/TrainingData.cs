namespace BalatroAI;

using System.Diagnostics;
using static TorchSharp.torch;

public static class TrainingData
{
    public static readonly List<EvaluationTrainingSample> EvaluationTrainingData = new();

    public const int PolicyOutputWidth = 9;

    static void GenerateGRPOTrainingDataGroup(GameEvalModel model, int groupSize = 8)
    {
        FastRandom random = FastRandom.SeededByClock();
        GameData gameData = new();
        using var scope = NewDisposeScope();
        List<EvaluationTrainingSample>[] groupGames = new List<EvaluationTrainingSample>[groupSize];
        float[] groupRewards = new float[groupSize];
        using (no_grad())
        {
            for (int group = 0; group < groupSize; ++group)
            {
                List<EvaluationTrainingSample> gameSamples = new();
                groupGames[group] = gameSamples;
                GameState gameState = new(gameData);
                RamenAgent agent = new(gameState, model);

                gameState.AdvanceToNextPlayerChoice();

                while (gameState.HandState.RemainingHands > 0)
                {
                    gameState.AdvanceToNextPlayerChoice();
                    if (!agent.MakeMoveStochastic(1f, out EvaluationTrainingSample sample, 12, true))
                        break;
                    gameSamples.Add(sample);
                }

                groupRewards[group] = agent.GetCurrentReward();
            }
        }
        Array.Sort(groupRewards, groupGames);
        for (int group = 0; group < groupSize; ++group)
        {
            float percentile = (group + 0.5f) / groupSize;
            float advantage = (float)NormalCDFInverse(percentile);

            foreach (EvaluationTrainingSample sample in groupGames[group])
                EvaluationTrainingData.Add(sample with { Advantage = tensor(advantage).DetachFromDisposeScope() });
        }
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

    public static void GroupSameMoveCountSamples()
    {
        // Group by the N dimension (dim 0 of Target)
        var groups = EvaluationTrainingData
            .GroupBy(s => s.ProbDist.shape[0])
            .ToList();

        EvaluationTrainingData.Clear();

        foreach (var group in groups)
        {
            var samples = group.ToList();

            // If multiple samples have the same N, merge them; otherwise, keep as is
            if (samples.Count > 1)
            {
                EvaluationTrainingData.Add(EvaluationTrainingSample.Stack(samples, true, false));
            }
            else
            {
                EvaluationTrainingData.Add(samples[0]);
            }
        }
    }

    static void GenerateEvalTrainingDataJob(GameEvalModel model, int samples)
    {
        while (EvaluationTrainingData.Count < samples)
        {
            GenerateGRPOTrainingDataGroup(model);
        }
    }

    public static void GenerateEvaluationTrainingData(GameEvalModel model, int samples)
    {
        Stopwatch watch = Stopwatch.StartNew();

        if (true)
        {

            Task[] tasks = new Task[1];
            for (int i = 0; i < tasks.Length; ++i)
            {
                tasks[i] = Task.Run(() =>
                {
                    GenerateEvalTrainingDataJob(model, samples);
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
            GenerateEvalTrainingDataJob(model, samples);
        }
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
                        GameStateTensors =  default,
                        ProbDist = target.clone().DetachFromDisposeScope(),
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
}


public struct GameStateTensors : IDisposable
{
    public Tensor Hand;
    public Tensor RemainingDeck;
    public Tensor FullDeck;
    public Tensor OtherState;

    public static GameStateTensors Stack(IReadOnlyList<GameStateTensors> tensors, bool disposeInputs, bool concat)
    {
        GameStateTensors result = new()
        {
            Hand = concat ? cat(tensors.Select(t => t.Hand).ToArray(), dim: 0) : stack(tensors.Select(t => t.Hand).ToArray()),
            // RemainingDeck = concat ? cat(tensors.Select(t => t.RemainingDeck).ToArray(), dim: 0) : stack(tensors.Select(t => t.RemainingDeck).ToArray()),
            // FullDeck = concat ? cat(tensors.Select(t => t.FullDeck).ToArray(), dim: 0) : stack(tensors.Select(t => t.FullDeck).ToArray()),
            OtherState = concat ? cat(tensors.Select(t => t.OtherState).ToArray(), dim: 0) : stack(tensors.Select(t => t.OtherState).ToArray()),
            RemainingDeck = concat ? cat(tensors.Select(t => t.RemainingDeck).ToArray(), dim: 0) : stack(tensors.Select(t => t.RemainingDeck).ToArray()),
        };

        if (disposeInputs)
        {
            foreach (var t in tensors)
                t.Dispose();
        }

        return result;
    }

    public GameStateTensors GetBatch(int start, int end)
    {
        return new GameStateTensors()
        {
            Hand = Hand?[start..end],
            FullDeck = FullDeck?[start..end],
            OtherState = OtherState?[start..end],
            RemainingDeck = RemainingDeck?[start..end],
        };
    }

    public GameStateTensors IndexSelect(Tensor indices)
    {
        return new GameStateTensors()
        {
            Hand = Hand?.index_select(0, indices),
            RemainingDeck = RemainingDeck?.index_select(0, indices),
            FullDeck = FullDeck?.index_select(0, indices),
            OtherState = OtherState?.index_select(0, indices),
        };
    }

    public GameStateTensors Clone()
    {
        return new()
        {
            Hand = Hand?.clone(),
            FullDeck = FullDeck?.clone(),
            OtherState = OtherState?.clone(),
            RemainingDeck = RemainingDeck?.clone(),
        };
    }

    public void Dispose()
    {
        Hand?.Dispose();
        OtherState?.Dispose();
        FullDeck?.Dispose();
        RemainingDeck?.Dispose();
    }

    public GameStateTensors DetachFromDisposeScope()
    {
        Hand?.DetachFromDisposeScope();
        OtherState?.DetachFromDisposeScope();
        FullDeck?.DetachFromDisposeScope();
        RemainingDeck?.DetachFromDisposeScope();
        return this;
    }
}

public struct EvaluationTrainingSample : IDisposable
{
    public GameStateTensors GameStateTensors;
    public Tensor ProbDist;
    public Tensor Advantage;

    public static EvaluationTrainingSample Stack(IReadOnlyList<EvaluationTrainingSample> samples, bool disposeInputs, bool concat)
    {
        EvaluationTrainingSample result = new()
        {
            GameStateTensors = GameStateTensors.Stack(samples.Select(s => s.GameStateTensors).ToList(), disposeInputs, concat),
            ProbDist = concat ? 
                cat(samples.Select(s => s.ProbDist).ToArray(), dim: 0) : 
                stack(samples.Select(s => s.ProbDist).ToArray()),
            Advantage = concat ?
                cat(samples.Select(s => s.Advantage).ToArray(), dim: 0) :
                stack(samples.Select(s => s.Advantage).ToArray()),
        };

        if (disposeInputs)
        {
            foreach (var s in samples) 
                s.ProbDist?.Dispose();
        }

        return result;
    }

    public EvaluationTrainingSample Shuffle()
    {
        long n = ProbDist.shape[0];
        using (Tensor indices = randperm(n, device: ProbDist.device))
        {
            return new EvaluationTrainingSample()
            {
                GameStateTensors = GameStateTensors.IndexSelect(indices),
                ProbDist = ProbDist.index_select(0, indices),
                Advantage = Advantage.index_select(0, indices),
            };
        }
    }

    public EvaluationTrainingSample GetBatch(int start, int end)
    {
        return new()
        {
            GameStateTensors = GameStateTensors.GetBatch(start, end),
            ProbDist = ProbDist[start..end],
            Advantage = Advantage[start..end]
        };
    }


    public void Dispose()
    {
        GameStateTensors.Dispose();
        ProbDist?.Dispose();
        Advantage?.Dispose();
    }
}