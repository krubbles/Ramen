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
                    if (!agent.MakeMoveStochastic(1f, out EvaluationTrainingSample sample, true))
                        break;
                    gameSamples.Add(sample);
                }

                groupRewards[group] = agent.GetCurrentReward();
            }
        }
        Array.Sort(groupRewards, groupGames);
        for (int group = groupSize / 2; group < groupSize; ++group)
        {
            foreach (EvaluationTrainingSample sample in groupGames[group])
                EvaluationTrainingData.Add(sample);
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
                Console.WriteLine($"Samples {EvaluationTrainingData.Count * 100}, time {watch.Elapsed.TotalSeconds:F1}, rate {EvaluationTrainingData.Count * 100f / watch.Elapsed.TotalSeconds:F1}");
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
                        Target = target.clone().DetachFromDisposeScope(),
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

    public static GameStateTensors Stack(IReadOnlyList<GameStateTensors> tensors, bool disposeInputs)
    {
        GameStateTensors result = new()
        {
            Hand = cat(tensors.Select(t => t.Hand).ToArray(), dim: 0),
            RemainingDeck = cat(tensors.Select(t => t.RemainingDeck).ToArray(), dim: 0),
            FullDeck = cat(tensors.Select(t => t.FullDeck).ToArray(), dim: 0),
            OtherState = cat(tensors.Select(t => t.OtherState).ToArray(), dim: 0),
        };

        if (disposeInputs)
        {
            foreach (var t in tensors) t.Dispose();
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
    public Tensor Target;

    public static EvaluationTrainingSample Stack(IReadOnlyList<EvaluationTrainingSample> samples, bool disposeInputs)
    {
        EvaluationTrainingSample result = new()
        {
            GameStateTensors = GameStateTensors.Stack(samples.Select(s => s.GameStateTensors).ToList(), disposeInputs),
            Target = cat(samples.Select(s => s.Target).ToArray(), dim: 0),
        };

        if (disposeInputs)
        {
            foreach (var s in samples) 
                s.Target?.Dispose();
        }

        return result;
    }

    public EvaluationTrainingSample Shuffle()
    {
        long n = Target.shape[0];
        using (Tensor indices = randperm(n, device: Target.device))
        {
            return new EvaluationTrainingSample()
            {
                GameStateTensors = GameStateTensors.IndexSelect(indices),
                Target = Target.index_select(0, indices)
            };
        }
    }

    public void Dispose()
    {
        GameStateTensors.Dispose();
        Target?.Dispose();
    }
}