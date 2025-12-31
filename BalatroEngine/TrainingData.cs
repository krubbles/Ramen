namespace BalatroAI;

using System.Diagnostics;
using static TorchSharp.torch;

public static class TrainingData
{
    public static readonly List<EvaluationTrainingSample> EvaluationTrainingData = new();
    public static readonly List<PolicyTrainingSample> PolicyTrainingData = new();

    public const int PolicyOutputWidth = 9;

    public static void GenerateEvaluationTrainingData(GameEvalModel model, int samples)
    {
        Stopwatch watch = Stopwatch.StartNew();

        Task[] tasks = new Task[8];
        for (int i = 0; i < tasks.Length; ++i)
        {
            tasks[i] = Task.Run(() =>
            {
                using var scope = NewDisposeScope();
                using var nograd = no_grad();
                FastRandom random = FastRandom.SeededByClock();
                GameData gameData = new();
                while (EvaluationTrainingData.Count < samples)
                {
                    GameState gameState = new(gameData);
                    gameState.AdvanceToNextPlayerChoice();
                    RamenAgent agent = new(gameState, model);

                    List<GameStateTensors> states = new();

                    while (gameState.HandState.RemainingHands > 0)
                    {
                        gameState.AdvanceToNextPlayerChoice();
                        List<Move> moves = gameState.GetMoveOptions();
                        if (moves.Count == 0)
                            break;
                        agent.MakeMoveStochastic(1f);
                        states.Add(agent.TensorsCloned);
                    }

                    float reward = (float)gameState.ScoringState.CurrentRoundTotalChips / 100f;
                    foreach (GameStateTensors state in states)
                    {
                        lock (EvaluationTrainingData)
                        {
                            EvaluationTrainingData.Add(new()
                            {
                                GameStateTensors = state,
                                Target = tensor(reward).unsqueeze(0).unsqueeze(0).DetachFromDisposeScope()
                            });
                        }
                    }
                }
            });
        }

        while (EvaluationTrainingData.Count < samples)
        {
            Thread.Sleep(1000);
            Console.WriteLine($"Samples {EvaluationTrainingData.Count}, time {watch.Elapsed.TotalSeconds:F1}, rate {EvaluationTrainingData.Count / watch.Elapsed.TotalSeconds:F1}");
        }

        Task.WaitAll(tasks);
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
        Tensor[] handStates = new Tensor[tensors.Count];
        Tensor[] otherStates = new Tensor[tensors.Count];
        for (int i = 0; i < tensors.Count; ++i)
        {
            handStates[i] = tensors[i].Hand;
            otherStates[i] = tensors[i].OtherState;
        }

        GameStateTensors result = new()
        {
            Hand = concat(handStates, dim: 0),
            OtherState = concat(otherStates, dim: 0),
        };

        if (disposeInputs)
        {
            for (int i = 0; i < tensors.Count; ++i)
            {
                handStates[i].Dispose();
                otherStates[i].Dispose();
            }
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

public struct PolicyTrainingSample : IDisposable
{
    public GameStateTensors GameStateTensors;
    public Tensor InUseMask;
    public Tensor Output;

    public static PolicyTrainingSample Stack(IReadOnlyList<PolicyTrainingSample> samples, bool disposeInputs)
    {
        GameStateTensors[] gameStates = new GameStateTensors[samples.Count];
        Tensor[] outputs = new Tensor[samples.Count];
        Tensor[] workingHands = new Tensor[samples.Count];

        for (int i = 0; i < samples.Count; ++i)
        {
            gameStates[i] = samples[i].GameStateTensors;
            outputs[i] = samples[i].Output;
            workingHands[i] = samples[i].InUseMask;
        }
        PolicyTrainingSample result = new()
        {
            GameStateTensors = GameStateTensors.Stack(gameStates, disposeInputs),
            Output = concat(outputs, dim: 0),
            InUseMask = concat(workingHands, dim: 0)
        };
        if (disposeInputs)
        {
            for (int i = 0; i < samples.Count; ++i)
            {
                samples[i].Dispose();
            }
        }
        return result;
    }

    public void Dispose()
    {
        GameStateTensors.Dispose();
        InUseMask.Dispose();
        Output.Dispose();
    }
}

public struct EvaluationTrainingSample : IDisposable
{
    public GameStateTensors GameStateTensors;
    public Tensor Target; // scalar reward

    public static EvaluationTrainingSample Stack(IReadOnlyList<EvaluationTrainingSample> samples, bool disposeInputs)
    {
        GameStateTensors[] gameStates = new GameStateTensors[samples.Count];
        Tensor[] targets = new Tensor[samples.Count];

        for (int i = 0; i < samples.Count; ++i)
        {
            gameStates[i] = samples[i].GameStateTensors;
            targets[i] = samples[i].Target;
        }

        EvaluationTrainingSample result = new()
        {
            GameStateTensors = GameStateTensors.Stack(gameStates, disposeInputs),
            Target = concat(targets, dim: 0),
        };

        if (disposeInputs)
        {
            for (int i = 0; i < samples.Count; ++i)
            {
                samples[i].Dispose();
            }
        }

        return result;
    }

    public void Dispose()
    {
        GameStateTensors.Dispose();
        Target.Dispose();
    }
}
