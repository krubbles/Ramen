namespace BalatroAI;

using static TorchSharp.torch;

public static class TrainingData
{
    public static readonly List<EvaluationTrainingSample> EvaluationTrainingData = new();
    public static readonly List<PolicyTrainingSample> PolicyTrainingData = new();

    public const int PolicyOutputWidth = 9;

    public static void GeneratePolicyTrainingData(AIModels models, int samples)
    {
        using var scope = NewDisposeScope();
        int currentCount = PolicyTrainingData.Count;
        int lastLogCount = 0;
        FastRandom random = FastRandom.SeededByClock();
        GameData gameData = new();
        using (no_grad())
        {
            while (PolicyTrainingData.Count - currentCount < samples)
            {
                GameState gameState = new(gameData);
                AIGameState aigs = new(gameState, models);
                AIGameState aigsCopy = new(gameState, models);
                gameState.StartRound();
                while (aigs.CurrentMaxMoveCount() > 0)
                {
                    Tensor output = zeros(PolicyOutputWidth);
                    float bestReward = float.MinValue;
                    for (int move = 0; move < aigs.CurrentMaxMoveCount(); ++move)
                    {
                        if (!aigs.MoveIsValid(move))
                        {
                            continue;
                        }
                        aigsCopy.MakeMove(move);
                        float reward = models.GetExpectedReward(aigsCopy.GameStateTensors.Hand, aigsCopy.GameStateTensors.OtherState, aigsCopy.InUseMaskTensor).item<float>();
                        if (reward > bestReward)
                        {
                            bestReward = reward;
                            output.zero_();
                            output[move] = 1f;
                        }
                        else if (reward == bestReward)
                        {
                            output[move] = 1f;
                        }
                    }
                    output = output.unsqueeze(0);
                    PolicyTrainingData.Add(new()
                    {
                        GameStateTensors = aigs.GameStateTensors.Clone().DetachFromDisposeScope(),
                        InUseMask = aigs.InUseMaskTensor.clone().DetachFromDisposeScope(),
                        Output = output.clone().DetachFromDisposeScope()
                    });
                    if (PolicyTrainingData.Count - currentCount >= lastLogCount + 1000)
                    {
                        lastLogCount += 1000;
                        Console.WriteLine($"Generated {PolicyTrainingData.Count - currentCount} / {samples} policy training samples");
                    }
                    aigs.MakeMoveStochastic(TrainingConfig.ExploratoryPlayTemp);
                }
            }
        }
        Console.WriteLine("Final policy training data count: " + PolicyTrainingData.Count);
    }

    public static void GenerateEvaluationTrainingData(AIModels models, int samples)
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
                AIGameState aigs = new(gameState, models);

                List<(GameStateTensors gameState, Tensor inUseMask)> states = new();

                while (aigs.CurrentMaxMoveCount() > 0)
                {
                    states.Add((aigs.GameStateTensors.Clone(), aigs.InUseMaskTensor.clone()));
                    aigs.MakeMoveStochastic(TrainingConfig.GoodPlayTemp);
                }

                float finalReward = aigs.GetCurrentReward();
                Tensor target = tensor(finalReward).unsqueeze(0).unsqueeze(0);

                lock (EvaluationTrainingData)
                {
                    foreach (var st in states)
                    {
                        EvaluationTrainingData.Add(new() 
                        {
                            GameStateTensors = st.gameState.DetachFromDisposeScope(),
                            Target = target.clone().DetachFromDisposeScope(),
                            InUseMask = st.inUseMask.DetachFromDisposeScope()
                        });
                        if (EvaluationTrainingData.Count - startingSampleCount >= lastLogCount + 1000)
                        {
                            lastLogCount += 1000;
                            Console.WriteLine($"Generated {EvaluationTrainingData.Count - startingSampleCount} / {samples} evaluation training samples");
                        }
                    }
                }
                target.Dispose();
            }
        }
        Console.WriteLine("Final eval training data count: " + EvaluationTrainingData.Count);
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
                        GameStateTensors = GameStateTensors.Create(gameState).DetachFromDisposeScope(),
                        Target = target.clone().DetachFromDisposeScope(),
                        InUseMask = zeros(1, 9).DetachFromDisposeScope(),
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

public class GameStateEmbedder 
{
    public readonly GameState GameState;

    GameStateTensors _tensors;

    bool _handValid;
    bool _remainingDeckValid;
    bool _fullDeckValid;
    bool _otherStateValid;

    public GameStateEmbedder(GameState gameState)
    {
        GameState = gameState;
    }
}
public struct GameStateTensors : IDisposable
{
    public Tensor Hand;
    public Tensor RemainingDeck;
    public Tensor FullDeck;
    public Tensor OtherState;

    public void UpdateHandTensor(GameState gameState)
    {
        Hand = EmbedHand(gameState.HandState.Hand).unsqueeze(0);
    }

    public void UpdateOtherState(GameState other)
    {
        OtherState = tensor(
        [
            (float)other.ScoringState.CurrentRoundTotalChips,
            other.HandState.RemainingDiscards,
            other.HandState.RemainingHands
        ]).unsqueeze(0);
    }

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

    public GameStateTensors Clone()
    {
        return new()
        {
            Hand = Hand?.clone(),
            OtherState = OtherState?.clone()
        };
    }

    public void Dispose()
    {
        Hand.Dispose();
        OtherState.Dispose();
    }

    public GameStateTensors DetachFromDisposeScope()
    {
        Hand = Hand.DetachFromDisposeScope();
        OtherState = OtherState.DetachFromDisposeScope();
        return this;
    }   

    public static Tensor EmbedHand(ReadOnlySpan<Card> hand)
    {
        using var scope = NewDisposeScope();
        Tensor[] cards = new Tensor[hand.Length + 1];
        for (int i = 0; i < cards.Length - 1; ++i)
        {
            cards[i] = EmbedCard(hand[i]);
        }
        cards[^1] = EmbedCard(Card.Null);

        Tensor handTensor = stack(cards);
        return handTensor.MoveToOuterDisposeScope();
    }

    public static Tensor EmbedCard(Card card)
    {
        long value;
        if (card.IsNull)
        {
            value = 52; // Use index 52 for null cards
        }
        else
        {
            value = card.Rank - 2 + ((int)card.Suit - 1) * 13;
        }
        return tensor(value, dtype: ScalarType.Int64);
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
    public Tensor InUseMask;
    public Tensor Target; // (1) scalar

    public static EvaluationTrainingSample Stack(IReadOnlyList<EvaluationTrainingSample> samples, bool disposeInputs)
    {
        GameStateTensors[] gameStates = new GameStateTensors[samples.Count];
        Tensor[] targets = new Tensor[samples.Count];
        Tensor[] inUseMasks = new Tensor[samples.Count];

        for (int i = 0; i < samples.Count; ++i)
        {
            gameStates[i] = samples[i].GameStateTensors;
            targets[i] = samples[i].Target;
            inUseMasks[i] = samples[i].InUseMask;
        }

        EvaluationTrainingSample result = new()
        {
            GameStateTensors = GameStateTensors.Stack(gameStates, disposeInputs),
            Target = concat(targets, dim: 0),
            InUseMask = concat(inUseMasks, dim: 0)
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
        InUseMask.Dispose();
    }
}
