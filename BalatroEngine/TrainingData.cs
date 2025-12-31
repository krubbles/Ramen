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
                GameState gameState = new(gameData);
                GameStateEmbedder embedding = new(gameState);

                List<GameStateTensors> states = new();

                while (gameState.HandState.RemainingHands > 0)
                {
                    gameState.AdvanceToNextPlayerChoice();
                    List<Move> moves = gameState.GetMoveOptions();
                    if (moves.Count == 0)
                        break;
                    Move bestMove = null;
                    float[] predictedRewards = new float[moves.Count];
                    float[] predictedStdDevs = new float[moves.Count];
                    for (int i = 0; i < moves.Count; ++i)
                    {
                        moves[i].Apply(gameState);
                        Tensor predictions = models.Evaluation.forward(embedding.Tensors);
                        float predictedReward = predictions[0, 0].item<float>();
                        float predictedStdDev = MathF.Log(predictions[0, 1].item<float>());
                        predictedRewards[i] = predictedReward;
                        predictedStdDevs[i] = predictedStdDev;
                        moves[i].Revert(gameState);
                    }
                    float[] probDist = MeanDistributionAnalyzer.GetProbabilityDistribution(predictedRewards, predictedStdDevs);
                    int moveIndex = MeanDistributionAnalyzer.SampleFromDistribution(random, probDist);

                    moves[moveIndex].Apply(gameState);
                    states.Add(embedding.TensorsCloned);
                }

                float reward = (float)gameState.ScoringState.CurrentRoundTotalChips / 100f;
                foreach (GameStateTensors state in states)
                {
                    EvaluationTrainingData.Add(new()
                    {
                        GameStateTensors = state,
                        Target = reward
                    });
                }
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
        RegisterCallbacks();
    }

    public GameStateTensors Tensors => new()
    {
        Hand = HandTensor,
        FullDeck = FullDeckTensor,
        RemainingDeck = RemainingDeckTensor,
        OtherState = OtherStateTensor,
    };

    public GameStateTensors TensorsCloned => _tensors.Clone().DetachFromDisposeScope();

    public Tensor HandTensor
    {
        get
        {
            if (!_handValid)
            {
                _handValid = true;
                EmbedHand();
            }
            return _tensors.Hand;
        }
    }

    public Tensor RemainingDeckTensor
    {
        get
        {
            if (!_remainingDeckValid)
            {
                _remainingDeckValid = true;
                EmbedRemainingDeck();
            }
            return _tensors.RemainingDeck;
        }
    }

    public Tensor FullDeckTensor
    {
        get
        {
            if (!_fullDeckValid)
            {
                _fullDeckValid = true;
                EmbedFullDeck();
            }
            return _tensors.FullDeck;
        }
    }

    public Tensor OtherStateTensor
    {
        get
        {
            if (!_otherStateValid)
            {
                _otherStateValid = true;
                EmbedOtherState();
            }
            return _tensors.FullDeck;
        }
    }

    void EmbedHand()
    {
        _tensors.Hand?.Dispose();
        _tensors.Hand = EmbedCardSet(GameState.HandState.Hand).DetachFromDisposeScope();
    }

    void EmbedRemainingDeck()
    {
        _tensors.RemainingDeck?.Dispose();
        _tensors.RemainingDeck = EmbedCardSet(GameState.DeckState.RemainingDeck).DetachFromDisposeScope();
    }

    void EmbedFullDeck()
    {
        _tensors.FullDeck?.Dispose();
        _tensors.FullDeck = EmbedCardSet(GameState.DeckState.FullDeck).DetachFromDisposeScope();
    }

    void EmbedOtherState()
    {
        _tensors.OtherState?.Dispose();
        _tensors.OtherState = tensor(
        [
            (float)GameState.ScoringState.CurrentRoundTotalChips,
            GameState.HandState.RemainingHands,
            GameState.HandState.RemainingDiscards
        ]).DetachFromDisposeScope();
    }

    void RegisterCallbacks() 
    {
        GameState.HandState.OnRemainingHandsOrDiscardsChanged += () => _otherStateValid = false;
        GameState.HandState.OnHandChanged += () => _handValid = false;
        GameState.DeckState.OnRemainingDeckChanged += () => _remainingDeckValid = false;
        GameState.DeckState.OnFullDeckChanged += () => _fullDeckValid = false;
    }

    static Tensor EmbedCardSet(ReadOnlySpan<Card> hand)
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

    static Tensor EmbedCard(Card card)
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
