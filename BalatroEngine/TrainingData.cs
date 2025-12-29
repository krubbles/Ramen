namespace BalatroAI;

using static TorchSharp.torch;

public static class TrainingData
{
    public static readonly List<EvaluationTrainingSample> EvaluationData = new();
    public static readonly List<PolicyTrainingSample> PolicyTrainingData = new();

    public static void GeneratePolicyTrainingData(AIModels models, int samples, int logCount, bool log = false)
    {
        int currentCount = PolicyTrainingData.Count;
        FastRandom random = FastRandom.SeededByClock();
        GameData gameData = new();
        using (no_grad())
        {
            int lastLoggedTrainingData = 0;
            while (PolicyTrainingData.Count - currentCount < samples)
            {
                gameData.Seed = random.Next();
                GameState gameState = new(gameData);
                gameState.StartRound();
                AIGameState aigs = new(gameState, models);
                while (aigs.CurrentMaxMoveCount() > 0)
                {
                    var sample = aigs.MakeTrainingSample(true, TrainingConfig.SampleCount, log && PolicyTrainingData.Count - currentCount < logCount);
                    lock (PolicyTrainingData)
                    {
                        PolicyTrainingData.Add(sample);
                    }
                }
                if (PolicyTrainingData.Count - lastLoggedTrainingData > 500)
                {
                    if (log)
                        Console.WriteLine("Sample Count: " + PolicyTrainingData.Count);
                    lastLoggedTrainingData = PolicyTrainingData.Count;
                }
            }
        }
    }

    public static void GenerateEvaluationTrainingData(AIModels models, int episodes, bool log = false)
    {
        FastRandom random = FastRandom.SeededByClock();
        GameData gameData = new();
        using (no_grad())
        {
            for (int e = 0; e < episodes; ++e)
            {
                gameData.Seed = random.Next();
                GameState gameState = new(gameData);
                gameState.StartRound();
                AIGameState aigs = new(gameState, models);

                List<GameStateTensors> states = new();

                // simulate until round/game end (RemainingHands < 4)
                while (aigs.GameState.HandState.RemainingHands >= 4)
                {
                    // ensure embedded tensors are up to date and batch size 1
                    aigs.UpdateGameStateTensors();
                    // clone so stored tensors are independent
                    var gsClone = aigs.GameStateTensors.Clone();
                    states.Add(gsClone);

                    // advance game stochastically
                    aigs.MakeMoveStochastic(TrainingConfig.GoodPlayTemperature);
                }

                float finalReward = aigs.GetCurrentReward();
                Tensor target = new float[] { finalReward };

                // add samples to EvaluationData under lock
                lock (EvaluationData)
                {
                    foreach (var st in states)
                    {
                        EvaluationData.Add(new EvaluationTrainingSample() { GameStateTensors = st, Target = target.clone() });
                    }
                }

                // dispose the original target tensor we made
                target.Dispose();
            }
        }
    }

}

public struct GameStateTensors : IDisposable
{
    public Tensor FullHand;
    public Tensor OtherState;

    public void MakeBatchSize1()
    {
        FullHand = FullHand.unsqueeze(0);
        OtherState = OtherState.unsqueeze(0);
    }

    public static GameStateTensors Stack(IReadOnlyList<GameStateTensors> tensors, bool disposeInputs)
    {
        Tensor[] handStates = new Tensor[tensors.Count];
        Tensor[] otherStates = new Tensor[tensors.Count];
        for (int i = 0; i < tensors.Count; ++i)
        {
            handStates[i] = tensors[i].FullHand;
            otherStates[i] = tensors[i].OtherState;
        }

        GameStateTensors result = new()
        {
            FullHand = concat(handStates, dim: 0),
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
            FullHand = FullHand?.clone(),
            OtherState = OtherState?.clone()
        };
    }

    public void Dispose()
    {
        FullHand.Dispose();
        OtherState.Dispose();
    }

    public static GameStateTensors Create(GameState gameState)
    {
        float[] otherVec = [(float)gameState.ScoringState.CurrentRoundTotalChips, gameState.HandState.RemainingDiscards, gameState.HandState.RemainingHands];
        Tensor otherState = otherVec;

        return new GameStateTensors() { FullHand = EmbedHand(gameState.HandState.Hand), OtherState = otherState };
    }

    public static Tensor EmbedHand(ReadOnlySpan<Card> hand)
    {
        Tensor[] cards = new Tensor[hand.Length + 1];
        for (int i = 0; i < cards.Length - 1; ++i)
        {
            cards[i] = EmbedCard(hand[i]);
        }
        cards[^1] = EmbedCard(Card.Null);

        Tensor handTensor = stack(cards);

        foreach (Tensor card in cards)
            card.Dispose();

        return handTensor;
    }

    public static Tensor EmbedCard(Card card)
    {
        float[] handArray = new float[53];

        int CardIndex(Card card) => ((int)card.Suit - 1) + (card.Rank - 2) * 4;

        if (card.IsNull)
        {
            handArray[^1] = 1f;
        }
        else
        {
            handArray[CardIndex(card)] += 1;
        }

        Tensor t = handArray;
        return t;
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
