using System;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using BalatroAI;
using BalatroAI.ConsoleApp;
using System.Buffers;

class Program
{

    public static List<TrainingSample> TrainingData = new();

    public const int HandSize = 8;

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

    public static Card[] ExtractHand(Tensor fullHand, Tensor hand)
    {
        Tensor capped = min(fullHand, hand);
        float[] data = capped.data<float>().ToArray();
        capped.Dispose();
        List<Card> cards = new();
        List<float> extras = new();
        for (int i = 0; i < data.Length; ++i)
        {
            int count = (int)Math.Round(data[i]);
            if (count == 0)
                continue;
            float extra = data[i] - count;
            Card card = new(rank: (i / 4) + 2, suit: (Suit)(i % 4 + 1));
            for (int j = 0; j < count; ++j)
            {
                cards.Add(card);
                extras.Add(extra);
                extra++;
            }
        }
        while (cards.Count > 5)
        {
            int toRemoveIndex = 0;
            float lowestExtra = float.MaxValue;
            for (int i = 0; i < cards.Count; ++i)
            {
                if (extras[i] < lowestExtra)
                {
                    lowestExtra = extras[i];
                    toRemoveIndex = i;
                }
            }
            cards.RemoveAt(toRemoveIndex);
            extras.RemoveAt(toRemoveIndex);
        }
        if (cards.Count > 0)
        {
            return cards.ToArray();
        }
        else
        {
            float strongestActivation = float.MinValue;
            int strongestActivationIndex = 0;
            for (int i = 0; i < data.Length; ++i)
            {
                float activation = data[i];
                if (activation > strongestActivation)
                {
                    strongestActivation = activation;
                    strongestActivationIndex = i;
                }
            }
            return [new(rank: (strongestActivationIndex / 4) + 2, suit: (Suit)(strongestActivationIndex % 4 + 1))];
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
    }

    public static GameStateTensors EmbedGameState(GameState gameState)
    {
        float[] otherVec = [(float)gameState.ScoringState.CurrentRoundTotalChips, gameState.HandState.RemainingDiscards, gameState.HandState.RemainingHands];
        Tensor otherState = otherVec;

        return new GameStateTensors() { FullHand = EmbedHand(gameState.HandState.Hand), OtherState = otherState };
    }

    static void GenerateTrainingData(MoveSelectorModule model, int samples, bool useHighestScoringHand, int logCount, bool log)
    {
        int currentCount = TrainingData.Count;
        FastRandom random = FastRandom.SeededByClock();
        GameData gameData = new();
        using (no_grad())
        {
            int lastLoggedTrainingData = 0;
            while (TrainingData.Count - currentCount < samples)
            {
                gameData.Seed = random.Next();
                GameState gameState = new(gameData);
                gameState.StartRound();
                AIGameState aigs = new(gameState, model);
                while (aigs.CurrentMaxMoveCount() > 0)
                {
                    var sample = aigs.MakeTrainingSample(true, TrainingConfig.SampleCount, log && TrainingData.Count - currentCount < logCount);
                    lock (TrainingData)
                    {
                        TrainingData.Add(sample);
                    }
                }
                if (TrainingData.Count - lastLoggedTrainingData > 500)
                {
                    if (log)
                        Console.WriteLine("Sample Count: " + TrainingData.Count);
                    lastLoggedTrainingData = TrainingData.Count;
                }
            }
        }
    }

    static void Main()
    {
        int threadsDefault = get_num_threads();
        int interopThreadsDefault = get_num_interop_threads();
        Console.WriteLine($"Threading default: {threadsDefault}, interopThreads {interopThreadsDefault}");

        torch.set_num_threads(1);
        torch.set_num_interop_threads(1);

        // Reproducibility
        random.manual_seed(0);

        var device = CPU;

        // -------------------------
        // Generate synthetic dataset
        // -------------------------

        // Inputs: (N, 2)

        // -------------------------
        // Define linear regression model
        // -------------------------

        MoveSelectorModule model = new();

        var optimizer = optim.Adam(model.parameters(), lr: TrainingConfig.LearningRate);
        var lossFunc = CrossEntropyLoss();

        long totalTrainableParams = 0;

        // Iterate over all named parameters in the model
        foreach (var param in model.named_parameters())
        {
            // Check if the parameter requires a gradient (is trainable)
            if (param.parameter.requires_grad)
            {
                totalTrainableParams += param.parameter.numel();
            }
        }
        Console.WriteLine($"Total number of trainable parameters: {totalTrainableParams}");

        ShowExampleMoveRewards();

        {
            List<Task> tasks = new();
            for (int i = 0; i < 8; ++i)
            {
                int boxedI = i;
                tasks.Add(Task.Run(() =>
                {
                    GenerateTrainingData(model, TrainingConfig.DataSize, false, 0, log: boxedI == 0);
                }));
            }
            Task.WaitAll(tasks.ToArray());
        }
        // int toRemove = TrainingData.Count / 2;

        TrainingSample stackedSamples = default;
        lock (TrainingData)
        {
            stackedSamples = TrainingSample.Stack(TrainingData, false);
        }
        // -------------------------
        // Training loop
        // -------------------------

        int epochs = 1000;

        float lossAvg = 0;
        int batchCount = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {

            int samples = (int)stackedSamples.Output.size(dim: 0);
            int trainSampleCount = samples;
            int batchSize = TrainingConfig.BatchSize;


            lossAvg = 0;
            batchCount = 0;
            set_num_threads(threadsDefault);
            for (int i = 0; i < trainSampleCount; i += batchSize)
            {
                var batchFullHands = stackedSamples.GameStateTensors.FullHand[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOtherInputs = stackedSamples.GameStateTensors.OtherState[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchInUseMasks = stackedSamples.InUseMask[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOutputs = stackedSamples.Output[i..Math.Min(i + batchSize, trainSampleCount)];
                optimizer.zero_grad();
                Tensor fullHandEmbedded = model.EmbedCards(batchFullHands);
                var predictions = model.GetCardUseRewards(fullHandEmbedded, batchOtherInputs, batchInUseMasks);
                var loss = lossFunc.forward(predictions, batchOutputs);
                loss.backward();
                optimizer.step();
                lossAvg += loss.item<float>();
                batchCount++;

            }
            set_num_threads(1);

            lossAvg /= batchCount;

            Console.WriteLine($"Epoch {epoch} | Loss = {lossAvg}");

            if (epoch % 12 == 5)
            {
                FastRandom r = FastRandom.SeededByClock();
                float totalReward = 0;
                for (int i = 0; i < TrainingConfig.ScoringHeuristicSampleSize; ++i)
                {
                    GameState gs = new GameState(new());
                    gs.Reseed(r.Next());
                    gs.StartRound();
                    AIGameState aigs = new(gs, model);
                    totalReward += aigs.EstimateTrueReward(1, 0f) * 300;

                }
                ShowExampleMoveRewards();
                Console.WriteLine($"Score: {totalReward / TrainingConfig.ScoringHeuristicSampleSize}");
            }

            if (epoch % TrainingConfig.EpochsPerDataGen == TrainingConfig.EpochsPerDataGen - 1)
            {
                TrainingData.RemoveRange(0, TrainingConfig.DataGenAmount);
                List<Task> tasks = new();
                for (int i = 0; i < 8; ++i)
                {
                    int boxedI = i;
                    tasks.Add(Task.Run(() =>
                    {
                        GenerateTrainingData(model, TrainingConfig.DataGenAmount, false, 20, log: boxedI == 0);
                    }));
                }
                Task.WaitAll(tasks.ToArray());
                stackedSamples.Dispose();
                stackedSamples = TrainingSample.Stack(TrainingData, false);
            }

#if false
            var perm = torch.randperm(trainSampleCount);
            cardInputs[0..trainSampleCount] = cardInputs[0..trainSampleCount].index_select(0, perm);
            otherInputs[0..trainSampleCount] = otherInputs[0..trainSampleCount].index_select(0, perm);
            moveInputs[0..trainSampleCount] = moveInputs[0..trainSampleCount].index_select(0, perm);
            outputs[0..trainSampleCount] = outputs[0..trainSampleCount].index_select(0, perm);
#endif
        }

        void ShowExampleMoveRewards()
        {
            for (int i = 0; i < 3; ++i)
            {
                FastRandom r = FastRandom.SeededByClock();
                GameState gs = new GameState(new());
                gs.Reseed(r.Next());
                gs.StartRound();
                Console.WriteLine(gs.HandToString());
                AIGameState aigs = new(gs, model);
                float[] rewards = model.GetCardUseRewards(aigs.HandTensor, aigs.GameStateTensors.OtherState, aigs.InUseMaskTensor).div(1f).softmax(dim: 1).data<float>().ToArray();
                Console.WriteLine(LoggingUtility.FormatArray(rewards));
            }
        }
    }


    public struct TrainingSample : IDisposable
    {
        public GameStateTensors GameStateTensors;
        public Tensor InUseMask;
        public Tensor Output;

        public static TrainingSample Stack(IReadOnlyList<TrainingSample> samples, bool disposeInputs)
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
            TrainingSample result = new()
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

    class AIGameState
    {
        public MoveSelectorModule Model;
        public GameState GameState;
        public GameStateTensors _embeddedGameState;
        public FastRandom Random;

        Tensor _embeddedFullHandCache;
        bool _handCachesValid;
        bool _embeddedGameStateValid;

        public int ToUseCount;
        public int[] ToUseIndices = new int[30];
        public bool[] AlreadyPlayedCards = new bool[30];

        readonly float[] _cumWeight = new float[30];

        public Tensor InUseMaskTensor = zeros(1, HandSize + 1);

        public AIGameState(GameState gameState, MoveSelectorModule model)
        {
            GameState = gameState;
            Model = model;
            Random = FastRandom.SeededByClock();
        }

        public void CloneFrom(AIGameState other)
        {
            Model = other.Model;
            GameState.CloneFrom(other.GameState);
            GameState.Reseed(Random.Next());
            _embeddedGameState = other._embeddedGameState.Clone();
            _embeddedFullHandCache = other._embeddedFullHandCache?.clone();
            _handCachesValid = other._handCachesValid;
            _embeddedGameStateValid = other._embeddedGameStateValid;

            ToUseCount = other.ToUseCount;
            other.ToUseIndices.CopyTo(ToUseIndices.AsSpan(0, 30));
            other.AlreadyPlayedCards.CopyTo(AlreadyPlayedCards.AsSpan(0, 30));

            InUseMaskTensor.copy_(other.InUseMaskTensor);
        }

        public int CurrentMaxMoveCount()
        {
            if (GameState.HandState.RemainingHands == 3)
                return 0;

            int cardsInHand = GameState.HandState.CardsInHand;
            if (ToUseCount == 0)
                return cardsInHand;
            else
                return cardsInHand + 1; // extra for play hand 
        }

        public bool MoveIsValid(int move) // assumes move is in [0, CurrentMaxMoveCount())
        {
            return !AlreadyPlayedCards[move];
        }

        public void MakeMove(int move) // assumes move is valid
        {
            ToUseIndices[ToUseCount] = move;
            ToUseCount++;

            if (ToUseCount >= GameData.MaxPlayedHandSize || move == GameState.HandState.CardsInHand) // play hand
            {
                PlayCurrentHand();
                return;
            }

        }

        public float GetCurrentReward()
        {
            return (float)GameState.ScoringState.CurrentRoundTotalChips / 300f;
        }

        public Tensor HandTensor
        {
            get
            {
                UpdateHandCaches();
                return _embeddedFullHandCache;
            }
        }


        void UpdateHandCaches()
        {
            if (_handCachesValid)
                return;
            _handCachesValid = true;
            _embeddedFullHandCache?.Dispose();
            _embeddedFullHandCache = Model.EmbedCards(GameStateTensors.FullHand);
            _embeddedFullHandCache.DetachFromDisposeScope();
        }

        public GameStateTensors GameStateTensors
        {
            get
            {
                UpdateGameStateTensors();
                return _embeddedGameState;
            }
        }

        public void UpdateGameStateTensors()
        {
            if (_embeddedGameStateValid)
                return;
            _embeddedGameStateValid = true;
            _embeddedGameState = EmbedGameState(GameState);
            _embeddedGameState.MakeBatchSize1();
            _embeddedGameState.FullHand.DetachFromDisposeScope();
            _embeddedGameState.OtherState.DetachFromDisposeScope();
        }

        public void MarkGameStateChanged() 
        {
            _handCachesValid = false;
            _embeddedGameStateValid = false;
        }

        public void ResetUseCardsState()
        {
            InUseMaskTensor.zero_();
            ToUseCount = 0;
            Array.Clear(AlreadyPlayedCards);
        }

        public float[] GetExpectedRewardsForMoves(int samples)
        {
            int moveMax = CurrentMaxMoveCount();
            AIGameState copy = new(new GameState(GameState.GameData), Model);
            float greatestReward = -1;
            float[] output = new float[9];
            for (int move = 0; move < output.Length; ++move)
            {
                if (move >= moveMax || !MoveIsValid(move))
                {
                    output[move] = -1;
                    continue;
                }
                copy.CloneFrom(this);
                copy.MakeMove(move);
                float reward = copy.EstimateTrueReward(samples, TrainingConfig.Temperature);
                output[move] = reward;
                if (reward > greatestReward)
                    greatestReward = reward;
            }
            return output;
        }

        public TrainingSample MakeTrainingSample(bool playMoveStochastic, int samples, bool debugLog = false)
        {
            if (debugLog)
            {
                Console.WriteLine("------ Training Sample -------");
                Console.WriteLine(GameState.HandToString());
            }
            int moveMax = CurrentMaxMoveCount();
            float[] output = new float[9];

            AIGameState copy = new(new GameState(GameState.GameData), Model);
            MeanDistributionAnalyzer mda = new(9);
            for (int pass = 0; pass < samples; ++pass)
            {
                for (int move = 0; move < output.Length; ++move)
                {
                    if (move >= moveMax || !MoveIsValid(move))
                    {
                        output[move] = -1;
                        continue;
                    }
                    copy.CloneFrom(this);
                    copy.MakeMove(move);
                    float reward = copy.EstimateTrueReward(1, TrainingConfig.Temperature);
                    output[move] = reward;
                }
                mda.AddSample(output);
            }
            output = mda.GetProbabilityDistribution();

            if (debugLog)
            {
                Console.WriteLine(LoggingUtility.FormatArray(output));
                Console.WriteLine("To Use: " + LoggingUtility.FormatArray(ToUseIndices[0..ToUseCount]));
                Console.WriteLine("------------------------------");
            }

            Tensor outputTensor = output;

            if (playMoveStochastic)
            {
                PlayRandomMove(output, 3, ToUseCount > 0);
            }
            return new()
            {
                GameStateTensors = GameStateTensors.Clone(),
                Output = outputTensor.unsqueeze(0),
                InUseMask = InUseMaskTensor.clone()
            };
        }

        public float EstimateTrueReward(int samples, float temp)
        {
            AIGameState tester = new(new GameState(GameState.GameData), Model);
            float totalReward = 0;
            for (int i = 0; i < samples; ++i)
            {
                tester.CloneFrom(this);
                while (tester.CurrentMaxMoveCount() > 0)
                {
                    tester.MakeMoveStochastic(temp);
                }
                totalReward += tester.GetCurrentReward();
            }
            return totalReward / samples;
        }

        int PlayRandomMove(float[] weights, float temp, bool allowPlayNothing)
        {
            int moves = GameState.HandState.CardsInHand + (allowPlayNothing ? 1 : 0);
            float totalWeight = 0;
            for (int i = 0; i < moves; ++i)
            {
                totalWeight += weights[i];
            }
            float averageWeight = totalWeight / moves;
            for (int i = 0; i < moves; ++i)
            {
                if (!AlreadyPlayedCards[i])
                {
                    totalWeight += MathF.Exp(Math.Clamp((weights[i] - averageWeight) / Math.Max(0.00001f, temp), -40, 40));
                }
                _cumWeight[i] = totalWeight;
            }

            float sampleValue = Random.NextPortion() * totalWeight; // [0, 1)
            int playIndex = 0;
            for (int i = 0; i < _cumWeight.Length; ++i)
            {
                if (sampleValue < _cumWeight[i])
                {
                    playIndex = i;
                    AlreadyPlayedCards[i] = true;
                    break;
                }
            }

            MakeMove(playIndex);
            return playIndex;
        }

        public void MakeMoveStochastic(float temperature)
        {
            using var scope = NewDisposeScope();

            Tensor cardRewardsTensor = Model.GetCardUseRewards(HandTensor, GameStateTensors.OtherState, InUseMaskTensor);

            int playIndex = PlayRandomMove(cardRewardsTensor.data<float>().ToArray(), temperature, ToUseCount > 0);

            if (ToUseCount > 0) // didn't play hand
            {
                if (playIndex < 8)
                {
                    Tensor processedCardTensor = Model.EmbedCards(EmbedCard(GameState.HandState.Hand[playIndex])).unsqueeze(0);
                    InUseMaskTensor[0, playIndex] = 1;
                }
            }
        }

        void PlayCurrentHand()
        {
            GameState.HandState.PlayHand(ToUseIndices[0..ToUseCount]);
            MarkGameStateChanged();
            ResetUseCardsState();
        }
    }

    class ResidualMLP : Module<Tensor, Tensor>
    {
        private ModuleList<Linear> upLayers = new();
        private ModuleList<Linear> downLayers = new();

        private ModuleList<LayerNorm> norms = new();
        private ModuleList<GELU> activations = new();

        public ResidualMLP(int size, int depth) : base("ResidualMLP")
        {
            for (int i = 0; i < depth; ++i)
            {
                upLayers.append(Linear(size, size * 4));
                downLayers.append(Linear(size * 4, size));
                activations.append(GELU());
                norms.append(LayerNorm(size));
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            for (int i = 0; i < upLayers.Count; i++)
            {
                Tensor normed = norms[i].forward(x);
                Tensor up = upLayers[i].forward(x);
                Tensor activated = activations[i].forward(up);
                Tensor down = downLayers[i].forward(activated);
                x = x / upLayers.Count + down;
            }
            return x;
        }
    }
}