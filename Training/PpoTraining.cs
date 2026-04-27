namespace Ramen.Training;

using System.Globalization;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public readonly record struct ExperimentConfig
(
    string ExperimentName,
    int RolloutSize,
    int RolloutParallelGameCount,
    int BatchSize,
    int TrainingEpochsPerStep,
    int StepCount,
    int SampledSoftmaxCount,
    float LearningRate,
    float AdamBeta1,
    float AdamBeta2,
    float WeightDecay,
    float PpoEpsilon,
    float EntropyCoefficient,
    int ValueReplayBufferCapacity,
    int SnapshotFrequency,
    int RandomSeed,
    int InitialHandsPerRound,
    int InitialDiscardsPerRound,
    bool UseRandomDeckInitializer,
    string ResumeSourceExperimentName,
    string NotebookReferenceExperimentName
);

public readonly record struct StepMetrics
(
    int Step,
    float WallClockSeconds,
    float AverageReward,
    float AverageMoveEntropy,
    float ValueMseMean,
    float PolicyLossMean,
    int CompletedGameCount,
    float LearningRate,
    int ValueReplayCount
);

public readonly record struct TrainingMetrics
(
    float PolicyLossMean,
    float ValueMseMean
);

public static class PpoTraining
{
    public static PpoRolloutDataset GenerateRollout(PpoPolicyValueModel model, ExperimentConfig config, Random random)
    {
        PpoRolloutDataset rollout = new(config.RolloutSize, config.SampledSoftmaxCount);
        GameData rolloutGameData = CreateConfiguredGameData(config);
        GameState[] gameStates = new GameState[config.RolloutParallelGameCount];
        List<TrajectoryPosition>[] activeTrajectories = new List<TrajectoryPosition>[config.RolloutParallelGameCount];
        float rewardSum = 0f;
        float entropySum = 0f;
        int rewardGameCount = 0;

        for (int slot = 0; slot < gameStates.Length; ++slot)
        {
            gameStates[slot] = new(rolloutGameData);
            activeTrajectories[slot] = [];
        }

        while (rollout.Count < config.RolloutSize)
        {
            List<int> activeIndices = [];
            List<TrajectoryPosition> activePositions = [];

            for (int slot = 0; slot < gameStates.Length; ++slot)
            {
                GameState gameState = gameStates[slot];
                gameState.AdvanceToNextPlayerChoice();

                if (gameState.GameIsDone)
                {
                    FinalizeCompletedTrajectory(
                        rollout: rollout,
                        trajectory: activeTrajectories[slot],
                        terminalState: gameState,
                        rewardSum: ref rewardSum,
                        rewardGameCount: ref rewardGameCount,
                        entropySum: ref entropySum);
                    if (rollout.Count >= config.RolloutSize)
                        break;

                    gameStates[slot] = new(rolloutGameData);
                    activeTrajectories[slot] = [];
                    gameState = gameStates[slot];
                    gameState.AdvanceToNextPlayerChoice();
                }

                if (rollout.Count >= config.RolloutSize)
                    break;

                activeIndices.Add(slot);
                activePositions.Add(new(gameState, config.SampledSoftmaxCount));
            }

            if (rollout.Count >= config.RolloutSize)
                break;

            if (activeIndices.Count == 0)
                continue;

            using var scope = NewDisposeScope();

            (GameStateTensors stateTensors, UseHandTensors useHandTensors) = BuildRolloutBatch(activePositions);
            (Tensor logits, Tensor values) = model.GetPolicyLogitsAndValues(stateTensors, useHandTensors);
            Tensor illegalMask = BuildIllegalMoveMask(
                remainingHands: stateTensors.RemainingHands.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64),
                remainingDiscards: stateTensors.RemainingDiscards.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64));
            Tensor probs = functional.softmax(logits + illegalMask, dim: 1).to(CPU);
            float[] flatProbs = [.. probs.data<float>()];
            float[] flatValues = [.. values.to(CPU).data<float>()];

            for (int activeIndex = 0; activeIndex < activeIndices.Count; ++activeIndex)
            {
                int rowOffset = activeIndex * PpoPolicyValueModel.MoveCount;
                Span<float> rowSpan = flatProbs.AsSpan(rowOffset, PpoPolicyValueModel.MoveCount);
                activePositions[activeIndex].PositionValueEstimate = flatValues[activeIndex];
                int chosenMoveIndex = SampleMoveIndex(rowSpan, random);
                FillSampledSoftmaxTargets(
                    position: activePositions[activeIndex],
                    chosenMoveIndex: chosenMoveIndex,
                    fullProbs: rowSpan,
                    random: random);

                activeTrajectories[activeIndices[activeIndex]].Add(activePositions[activeIndex]);
                UseHandMove move = PolicyOnlyAgent.MoveForIndex(
                    state: gameStates[activeIndices[activeIndex]],
                    index: chosenMoveIndex);
                move.Apply(gameStates[activeIndices[activeIndex]]);
            }
        }

        rollout.SetMetrics(
            averageReward: rewardGameCount == 0 ? 0f : rewardSum / rewardGameCount,
            averageMoveEntropy: rollout.Count == 0 ? 0f : entropySum / rollout.Count,
            completedGameCount: rewardGameCount);
        return rollout;
    }


    public static TrainingMetrics TrainStep(
        PpoPolicyValueModel model,
        AdamW optimizer,
        PpoRolloutDataset rollout,
        PpoMiniBatchBuffers batchBuffers,
        Random shuffleRandom,
        ExperimentConfig config)
    {
        float totalPolicyLoss = 0f;
        float totalValueMse = 0f;
        int batchCount = 0;

        for (int epoch = 0; epoch < config.TrainingEpochsPerStep; ++epoch)
        {
            int[] shuffledIndices = BuildShuffledIndices(rollout.Count, shuffleRandom);

            for (int batchStart = 0; batchStart < rollout.Count; batchStart += config.BatchSize)
            {
                using var scope = NewDisposeScope();

                rollout.FillBatch(
                    shuffledIndices: shuffledIndices,
                    batchStart: batchStart,
                    batchSize: config.BatchSize,
                    batchBuffers: batchBuffers);

                GameStateTensors stateTensors = new()
                {
                    FullHand = batchBuffers.Batch.StateTensors.FullHand.to(PpoPolicyValueModel.EvalDevice),
                    RemainingDeck = batchBuffers.Batch.StateTensors.RemainingDeck.to(PpoPolicyValueModel.EvalDevice),
                    RemainingHands = batchBuffers.Batch.StateTensors.RemainingHands.to(PpoPolicyValueModel.EvalDevice),
                    RemainingDiscards = batchBuffers.Batch.StateTensors.RemainingDiscards.to(PpoPolicyValueModel.EvalDevice),
                    OwnedJokers = batchBuffers.Batch.StateTensors.OwnedJokers.to(PpoPolicyValueModel.EvalDevice),
                    StoreJokers = batchBuffers.Batch.StateTensors.StoreJokers.to(PpoPolicyValueModel.EvalDevice),
                    StorePrices = batchBuffers.Batch.StateTensors.StorePrices.to(PpoPolicyValueModel.EvalDevice),
                    RerollPrice = batchBuffers.Batch.StateTensors.RerollPrice.to(PpoPolicyValueModel.EvalDevice),
                    Money = batchBuffers.Batch.StateTensors.Money.to(PpoPolicyValueModel.EvalDevice),
                    Round = batchBuffers.Batch.StateTensors.Round.to(PpoPolicyValueModel.EvalDevice),
                    Stage = batchBuffers.Batch.StateTensors.Stage.to(PpoPolicyValueModel.EvalDevice),
                    Score = batchBuffers.Batch.StateTensors.Score.to(PpoPolicyValueModel.EvalDevice),
                };
                UseHandTensors useHandTensors = new()
                {
                    Score = batchBuffers.Batch.UseHandTensors.Score.to(PpoPolicyValueModel.EvalDevice),
                };
                Tensor sampledMoveIndices = batchBuffers.Batch.SampledMoveIndices.to(PpoPolicyValueModel.EvalDevice);
                Tensor sampledMoveLogQ = batchBuffers.Batch.SampledMoveLogQ.to(PpoPolicyValueModel.EvalDevice);
                Tensor sampledMoveValidMask = batchBuffers.Batch.SampledMoveValidMask.to(PpoPolicyValueModel.EvalDevice);
                Tensor oldSampledProbs = batchBuffers.Batch.OldSampledMoveProbs.to(PpoPolicyValueModel.EvalDevice);
                Tensor valueTargets = batchBuffers.Batch.ValueTargets.to(PpoPolicyValueModel.EvalDevice);

                optimizer.zero_grad();

                (Tensor sampledLogits, Tensor values) = model.GetSelectedPolicyLogitsAndValues(
                    gameStateTensors: stateTensors,
                    useHandTensors: useHandTensors,
                    moveIndices: sampledMoveIndices);
                Tensor correctedLogits = sampledLogits - sampledMoveLogQ;
                Tensor invalidMask = (1f - sampledMoveValidMask) * -1e9f;
                Tensor maskedCorrectedLogits = correctedLogits + invalidMask;
                Tensor logProbs = functional.log_softmax(maskedCorrectedLogits, dim: 1);
                Tensor probs = exp(logProbs) * sampledMoveValidMask;
                Tensor entropy = -(probs * logProbs).sum(dim: 1).mean();
                Tensor logPiNew = logProbs.select(1, 0);
                Tensor logPiOld = oldSampledProbs.select(1, 0).clamp_min(1e-9f).log();
                Tensor ratio = exp(logPiNew - logPiOld);
                Tensor advantages = (valueTargets - values).detach();
                Tensor clippedRatio = clamp(
                    ratio,
                    min: 1f - config.PpoEpsilon,
                    max: 1f + config.PpoEpsilon);
                Tensor surrogate = min(ratio * advantages, clippedRatio * advantages);
                Tensor policyLoss = -surrogate.mean() - config.EntropyCoefficient * entropy;
                Tensor valueLoss = functional.mse_loss(values, valueTargets);
                Tensor totalLoss = policyLoss + valueLoss;
                totalLoss.backward();

                optimizer.step();

                totalPolicyLoss += policyLoss.item<float>();
                totalValueMse += valueLoss.item<float>();
                batchCount++;
            }
        }

        float divisor = Math.Max(batchCount, 1);
        return new(
            PolicyLossMean: totalPolicyLoss / divisor,
            ValueMseMean: totalValueMse / divisor);
    }


    public static float GetLearningRate(int continuationStep, float continuationLearningRate)
    {
        _ = continuationStep;
        return continuationLearningRate;
    }


    public static void SetOptimizerLearningRate(AdamW optimizer, float learningRate)
    {
        foreach (TorchSharp.torch.optim.ILearningRateController learningRateController in optimizer.ParamGroups)
        {
            learningRateController.LearningRate = learningRate;
            learningRateController.InitialLearningRate = learningRate;
        }
    }


    public static float GetStandardReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalScore >= 300f)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalScore / 1000f;
    }


    public static void WriteMetricsCsv(string filePath, IReadOnlyList<StepMetrics> metrics)
    {
        CSVBuilder output = new();
        for (int metricIndex = 0; metricIndex < metrics.Count; ++metricIndex)
        {
            StepMetrics metric = metrics[metricIndex];
            output
                .NextRow()
                .SetCell("step", metric.Step)
                .SetCell("wall_clock_seconds", metric.WallClockSeconds)
                .SetCell("average_reward", metric.AverageReward)
                .SetCell("average_move_entropy", metric.AverageMoveEntropy)
                .SetCell("value_mse_mean", metric.ValueMseMean)
                .SetCell("policy_loss_mean", metric.PolicyLossMean)
                .SetCell("completed_game_count", metric.CompletedGameCount)
                .SetCell("learning_rate", metric.LearningRate);
        }

        File.WriteAllText(filePath, output.ToString());
    }


    public static List<StepMetrics> LoadExistingMetrics(string filePath, int maxStepInclusive)
    {
        List<StepMetrics> metrics = [];
        if (!File.Exists(filePath))
            return metrics;

        string[] lines = File.ReadAllLines(filePath);
        for (int lineIndex = 1; lineIndex < lines.Length; ++lineIndex)
        {
            string line = lines[lineIndex].Trim();
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] cells = line.Split(',');
            if (cells.Length < 7)
                continue;

            int step = int.Parse(cells[0], CultureInfo.InvariantCulture);
            if (step > maxStepInclusive)
                continue;

            metrics.Add(new(
                Step: step,
                WallClockSeconds: float.Parse(cells[1], CultureInfo.InvariantCulture),
                AverageReward: float.Parse(cells[2], CultureInfo.InvariantCulture),
                AverageMoveEntropy: float.Parse(cells[3], CultureInfo.InvariantCulture),
                ValueMseMean: float.Parse(cells[4], CultureInfo.InvariantCulture),
                PolicyLossMean: float.Parse(cells[5], CultureInfo.InvariantCulture),
                CompletedGameCount: int.Parse(cells[6], CultureInfo.InvariantCulture),
                LearningRate: cells.Length > 7 ? float.Parse(cells[7], CultureInfo.InvariantCulture) : 0f,
                ValueReplayCount: 0));
        }

        return metrics;
    }


    static GameData CreateConfiguredGameData(ExperimentConfig config)
    {
        GameData gameData = new()
        {
            Seed = GameData.Default.Seed,
            Hands = config.InitialHandsPerRound,
            Discards = config.InitialDiscardsPerRound,
            RandomizeSeed = GameData.Default.RandomizeSeed,
            StartingHandBaseScore = [.. GameData.Default.StartingHandBaseScore],
            PlanetScores = [.. GameData.Default.PlanetScores],
            InitStartingDeck = config.UseRandomDeckInitializer ? GameData.InitErraticStartingDeck : GameData.InitErraticStartingDeck,
        };

        return gameData;
    }


    static void FinalizeCompletedTrajectory(
        PpoRolloutDataset rollout,
        List<TrajectoryPosition> trajectory,
        GameState terminalState,
        ref float rewardSum,
        ref int rewardGameCount,
        ref float entropySum)
    {
        if (trajectory.Count == 0)
            return;

        float finalReward = GetStandardReward(terminalState);
        float[] suffixValues = new float[trajectory.Count + 1];
        for (int index = trajectory.Count - 1; index >= 0; --index)
            suffixValues[index] = suffixValues[index + 1] + trajectory[index].PositionValueEstimate;

        int remainingCapacity = rollout.Capacity - rollout.Count;
        int addCount = Math.Min(remainingCapacity, trajectory.Count);
        if (addCount <= 0)
            return;

        rewardSum += finalReward;
        rewardGameCount++;

        for (int index = 0; index < addCount; ++index)
        {
            float futureValueSum = suffixValues[index + 1] + finalReward;
            int futureCount = trajectory.Count - index;
            float valueTarget = futureValueSum / futureCount;
            rollout.AddSample(trajectory[index], valueTarget);
            entropySum += trajectory[index].PolicyEntropy;
        }
    }


    static (GameStateTensors stateTensors, UseHandTensors useHandTensors) BuildRolloutBatch(IReadOnlyList<TrajectoryPosition> positions)
    {
        long[,] fullHand = new long[positions.Count, GameData.HandSize];
        long[,] remainingDeck = new long[positions.Count, 52];
        long[] remainingHands = new long[positions.Count];
        long[] remainingDiscards = new long[positions.Count];
        long[,] ownedJokers = new long[positions.Count, GameStateEmbedder.MaxOwnedJokerCount];
        long[,] storeJokers = new long[positions.Count, GameStateEmbedder.MaxStoreJokerCount];
        long[,] storePrices = new long[positions.Count, GameStateEmbedder.MaxStoreJokerCount];
        long[] rerollPrice = new long[positions.Count];
        long[] money = new long[positions.Count];
        long[] round = new long[positions.Count];
        long[] stage = new long[positions.Count];
        float[,] score = new float[positions.Count, 1];
        float[,] useHandScores = new float[positions.Count, PpoPolicyValueModel.UseableHandCount];

        for (int batchIndex = 0; batchIndex < positions.Count; ++batchIndex)
        {
            TrajectoryPosition position = positions[batchIndex];
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                fullHand[batchIndex, cardIndex] = position.FullHand[cardIndex];

            for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                remainingDeck[batchIndex, cardIndex] = position.RemainingDeck[cardIndex];

            remainingHands[batchIndex] = position.RemainingHands;
            remainingDiscards[batchIndex] = position.RemainingDiscards;
            rerollPrice[batchIndex] = position.RerollPrice;
            money[batchIndex] = position.Money;
            round[batchIndex] = position.Round;
            stage[batchIndex] = position.Stage;
            score[batchIndex, 0] = position.Score;

            for (int jokerIndex = 0; jokerIndex < GameStateEmbedder.MaxOwnedJokerCount; ++jokerIndex)
                ownedJokers[batchIndex, jokerIndex] = position.OwnedJokers[jokerIndex];

            for (int jokerIndex = 0; jokerIndex < GameStateEmbedder.MaxStoreJokerCount; ++jokerIndex)
            {
                storeJokers[batchIndex, jokerIndex] = position.StoreJokers[jokerIndex];
                storePrices[batchIndex, jokerIndex] = position.StorePrices[jokerIndex];
            }

            for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
                useHandScores[batchIndex, handIndex] = position.UseHandScores[handIndex];
        }

        return (
            stateTensors: new()
            {
                FullHand = tensor(fullHand, dtype: ScalarType.Int64),
                RemainingDeck = tensor(remainingDeck, dtype: ScalarType.Int64),
                RemainingHands = tensor(remainingHands, dtype: ScalarType.Int64),
                RemainingDiscards = tensor(remainingDiscards, dtype: ScalarType.Int64),
                OwnedJokers = tensor(ownedJokers, dtype: ScalarType.Int64),
                StoreJokers = tensor(storeJokers, dtype: ScalarType.Int64),
                StorePrices = tensor(storePrices, dtype: ScalarType.Int64),
                RerollPrice = tensor(rerollPrice, dtype: ScalarType.Int64),
                Money = tensor(money, dtype: ScalarType.Int64),
                Round = tensor(round, dtype: ScalarType.Int64),
                Stage = tensor(stage, dtype: ScalarType.Int64),
                Score = tensor(score, dtype: ScalarType.Float32),
            },
            useHandTensors: new()
            {
                Score = tensor(useHandScores, dtype: ScalarType.Float32),
            });
    }


    static void FillSampledSoftmaxTargets(
        TrajectoryPosition position,
        int chosenMoveIndex,
        ReadOnlySpan<float> fullProbs,
        Random random)
    {
        Array.Clear(position.SampledMoveIndices, 0, position.SampledMoveIndices.Length);
        Array.Clear(position.OldSampledMoveProbs, 0, position.OldSampledMoveProbs.Length);
        Array.Clear(position.SampledMoveLogQ, 0, position.SampledMoveLogQ.Length);
        Array.Clear(position.SampledMoveValidMask, 0, position.SampledMoveValidMask.Length);

        position.SampledMoveIndices[0] = chosenMoveIndex;
        position.SampledMoveValidMask[0] = 1f;
        position.PolicyEntropy = GetEntropy(fullProbs);

        int[] candidatePool = new int[PpoPolicyValueModel.MoveCount];
        int poolCount = BuildNegativeMovePool(
            remainingHands: position.RemainingHands,
            remainingDiscards: position.RemainingDiscards,
            targetMoveIndex: chosenMoveIndex,
            output: candidatePool);
        int negativeSampleCount = Math.Min(position.SampledMoveIndices.Length - 1, poolCount);
        float negativeLogQ = negativeSampleCount == 0 ? 0f : MathF.Log(negativeSampleCount / (float)poolCount);

        for (int sampleIndex = 1; sampleIndex <= negativeSampleCount; ++sampleIndex)
        {
            int selectedIndex = random.Next(poolCount);
            position.SampledMoveIndices[sampleIndex] = candidatePool[selectedIndex];
            position.SampledMoveLogQ[sampleIndex] = negativeLogQ;
            position.SampledMoveValidMask[sampleIndex] = 1f;
            poolCount--;
            candidatePool[selectedIndex] = candidatePool[poolCount];
        }

        int validCount = negativeSampleCount + 1;
        float maxCorrectedLogit = float.NegativeInfinity;
        float[] correctedLogits = new float[validCount];

        for (int sampleIndex = 0; sampleIndex < validCount; ++sampleIndex)
        {
            int moveIndex = (int)position.SampledMoveIndices[sampleIndex];
            float correctedLogit = MathF.Log(MathF.Max(fullProbs[moveIndex], 1e-9f)) - position.SampledMoveLogQ[sampleIndex];
            correctedLogits[sampleIndex] = correctedLogit;
            if (correctedLogit > maxCorrectedLogit)
                maxCorrectedLogit = correctedLogit;
        }

        float expSum = 0f;
        for (int sampleIndex = 0; sampleIndex < validCount; ++sampleIndex)
            expSum += MathF.Exp(correctedLogits[sampleIndex] - maxCorrectedLogit);

        for (int sampleIndex = 0; sampleIndex < validCount; ++sampleIndex)
            position.OldSampledMoveProbs[sampleIndex] = MathF.Exp(correctedLogits[sampleIndex] - maxCorrectedLogit) / MathF.Max(expSum, 1e-9f);
    }


    static int BuildNegativeMovePool(long remainingHands, long remainingDiscards, int targetMoveIndex, int[] output)
    {
        int count = 0;
        if (remainingHands > 0 && remainingDiscards > 0)
        {
            for (int moveIndex = 0; moveIndex < PpoPolicyValueModel.MoveCount; ++moveIndex)
            {
                if (moveIndex == targetMoveIndex)
                    continue;

                output[count++] = moveIndex;
            }

            return count;
        }

        int actionOffset = remainingHands > 0 ? 0 : 1;
        for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
        {
            int moveIndex = handIndex * 2 + actionOffset;
            if (moveIndex == targetMoveIndex)
                continue;

            output[count++] = moveIndex;
        }

        return count;
    }


    static Tensor BuildIllegalMoveMask(Tensor remainingHands, Tensor remainingDiscards)
    {
        using var scope = NewDisposeScope();

        Tensor noHandsMask = remainingHands
            .eq(0)
            .to_type(ScalarType.Float32)
            .unsqueeze(-1);
        Tensor noDiscardsMask = remainingDiscards
            .eq(0)
            .to_type(ScalarType.Float32)
            .unsqueeze(-1);
        Tensor playMask = noHandsMask.expand(remainingHands.size(0), PpoPolicyValueModel.UseableHandCount);
        Tensor discardMask = noDiscardsMask.expand(remainingHands.size(0), PpoPolicyValueModel.UseableHandCount);
        Tensor stacked = stack([playMask, discardMask], dim: 2).view([remainingHands.size(0), PpoPolicyValueModel.MoveCount]);
        Tensor illegalMask = stacked * -1e9f;
        illegalMask.MoveToOuterDisposeScope();
        return illegalMask;
    }


    static int[] BuildShuffledIndices(int count, Random random)
    {
        int[] indices = new int[count];
        for (int index = 0; index < count; ++index)
            indices[index] = index;

        for (int index = indices.Length - 1; index > 0; --index)
        {
            int swapIndex = random.Next(index + 1);
            (indices[index], indices[swapIndex]) = (indices[swapIndex], indices[index]);
        }

        return indices;
    }


    static int SampleMoveIndex(ReadOnlySpan<float> probs, Random random)
    {
        float sample = (float)random.NextDouble();
        float cumulative = 0f;

        for (int moveIndex = 0; moveIndex < probs.Length; ++moveIndex)
        {
            cumulative += probs[moveIndex];
            if (sample <= cumulative)
                return moveIndex;
        }

        return probs.Length - 1;
    }


    static float GetEntropy(ReadOnlySpan<float> probs)
    {
        float entropy = 0f;
        for (int moveIndex = 0; moveIndex < probs.Length; ++moveIndex)
        {
            float prob = probs[moveIndex];
            if (prob <= 0f)
                continue;

            entropy -= prob * MathF.Log(MathF.Max(prob, 1e-9f));
        }

        return entropy;
    }
}

public sealed class PpoRolloutDataset : IDisposable
{
    readonly byte[,] _fullHand;
    readonly byte[,] _remainingDeck;
    readonly byte[] _remainingHands;
    readonly byte[] _remainingDiscards;
    readonly long[,] _ownedJokers;
    readonly long[,] _storeJokers;
    readonly long[,] _storePrices;
    readonly long[] _rerollPrice;
    readonly long[] _money;
    readonly long[] _round;
    readonly long[] _stage;
    readonly float[] _score;
    readonly float[,] _useHandScores;
    readonly long[,] _sampledMoveIndices;
    readonly float[,] _oldSampledMoveProbs;
    readonly float[,] _sampledMoveLogQ;
    readonly float[,] _sampledMoveValidMask;
    readonly float[] _valueTargets;

    public int Capacity { get; }

    public int Count { get; private set; }

    public float AverageReward { get; private set; }

    public float AverageMoveEntropy { get; private set; }

    public int CompletedGameCount { get; private set; }

    public PpoRolloutDataset(int capacity, int sampledSoftmaxCount)
    {
        Capacity = capacity;
        _fullHand = new byte[capacity, GameData.HandSize];
        _remainingDeck = new byte[capacity, 52];
        _remainingHands = new byte[capacity];
        _remainingDiscards = new byte[capacity];
        _ownedJokers = new long[capacity, GameStateEmbedder.MaxOwnedJokerCount];
        _storeJokers = new long[capacity, GameStateEmbedder.MaxStoreJokerCount];
        _storePrices = new long[capacity, GameStateEmbedder.MaxStoreJokerCount];
        _rerollPrice = new long[capacity];
        _money = new long[capacity];
        _round = new long[capacity];
        _stage = new long[capacity];
        _score = new float[capacity];
        _useHandScores = new float[capacity, PpoPolicyValueModel.UseableHandCount];
        _sampledMoveIndices = new long[capacity, sampledSoftmaxCount];
        _oldSampledMoveProbs = new float[capacity, sampledSoftmaxCount];
        _sampledMoveLogQ = new float[capacity, sampledSoftmaxCount];
        _sampledMoveValidMask = new float[capacity, sampledSoftmaxCount];
        _valueTargets = new float[capacity];
    }


    public void Dispose()
    {
    }


    public void AddSample(TrajectoryPosition position, float valueTarget)
    {
        int sampleIndex = Count;
        for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
            _fullHand[sampleIndex, cardIndex] = (byte)position.FullHand[cardIndex];

        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
            _remainingDeck[sampleIndex, cardIndex] = (byte)position.RemainingDeck[cardIndex];

        _remainingHands[sampleIndex] = (byte)position.RemainingHands;
        _remainingDiscards[sampleIndex] = (byte)position.RemainingDiscards;
        _rerollPrice[sampleIndex] = position.RerollPrice;
        _money[sampleIndex] = position.Money;
        _round[sampleIndex] = position.Round;
        _stage[sampleIndex] = position.Stage;
        _score[sampleIndex] = position.Score;
        _valueTargets[sampleIndex] = valueTarget;

        for (int jokerIndex = 0; jokerIndex < GameStateEmbedder.MaxOwnedJokerCount; ++jokerIndex)
            _ownedJokers[sampleIndex, jokerIndex] = position.OwnedJokers[jokerIndex];

        for (int jokerIndex = 0; jokerIndex < GameStateEmbedder.MaxStoreJokerCount; ++jokerIndex)
        {
            _storeJokers[sampleIndex, jokerIndex] = position.StoreJokers[jokerIndex];
            _storePrices[sampleIndex, jokerIndex] = position.StorePrices[jokerIndex];
        }

        for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
            _useHandScores[sampleIndex, handIndex] = position.UseHandScores[handIndex];

        for (int sampleMoveIndex = 0; sampleMoveIndex < position.SampledMoveIndices.Length; ++sampleMoveIndex)
        {
            _sampledMoveIndices[sampleIndex, sampleMoveIndex] = position.SampledMoveIndices[sampleMoveIndex];
            _oldSampledMoveProbs[sampleIndex, sampleMoveIndex] = position.OldSampledMoveProbs[sampleMoveIndex];
            _sampledMoveLogQ[sampleIndex, sampleMoveIndex] = position.SampledMoveLogQ[sampleMoveIndex];
            _sampledMoveValidMask[sampleIndex, sampleMoveIndex] = position.SampledMoveValidMask[sampleMoveIndex];
        }

        Count++;
    }


    public void SetMetrics(float averageReward, float averageMoveEntropy, int completedGameCount)
    {
        AverageReward = averageReward;
        AverageMoveEntropy = averageMoveEntropy;
        CompletedGameCount = completedGameCount;
    }


    public void FillBatch(int[] shuffledIndices, int batchStart, int batchSize, PpoMiniBatchBuffers batchBuffers)
    {
        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            int sampleIndex = shuffledIndices[batchStart + batchIndex];

            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                batchBuffers.FullHand[batchIndex, cardIndex] = _fullHand[sampleIndex, cardIndex];

            for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                batchBuffers.RemainingDeck[batchIndex, cardIndex] = _remainingDeck[sampleIndex, cardIndex];

            batchBuffers.RemainingHands[batchIndex] = _remainingHands[sampleIndex];
            batchBuffers.RemainingDiscards[batchIndex] = _remainingDiscards[sampleIndex];
            batchBuffers.RerollPrice[batchIndex] = _rerollPrice[sampleIndex];
            batchBuffers.Money[batchIndex] = _money[sampleIndex];
            batchBuffers.Round[batchIndex] = _round[sampleIndex];
            batchBuffers.Stage[batchIndex] = _stage[sampleIndex];
            batchBuffers.Score[batchIndex, 0] = _score[sampleIndex];
            batchBuffers.ValueTargets[batchIndex] = _valueTargets[sampleIndex];

            for (int jokerIndex = 0; jokerIndex < GameStateEmbedder.MaxOwnedJokerCount; ++jokerIndex)
                batchBuffers.OwnedJokers[batchIndex, jokerIndex] = _ownedJokers[sampleIndex, jokerIndex];

            for (int jokerIndex = 0; jokerIndex < GameStateEmbedder.MaxStoreJokerCount; ++jokerIndex)
            {
                batchBuffers.StoreJokers[batchIndex, jokerIndex] = _storeJokers[sampleIndex, jokerIndex];
                batchBuffers.StorePrices[batchIndex, jokerIndex] = _storePrices[sampleIndex, jokerIndex];
            }

            for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
                batchBuffers.UseHandScores[batchIndex, handIndex] = _useHandScores[sampleIndex, handIndex];

            for (int sampleMoveIndex = 0; sampleMoveIndex < batchBuffers.SampledSoftmaxCount; ++sampleMoveIndex)
            {
                batchBuffers.SampledMoveIndices[batchIndex, sampleMoveIndex] = _sampledMoveIndices[sampleIndex, sampleMoveIndex];
                batchBuffers.OldSampledMoveProbs[batchIndex, sampleMoveIndex] = _oldSampledMoveProbs[sampleIndex, sampleMoveIndex];
                batchBuffers.SampledMoveLogQ[batchIndex, sampleMoveIndex] = _sampledMoveLogQ[sampleIndex, sampleMoveIndex];
                batchBuffers.SampledMoveValidMask[batchIndex, sampleMoveIndex] = _sampledMoveValidMask[sampleIndex, sampleMoveIndex];
            }
        }

        batchBuffers.RefreshTensors();
    }
}

public sealed class PpoMiniBatchBuffers : IDisposable
{
    readonly int _batchSize;

    public readonly long[,] FullHand;
    public readonly long[,] RemainingDeck;
    public readonly long[] RemainingHands;
    public readonly long[] RemainingDiscards;
    public readonly long[,] OwnedJokers;
    public readonly long[,] StoreJokers;
    public readonly long[,] StorePrices;
    public readonly long[] RerollPrice;
    public readonly long[] Money;
    public readonly long[] Round;
    public readonly long[] Stage;
    public readonly float[,] Score;
    public readonly float[,] UseHandScores;
    public readonly long[,] SampledMoveIndices;
    public readonly float[,] OldSampledMoveProbs;
    public readonly float[,] SampledMoveLogQ;
    public readonly float[,] SampledMoveValidMask;
    public readonly float[] ValueTargets;

    public PpoMiniBatch Batch;

    public int BatchSize => _batchSize;

    public int SampledSoftmaxCount { get; }

    public PpoMiniBatchBuffers(int batchSize, int sampledSoftmaxCount)
    {
        _batchSize = batchSize;
        SampledSoftmaxCount = sampledSoftmaxCount;
        FullHand = new long[_batchSize, GameData.HandSize];
        RemainingDeck = new long[_batchSize, 52];
        RemainingHands = new long[_batchSize];
        RemainingDiscards = new long[_batchSize];
        OwnedJokers = new long[_batchSize, GameStateEmbedder.MaxOwnedJokerCount];
        StoreJokers = new long[_batchSize, GameStateEmbedder.MaxStoreJokerCount];
        StorePrices = new long[_batchSize, GameStateEmbedder.MaxStoreJokerCount];
        RerollPrice = new long[_batchSize];
        Money = new long[_batchSize];
        Round = new long[_batchSize];
        Stage = new long[_batchSize];
        Score = new float[_batchSize, 1];
        UseHandScores = new float[_batchSize, PpoPolicyValueModel.UseableHandCount];
        SampledMoveIndices = new long[_batchSize, sampledSoftmaxCount];
        OldSampledMoveProbs = new float[_batchSize, sampledSoftmaxCount];
        SampledMoveLogQ = new float[_batchSize, sampledSoftmaxCount];
        SampledMoveValidMask = new float[_batchSize, sampledSoftmaxCount];
        ValueTargets = new float[_batchSize];
        Batch = new();
        RefreshTensors();
    }


    public void Dispose()
    {
        Batch.Dispose();
    }


    public void RefreshTensors()
    {
        Batch.Dispose();
        Batch = new()
        {
            StateTensors = new()
            {
                FullHand = tensor(FullHand, dtype: ScalarType.Int64),
                RemainingDeck = tensor(RemainingDeck, dtype: ScalarType.Int64),
                RemainingHands = tensor(RemainingHands, dtype: ScalarType.Int64),
                RemainingDiscards = tensor(RemainingDiscards, dtype: ScalarType.Int64),
                OwnedJokers = tensor(OwnedJokers, dtype: ScalarType.Int64),
                StoreJokers = tensor(StoreJokers, dtype: ScalarType.Int64),
                StorePrices = tensor(StorePrices, dtype: ScalarType.Int64),
                RerollPrice = tensor(RerollPrice, dtype: ScalarType.Int64),
                Money = tensor(Money, dtype: ScalarType.Int64),
                Round = tensor(Round, dtype: ScalarType.Int64),
                Stage = tensor(Stage, dtype: ScalarType.Int64),
                Score = tensor(Score, dtype: ScalarType.Float32),
            },
            UseHandTensors = new()
            {
                Score = tensor(UseHandScores, dtype: ScalarType.Float32),
            },
            SampledMoveIndices = tensor(SampledMoveIndices, dtype: ScalarType.Int64),
            OldSampledMoveProbs = tensor(OldSampledMoveProbs, dtype: ScalarType.Float32),
            SampledMoveLogQ = tensor(SampledMoveLogQ, dtype: ScalarType.Float32),
            SampledMoveValidMask = tensor(SampledMoveValidMask, dtype: ScalarType.Float32),
            ValueTargets = tensor(ValueTargets, dtype: ScalarType.Float32),
        };
    }
}

public sealed class PpoMiniBatch : Ramen.AgentTools.ITensorGroup
{
    public GameStateTensors StateTensors;
    public UseHandTensors UseHandTensors;
    public Tensor SampledMoveIndices;
    public Tensor OldSampledMoveProbs;
    public Tensor SampledMoveLogQ;
    public Tensor SampledMoveValidMask;
    public Tensor ValueTargets;
}

public sealed class TrajectoryPosition
{
    public readonly long[] FullHand = new long[GameData.HandSize];
    public readonly long[] RemainingDeck = new long[52];
    public readonly long[] OwnedJokers = new long[GameStateEmbedder.MaxOwnedJokerCount];
    public readonly long[] StoreJokers = new long[GameStateEmbedder.MaxStoreJokerCount];
    public readonly long[] StorePrices = new long[GameStateEmbedder.MaxStoreJokerCount];
    public readonly float[] UseHandScores = new float[PpoPolicyValueModel.UseableHandCount];
    public readonly long[] SampledMoveIndices;
    public readonly float[] OldSampledMoveProbs;
    public readonly float[] SampledMoveLogQ;
    public readonly float[] SampledMoveValidMask;

    public readonly long RemainingHands;
    public readonly long RemainingDiscards;
    public readonly long RerollPrice;
    public readonly long Money;
    public readonly long Round;
    public readonly long Stage;
    public readonly float Score;
    public float PositionValueEstimate;
    public float PolicyEntropy;

    public TrajectoryPosition(GameState gameState, int sampledSoftmaxCount)
    {
        for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
            FullHand[cardIndex] = gameState.HandState.Hand[cardIndex].ToIndex();

        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
            RemainingDeck[cardIndex] = cardIndex < deck.Length ? deck[cardIndex].ToIndex() : 0;

        RemainingHands = gameState.HandState.RemainingHands;
        RemainingDiscards = gameState.HandState.RemainingDiscards;
        RerollPrice = gameState.ShopState.CurrentRerollCost;
        Money = gameState.ShopState.Money;
        Round = gameState.Round;
        Stage = IsStoreStage(gameState.Stage) ? 1 : 0;
        Score = (float)gameState.ScoringState.CurrentRoundTotalScore / 300f;

        WriteJokerSlots(gameState.JokerState.Jokers, gameState.GameData, OwnedJokers);
        WriteJokerSlots(gameState.ShopState.ShopOfferings, gameState.GameData, StoreJokers);
        WriteStorePrices(gameState.ShopState.ShopOfferings, StorePrices);
        WriteUseHandScores(gameState, UseHandScores);

        SampledMoveIndices = new long[sampledSoftmaxCount];
        OldSampledMoveProbs = new float[sampledSoftmaxCount];
        SampledMoveLogQ = new float[sampledSoftmaxCount];
        SampledMoveValidMask = new float[sampledSoftmaxCount];
    }


    static bool IsStoreStage(StageOfGame stage)
    {
        return stage == StageOfGame.EnterShop || stage == StageOfGame.InShop;
    }


    static void WriteJokerSlots(IReadOnlyList<JokerInstance> jokers, GameData gameData, long[] output)
    {
        for (int jokerListIndex = 0; jokerListIndex < jokers.Count && jokerListIndex < output.Length; ++jokerListIndex)
        {
            JokerInstance joker = jokers[jokerListIndex];
            if (joker is null)
                continue;

            int jokerTypeIndex = GetJokerTypeIndex(gameData, joker.JokerData);
            output[jokerListIndex] = jokerTypeIndex + 1;
        }
    }


    static void WriteStorePrices(IReadOnlyList<JokerInstance> jokers, long[] output)
    {
        for (int jokerListIndex = 0; jokerListIndex < jokers.Count && jokerListIndex < output.Length; ++jokerListIndex)
        {
            JokerInstance joker = jokers[jokerListIndex];
            output[jokerListIndex] = joker?.JokerData.BasePrice ?? 0;
        }
    }


    static int GetJokerTypeIndex(GameData gameData, Joker joker)
    {
        for (int jokerIndex = 0; jokerIndex < gameData.Jokers.Length; ++jokerIndex)
        {
            if (ReferenceEquals(gameData.Jokers[jokerIndex], joker))
                return jokerIndex;
        }

        throw new InvalidOperationException($"Joker {joker.Name} was not found in the current game data.");
    }


    static void WriteUseHandScores(GameState gameState, float[] output)
    {
        int handCardCount = gameState.HandState.HandCardCount;

        for (int handIndex = 0; handIndex < PpoPolicyValueModel.HandCombinations.Length; ++handIndex)
        {
            int[] cardIndices = PpoPolicyValueModel.HandCombinations[handIndex];
            if (cardIndices[^1] >= handCardCount)
            {
                output[handIndex] = 0f;
                continue;
            }

            UseHandMove move = new(isDiscard: false, cardIndices);
            move.Apply(gameState);
            float scoreAfter = (float)gameState.ScoringState.CurrentRoundTotalScore;
            output[handIndex] = scoreAfter / 300f;
            move.Revert(gameState);
        }
    }
}
