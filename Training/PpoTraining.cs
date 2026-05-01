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
    float ValueLossCoefficient,
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
    float ClipFractionMean,
    int CompletedGameCount,
    float LearningRate,
    int ValueReplayCount
);

public readonly record struct TrainingMetrics
(
    float PolicyLossMean,
    float ValueMseMean,
    float ClipFractionMean
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
                AdvanceToNextDecisionState(gameState);

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
                    AdvanceToNextDecisionState(gameState);
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

            ProcessRoundDecisions(
                model: model,
                random: random,
                gameStates: gameStates,
                activeIndices: activeIndices,
                activePositions: activePositions,
                activeTrajectories: activeTrajectories);
            ProcessStoreDecisions(
                model: model,
                random: random,
                gameStates: gameStates,
                activeIndices: activeIndices,
                activePositions: activePositions,
                activeTrajectories: activeTrajectories);
        }

        rollout.SetMetrics(
            averageReward: rewardGameCount == 0 ? 0f : rewardSum / rewardGameCount,
            averageMoveEntropy: rollout.Count == 0 ? 0f : entropySum / rollout.Count,
            completedGameCount: rewardGameCount);
        // Normalize rollout targets once so the policy and value losses stay on a stable scale.
        rollout.NormalizeSamples();
        return rollout;
    }


    public static TrainingMetrics TrainStep(
        PpoPolicyValueModel model,
        AdamW optimizer,
        PpoRolloutDataset rollout,
        Random shuffleRandom,
        ExperimentConfig config)
    {
        float totalPolicyLoss = 0f;
        float totalValueMse = 0f;
        float totalClipFraction = 0f;
        int metricBatchCount = 0;

        for (int epoch = 0; epoch < config.TrainingEpochsPerStep; ++epoch)
        {
            int[] shuffledRoundIndices = BuildShuffledIndices(rollout.RoundCount, shuffleRandom);
            int[] shuffledStoreIndices = BuildShuffledIndices(rollout.StoreCount, shuffleRandom);
            int roundBatchCount = DivideRoundUp(rollout.RoundCount, config.BatchSize);
            int storeBatchCount = DivideRoundUp(rollout.StoreCount, config.BatchSize);
            int totalBatchCount = roundBatchCount + storeBatchCount;

            // Bresenham-style dither: emit round and store batches proportionally,
            // completing exactly once every sample from each dataset has been covered.
            int roundBatchesEmitted = 0;
            int storeBatchesEmitted = 0;
            int ditherAccumulator = 0; // scaled by totalBatchCount

            for (int stepIndex = 0; stepIndex < totalBatchCount; ++stepIndex)
            {
                using var scope = NewDisposeScope();
                optimizer.zero_grad();

                // Decide whether to emit a store or round batch this step.
                // Advance the accumulator by storeBatchCount; if it exceeds half the
                // total we emit a store batch (and subtract totalBatchCount), otherwise
                // a round batch. This distributes store batches evenly across the epoch.
                bool emitStore;
                if (storeBatchesEmitted >= storeBatchCount)
                    emitStore = false;
                else if (roundBatchesEmitted >= roundBatchCount)
                    emitStore = true;
                else
                {
                    ditherAccumulator += storeBatchCount;
                    if (ditherAccumulator >= totalBatchCount)
                    {
                        ditherAccumulator -= totalBatchCount;
                        emitStore = true;
                    }
                    else
                        emitStore = false;
                }

                if (emitStore)
                {
                    int batchStart = storeBatchesEmitted * config.BatchSize;
                    using PpoStoreMiniBatch batch = rollout.Store.FillBatch(shuffledStoreIndices, batchStart, config.BatchSize);
                    (float policyLoss, float valueMse, float clipFraction) = TrainStoreBatch(model, optimizer, batch, config);
                    totalPolicyLoss += policyLoss;
                    totalValueMse += valueMse;
                    totalClipFraction += clipFraction;
                    metricBatchCount++;
                    storeBatchesEmitted++;
                }
                else
                {
                    int batchStart = roundBatchesEmitted * config.BatchSize;
                    using PpoRoundMiniBatch batch = rollout.Round.FillBatch(shuffledRoundIndices, batchStart, config.BatchSize);
                    (float policyLoss, float valueMse, float clipFraction) = TrainRoundBatch(model, optimizer, batch, config);
                    totalPolicyLoss += policyLoss;
                    totalValueMse += valueMse;
                    totalClipFraction += clipFraction;
                    metricBatchCount++;
                    roundBatchesEmitted++;
                }

                optimizer.step();
            }
        }

        float divisor = Math.Max(metricBatchCount, 1);
        return new(
            PolicyLossMean: totalPolicyLoss / divisor,
            ValueMseMean: totalValueMse / divisor,
            ClipFractionMean: totalClipFraction / divisor);
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
        float roundsSurvived = gameState.Round / 3f;
        return roundsSurvived * roundsSurvived;
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
                .SetCell("clip_fraction_mean", metric.ClipFractionMean)
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
                ClipFractionMean: cells.Length > 8 ? float.Parse(cells[6], CultureInfo.InvariantCulture) : 0f,
                CompletedGameCount: int.Parse(cells.Length > 8 ? cells[7] : cells[6], CultureInfo.InvariantCulture),
                LearningRate: cells.Length > 8 ? float.Parse(cells[8], CultureInfo.InvariantCulture) : (cells.Length > 7 ? float.Parse(cells[7], CultureInfo.InvariantCulture) : 0f),
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
            float advantage = valueTarget - trajectory[index].PositionValueEstimate;
            rollout.AddSample(trajectory[index], valueTarget, advantage);
            entropySum += trajectory[index].PolicyEntropy;
        }
    }


    static void ProcessRoundDecisions(
        PpoPolicyValueModel model,
        Random random,
        IReadOnlyList<GameState> gameStates,
        IReadOnlyList<int> activeIndices,
        IReadOnlyList<TrajectoryPosition> activePositions,
        IReadOnlyList<List<TrajectoryPosition>> activeTrajectories)
    {
        List<int> roundDecisionSlots = [];
        List<TrajectoryPosition> roundPositions = [];
        for (int activeIndex = 0; activeIndex < activePositions.Count; ++activeIndex)
        {
            if (activePositions[activeIndex].IsStoreState)
                continue;

            roundDecisionSlots.Add(activeIndex);
            roundPositions.Add(activePositions[activeIndex]);
        }

        if (roundPositions.Count == 0)
            return;

        using var scope = NewDisposeScope();

        int actualCount = roundPositions.Count;
        int paddedCount = PadToPowerOfTwo(actualCount);
        if (paddedCount > actualCount)
            for (int i = actualCount; i < paddedCount; ++i)
                roundPositions.Add(roundPositions[0]);

        (GameStateTensors stateTensors, UseHandTensors useHandTensors) = BuildRoundRolloutBatch(roundPositions);
        (Tensor logits, Tensor values) = model.GetPolicyLogitsAndValues(stateTensors, useHandTensors);
        Tensor illegalMask = BuildIllegalMoveMask(
            remainingHands: stateTensors.RemainingHands.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64),
            remainingDiscards: stateTensors.RemainingDiscards.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64));
        Tensor probs = functional.softmax(logits + illegalMask, dim: 1).to(CPU);
        float[] flatProbs = [.. probs.data<float>()];
        float[] flatValues = [.. values.to(CPU).data<float>()];

        for (int batchIndex = 0; batchIndex < actualCount; ++batchIndex)
        {
            int rowOffset = batchIndex * PpoPolicyValueModel.MoveCount;
            Span<float> rowSpan = flatProbs.AsSpan(rowOffset, PpoPolicyValueModel.MoveCount);
            TrajectoryPosition position = roundPositions[batchIndex];
            position.PositionValueEstimate = flatValues[batchIndex];
            int chosenMoveIndex = SampleMoveIndex(rowSpan, random);
            FillRoundTargets(
                position: position,
                chosenMoveIndex: chosenMoveIndex,
                fullProbs: rowSpan,
                random: random);

            int activeSlot = roundDecisionSlots[batchIndex];
            int gameSlot = activeIndices[activeSlot];
            activeTrajectories[gameSlot].Add(position);
            UseHandMove move = PolicyOnlyAgent.MoveForIndex(gameStates[gameSlot], chosenMoveIndex);
            move.Apply(gameStates[gameSlot]);
        }
    }


    static void ProcessStoreDecisions(
        PpoPolicyValueModel model,
        Random random,
        IReadOnlyList<GameState> gameStates,
        IReadOnlyList<int> activeIndices,
        IReadOnlyList<TrajectoryPosition> activePositions,
        IReadOnlyList<List<TrajectoryPosition>> activeTrajectories)
    {
        List<int> storeDecisionSlots = [];
        List<TrajectoryPosition> storePositions = [];
        for (int activeIndex = 0; activeIndex < activePositions.Count; ++activeIndex)
        {
            if (!activePositions[activeIndex].IsStoreState)
                continue;

            storeDecisionSlots.Add(activeIndex);
            storePositions.Add(activePositions[activeIndex]);
        }

        if (storePositions.Count == 0)
            return;

        using var scope = NewDisposeScope();

        int actualCount = storePositions.Count;
        int paddedCount = PadToPowerOfTwo(actualCount);
        if (paddedCount > actualCount)
            for (int i = actualCount; i < paddedCount; ++i)
                storePositions.Add(storePositions[0]);

        GameStateTensors stateTensors = BuildStateTensorBatch(storePositions);
        Tensor logits = model.GetStorePolicyLogits(stateTensors);
        Tensor values = model.GetValues(stateTensors);
        Tensor illegalMask = BuildIllegalStoreMask(
            money: stateTensors.Money.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64),
            rerollPrice: stateTensors.RerollPrice.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64),
            storeJokers: stateTensors.StoreJokers.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64),
            storePrices: stateTensors.StorePrices.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64),
            ownedJokers: stateTensors.OwnedJokers.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64));
        Tensor probs = functional.softmax(logits + illegalMask, dim: 1).to(CPU);
        float[] flatProbs = [.. probs.data<float>()];
        float[] flatValues = [.. values.to(CPU).data<float>()];
        int storeMoveCount = (int)probs.size(1);

        for (int batchIndex = 0; batchIndex < actualCount; ++batchIndex)
        {
            int rowOffset = batchIndex * storeMoveCount;
            Span<float> rowSpan = flatProbs.AsSpan(rowOffset, storeMoveCount);
            TrajectoryPosition position = storePositions[batchIndex];
            position.PositionValueEstimate = flatValues[batchIndex];
            int chosenMoveIndex = SampleMoveIndex(rowSpan, random);
            FillStoreTargets(position, chosenMoveIndex, rowSpan);

            int activeSlot = storeDecisionSlots[batchIndex];
            int gameSlot = activeIndices[activeSlot];
            activeTrajectories[gameSlot].Add(position);
            Move move = MoveForStoreIndex(gameStates[gameSlot], chosenMoveIndex);
            move.Apply(gameStates[gameSlot]);
        }
    }


    internal static (GameStateTensors stateTensors, UseHandTensors useHandTensors) BuildRoundRolloutBatch(IReadOnlyList<TrajectoryPosition> positions)
    {
        GameStateTensors stateTensors = BuildStateTensorBatch(positions);
        float[,] useHandScores = new float[positions.Count, PpoPolicyValueModel.UseableHandCount];
        for (int batchIndex = 0; batchIndex < positions.Count; ++batchIndex)
        {
            for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
                useHandScores[batchIndex, handIndex] = positions[batchIndex].UseHandScores[handIndex];
        }

        return (
            stateTensors,
            new()
            {
                Score = tensor(useHandScores, dtype: ScalarType.Float32),
            });
    }


    internal static GameStateTensors BuildStateTensorBatch(IReadOnlyList<TrajectoryPosition> positions)
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
        }

        return new()
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
        };
    }


    static void FillRoundTargets(
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


    static void FillStoreTargets(TrajectoryPosition position, int chosenMoveIndex, ReadOnlySpan<float> fullProbs)
    {
        position.StoreActionIndex = chosenMoveIndex;
        position.OldStoreActionProb = fullProbs[chosenMoveIndex];
        position.PolicyEntropy = GetEntropy(fullProbs);
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


    internal static Tensor BuildIllegalStoreMask(Tensor money, Tensor rerollPrice, Tensor storeJokers, Tensor storePrices, Tensor ownedJokers)
    {
        using var scope = NewDisposeScope();

        Tensor exitMask = zeros([money.size(0), 1], dtype: ScalarType.Float32, device: money.device);
        Tensor rerollMask = money.lt(rerollPrice).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor nullStoreMask = storeJokers.eq(0).to_type(ScalarType.Float32);
        Tensor unaffordableStoreMask = money.unsqueeze(-1).lt(storePrices).to_type(ScalarType.Float32);
        Tensor ownedJokerCount = ownedJokers.ne(0).to_type(ScalarType.Int64).sum(dim: 1, keepdim: true);
        Tensor rosterFullMask = ownedJokerCount.ge(GameStateEmbedder.MaxOwnedJokerCount).to_type(ScalarType.Float32).expand_as(nullStoreMask);
        Tensor storeMask = (nullStoreMask + unaffordableStoreMask + rosterFullMask).clamp(0f, 1f);
        Tensor stacked = cat([exitMask, rerollMask, storeMask], dim: 1) * -1e9f;
        stacked.MoveToOuterDisposeScope();
        return stacked;
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


    static int PadToPowerOfTwo(int value)
    {
        int padded = 1;
        while (padded < value)
            padded <<= 1;
        return padded;
    }


    static void AdvanceToNextDecisionState(GameState gameState)
    {
        while (!gameState.GameIsDone && !IsDecisionStage(gameState.Stage))
        {
            Move[] moves = gameState.GetMoveOptions();
            if (moves.Length != 1)
                throw new InvalidOperationException($"Expected exactly one automatic move while advancing, got {moves.Length} in stage {gameState.Stage}.");

            moves[0].Apply(gameState);
        }
    }


    static bool IsDecisionStage(StageOfGame stage)
    {
        return stage == StageOfGame.InRoundPlayerChoice || stage == StageOfGame.InShop;
    }


    static Move MoveForStoreIndex(GameState state, int index)
    {
        _ = state;
        return index switch
        {
            0 => new ExitShopMove(),
            1 => new RerollMove(),
            2 => new BuyShopOfferMove(0),
            3 => new BuyShopOfferMove(1),
            _ => throw new InvalidOperationException($"Unsupported store move index {index}."),
        };
    }




    static int DivideRoundUp(int count, int batchSize)
    {
        if (count <= 0 || batchSize <= 0)
            return 0;

        return (count + batchSize - 1) / batchSize;
    }


    static (float policyLoss, float valueMse, float clipFraction) TrainRoundBatch(PpoPolicyValueModel model, AdamW optimizer, PpoRoundMiniBatch batch, ExperimentConfig config)
    {
        GameStateTensors stateTensors = batch.StateTensors.ToDevice(PpoPolicyValueModel.EvalDevice);
        UseHandTensors useHandTensors = batch.UseHandTensors.ToDevice(PpoPolicyValueModel.EvalDevice);
        Tensor sampledMoveIndices = batch.SampledMoveIndices.to(PpoPolicyValueModel.EvalDevice);
        Tensor sampledMoveLogQ = batch.SampledMoveLogQ.to(PpoPolicyValueModel.EvalDevice);
        Tensor sampledMoveValidMask = batch.SampledMoveValidMask.to(PpoPolicyValueModel.EvalDevice);
        Tensor oldSampledProbs = batch.OldSampledMoveProbs.to(PpoPolicyValueModel.EvalDevice);
        Tensor valueTargets = batch.ValueTargets.to(PpoPolicyValueModel.EvalDevice);

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
        Tensor advantages = batch.Advantages.to(PpoPolicyValueModel.EvalDevice);
        Tensor clippedRatio = clamp(ratio, max: config.PpoEpsilon);
        Tensor clipMask = ratio.greater(config.PpoEpsilon).to_type(ScalarType.Float32);
        Tensor weight = clippedRatio.detach();
        Tensor policyLoss = -(weight * advantages * logPiNew).mean() - config.EntropyCoefficient * entropy;
        Tensor valueLoss = functional.mse_loss(values, valueTargets);
        Tensor totalLoss = policyLoss + config.ValueLossCoefficient * valueLoss;
        totalLoss.backward();
        return (policyLoss.item<float>(), valueLoss.item<float>(), clipMask.mean().item<float>());
    }


    static (float policyLoss, float valueMse, float clipFraction) TrainStoreBatch(PpoPolicyValueModel model, AdamW optimizer, PpoStoreMiniBatch batch, ExperimentConfig config)
    {
        _ = optimizer;
        GameStateTensors stateTensors = batch.StateTensors.ToDevice(PpoPolicyValueModel.EvalDevice);
        Tensor actionIndices = batch.ActionIndices.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64);
        Tensor oldActionProbs = batch.OldActionProbs.to(PpoPolicyValueModel.EvalDevice);
        Tensor valueTargets = batch.ValueTargets.to(PpoPolicyValueModel.EvalDevice);

        Tensor logits = model.GetStorePolicyLogits(stateTensors);
        Tensor illegalMask = BuildIllegalStoreMask(
            money: stateTensors.Money.to_type(ScalarType.Int64),
            rerollPrice: stateTensors.RerollPrice.to_type(ScalarType.Int64),
            storeJokers: stateTensors.StoreJokers.to_type(ScalarType.Int64),
            storePrices: stateTensors.StorePrices.to_type(ScalarType.Int64),
            ownedJokers: stateTensors.OwnedJokers.to_type(ScalarType.Int64));
        Tensor maskedLogits = logits + illegalMask;
        Tensor logProbs = functional.log_softmax(maskedLogits, dim: 1);
        Tensor probs = exp(logProbs);
        Tensor entropy = -(probs * logProbs).sum(dim: 1).mean();
        Tensor logPiNew = logProbs.gather(1, actionIndices.unsqueeze(-1)).squeeze(-1);
        Tensor logPiOld = oldActionProbs.clamp_min(1e-9f).log();
        Tensor ratio = exp(logPiNew - logPiOld);
        Tensor values = model.GetValues(stateTensors);
        Tensor advantages = batch.Advantages.to(PpoPolicyValueModel.EvalDevice);
        Tensor clippedRatio = clamp(ratio, max: config.PpoEpsilon);
        Tensor clipMask = ratio.greater(config.PpoEpsilon).to_type(ScalarType.Float32);
        Tensor weight = clippedRatio.detach();
        Tensor policyLoss = -(weight * advantages * logPiNew).mean() - config.EntropyCoefficient * entropy;
        Tensor valueLoss = functional.mse_loss(values, valueTargets);
        Tensor totalLoss = policyLoss + config.ValueLossCoefficient * valueLoss;
        totalLoss.backward();
        return (policyLoss.item<float>(), valueLoss.item<float>(), clipMask.mean().item<float>());
    }
}

public sealed class PpoRolloutDataset : IDisposable
{
    public int Capacity { get; }
    public int Count => RoundCount + StoreCount;
    public int RoundCount => Round.Count;
    public int StoreCount => Store.Count;
    public PpoRoundRolloutDataset Round { get; }
    public PpoStoreRolloutDataset Store { get; }

    public float AverageReward { get; private set; }

    public float AverageMoveEntropy { get; private set; }

    public int CompletedGameCount { get; private set; }

    float _advantageTotal;
    float _advantageSquaredTotal;
    int _advantageCount;

    public PpoRolloutDataset(int capacity, int sampledSoftmaxCount)
    {
        Capacity = capacity;
        Round = new(capacity, sampledSoftmaxCount);
        Store = new(capacity);
    }


    public void Dispose()
    {
        Round.Dispose();
        Store.Dispose();
    }


    public void AddSample(TrajectoryPosition position, float valueTarget, float advantage)
    {
        if (position.IsStoreState)
            Store.AddSample(position, valueTarget, advantage);
        else
            Round.AddSample(position, valueTarget, advantage);

        _advantageTotal += advantage;
        _advantageSquaredTotal += advantage * advantage;
        _advantageCount++;
    }


    public void SetMetrics(float averageReward, float averageMoveEntropy, int completedGameCount)
    {
        AverageReward = averageReward;
        AverageMoveEntropy = averageMoveEntropy;
        CompletedGameCount = completedGameCount;
    }


    public void NormalizeSamples()
    {
        if (_advantageCount == 0)
            return;

        float advantageMean = _advantageTotal / _advantageCount;
        float advantageVariance = (_advantageSquaredTotal / _advantageCount) - advantageMean * advantageMean;
        float advantageStdDev = MathF.Sqrt(MathF.Max(0f, advantageVariance));
        float advantageScale = MathF.Max(advantageStdDev, 1e-8f);

        Round.NormalizeSamples(
            advantageMean: advantageMean,
            advantageScale: advantageScale);
        Store.NormalizeSamples(
            advantageMean: advantageMean,
            advantageScale: advantageScale);
    }
}

public sealed class PpoRoundRolloutDataset : IDisposable
{
    readonly List<TrajectoryPosition> _positions;
    readonly List<float> _valueTargets;
    readonly List<float> _advantages;
    readonly int _sampledSoftmaxCount;

    public int Count => _positions.Count;

    public PpoRoundRolloutDataset(int capacity, int sampledSoftmaxCount)
    {
        _positions = new(capacity);
        _valueTargets = new(capacity);
        _advantages = new(capacity);
        _sampledSoftmaxCount = sampledSoftmaxCount;
    }


    public void AddSample(TrajectoryPosition position, float valueTarget, float advantage)
    {
        _positions.Add(position);
        _valueTargets.Add(valueTarget);
        _advantages.Add(advantage);
    }


    public void NormalizeSamples(float advantageMean, float advantageScale)
    {
        for (int index = 0; index < _advantages.Count; ++index)
            _advantages[index] = (_advantages[index] - advantageMean) / advantageScale;
    }


    public PpoRoundMiniBatch FillBatch(int[] shuffledIndices, int batchStart, int batchSize)
    {
        TrajectoryPosition[] batchPositions = new TrajectoryPosition[batchSize];
        float[] valueTargets = new float[batchSize];
        float[] advantages = new float[batchSize];
        long[,] sampledMoveIndices = new long[batchSize, _sampledSoftmaxCount];
        float[,] oldSampledMoveProbs = new float[batchSize, _sampledSoftmaxCount];
        float[,] sampledMoveLogQ = new float[batchSize, _sampledSoftmaxCount];
        float[,] sampledMoveValidMask = new float[batchSize, _sampledSoftmaxCount];

        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            int sampleIndex = shuffledIndices[(batchStart + batchIndex) % shuffledIndices.Length];
            TrajectoryPosition position = _positions[sampleIndex];
            batchPositions[batchIndex] = position;
            valueTargets[batchIndex] = _valueTargets[sampleIndex];
            advantages[batchIndex] = _advantages[sampleIndex];

            for (int sampleMoveIndex = 0; sampleMoveIndex < _sampledSoftmaxCount; ++sampleMoveIndex)
            {
                sampledMoveIndices[batchIndex, sampleMoveIndex] = position.SampledMoveIndices[sampleMoveIndex];
                oldSampledMoveProbs[batchIndex, sampleMoveIndex] = position.OldSampledMoveProbs[sampleMoveIndex];
                sampledMoveLogQ[batchIndex, sampleMoveIndex] = position.SampledMoveLogQ[sampleMoveIndex];
                sampledMoveValidMask[batchIndex, sampleMoveIndex] = position.SampledMoveValidMask[sampleMoveIndex];
            }
        }

        (GameStateTensors stateTensors, UseHandTensors useHandTensors) = PpoTraining.BuildRoundRolloutBatch(batchPositions);
        return new()
        {
            StateTensors = stateTensors,
            UseHandTensors = useHandTensors,
            SampledMoveIndices = tensor(sampledMoveIndices, dtype: ScalarType.Int64),
            OldSampledMoveProbs = tensor(oldSampledMoveProbs, dtype: ScalarType.Float32),
            SampledMoveLogQ = tensor(sampledMoveLogQ, dtype: ScalarType.Float32),
            SampledMoveValidMask = tensor(sampledMoveValidMask, dtype: ScalarType.Float32),
            ValueTargets = tensor(valueTargets, dtype: ScalarType.Float32),
            Advantages = tensor(advantages, dtype: ScalarType.Float32),
        };
    }

    public void Dispose()
    {
    }
}

public sealed class PpoStoreRolloutDataset : IDisposable
{
    readonly List<TrajectoryPosition> _positions;
    readonly List<float> _valueTargets;
    readonly List<float> _advantages;

    public int Count => _positions.Count;

    public PpoStoreRolloutDataset(int capacity)
    {
        _positions = new(capacity);
        _valueTargets = new(capacity);
        _advantages = new(capacity);
    }

    public void AddSample(TrajectoryPosition position, float valueTarget, float advantage)
    {
        _positions.Add(position);
        _valueTargets.Add(valueTarget);
        _advantages.Add(advantage);
    }


    public void NormalizeSamples(float advantageMean, float advantageScale)
    {
        for (int index = 0; index < _advantages.Count; ++index)
            _advantages[index] = (_advantages[index] - advantageMean) / advantageScale;
    }

    public PpoStoreMiniBatch FillBatch(int[] shuffledIndices, int batchStart, int batchSize)
    {
        TrajectoryPosition[] batchPositions = new TrajectoryPosition[batchSize];
        long[] actionIndices = new long[batchSize];
        float[] oldActionProbs = new float[batchSize];
        float[] valueTargets = new float[batchSize];
        float[] advantages = new float[batchSize];

        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            int sampleIndex = shuffledIndices[(batchStart + batchIndex) % shuffledIndices.Length];
            TrajectoryPosition position = _positions[sampleIndex];
            batchPositions[batchIndex] = position;
            actionIndices[batchIndex] = position.StoreActionIndex;
            oldActionProbs[batchIndex] = position.OldStoreActionProb;
            valueTargets[batchIndex] = _valueTargets[sampleIndex];
            advantages[batchIndex] = _advantages[sampleIndex];
        }

        return new()
        {
            StateTensors = PpoTraining.BuildStateTensorBatch(batchPositions),
            ActionIndices = tensor(actionIndices, dtype: ScalarType.Int64),
            OldActionProbs = tensor(oldActionProbs, dtype: ScalarType.Float32),
            ValueTargets = tensor(valueTargets, dtype: ScalarType.Float32),
            Advantages = tensor(advantages, dtype: ScalarType.Float32),
        };
    }

    public void Dispose()
    {
    }
}

public sealed class PpoRoundMiniBatch : Ramen.AgentTools.ITensorGroup, IDisposable
{
    public GameStateTensors StateTensors;
    public UseHandTensors UseHandTensors;
    public Tensor SampledMoveIndices;
    public Tensor OldSampledMoveProbs;
    public Tensor SampledMoveLogQ;
    public Tensor SampledMoveValidMask;
    public Tensor ValueTargets;
    public Tensor Advantages;

    public void Dispose()
    {
        Ramen.AgentTools.TensorGroupExtentions.Dispose((Ramen.AgentTools.ITensorGroup)this);
    }
}

public sealed class PpoStoreMiniBatch : Ramen.AgentTools.ITensorGroup, IDisposable
{
    public GameStateTensors StateTensors;
    public Tensor ActionIndices;
    public Tensor OldActionProbs;
    public Tensor ValueTargets;
    public Tensor Advantages;

    public void Dispose()
    {
        Ramen.AgentTools.TensorGroupExtentions.Dispose((Ramen.AgentTools.ITensorGroup)this);
    }
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
    public readonly bool IsStoreState;
    public readonly float Score;
    public long StoreActionIndex;
    public float OldStoreActionProb;
    public float PositionValueEstimate;
    public float PolicyEntropy;

    public TrajectoryPosition(GameState gameState, int sampledSoftmaxCount)
    {
        ReadOnlySpan<Card> hand = gameState.HandState.Hand;
        for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
            FullHand[cardIndex] = cardIndex < hand.Length ? hand[cardIndex].ToIndex() : 0;

        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
            RemainingDeck[cardIndex] = cardIndex < deck.Length ? deck[cardIndex].ToIndex() : 0;

        RemainingHands = gameState.HandState.RemainingHands;
        RemainingDiscards = gameState.HandState.RemainingDiscards;
        RerollPrice = gameState.ShopState.CurrentRerollCost;
        Money = gameState.ShopState.Money;
        Round = gameState.Round;
        Stage = IsStoreStage(gameState.Stage) ? 1 : 0;
        IsStoreState = IsStoreStage(gameState.Stage);
        Score = (float)gameState.ScoringState.CurrentRoundTotalScore / 300f;

        WriteJokerSlots(gameState.JokerState.Jokers, gameState.GameData, OwnedJokers);
        WriteJokerSlots(gameState.ShopState.ShopOfferings, gameState.GameData, StoreJokers);
        WriteStorePrices(gameState.ShopState.ShopOfferings, StorePrices);
        if (!IsStoreState)
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
