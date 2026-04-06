namespace Ramen.ConsoleApp;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp.Modules;
using static TorchSharp.torch;

public static class Program
{
    static readonly ITrainingRunAnalyzer[] Analyzers =
    [
        new RewardStatsTrainingRunAnalyzer(),
        new PolicyEntropyTrainingRunAnalyzer(),
        new HandTypePresenceTrainingRunAnalyzer(),
        new EndStateHandCountTrainingRunAnalyzer(),
    ];

    static readonly ITrainingRunAnalyzer[] RewardAndEntropyAnalyzers =
    [
        new RewardStatsTrainingRunAnalyzer(),
        new PolicyEntropyTrainingRunAnalyzer(),
    ];

    public static void Main()
    {
        // Do not change START
        set_default_device(mps_is_available() ? MPS : CPU);
        Ramen.AI.TensorManager.Init();
        Console.WriteLine("=== START ===");
        // Do not change END

        string experimentName = "2026-04-05_pair_preference_same_step_saved_snapshots_r200_t0";
        string sourcePairWeightsPath = Path.Combine(
            FindRepoRoot(),
            "Analysis",
            "2026-04-03_pair_agent_distill_from_value_network_g10000_e10_r200_t0p01_c10",
            "weights",
            "latest.bin");
        string iter5WeightsPath = Path.Combine(
            FindRepoRoot(),
            "Analysis",
            "2026-04-03_pair_preference_same_step_g500_p50000_i10_top4p90_a200",
            "weights",
            "iter5.bin");
        string iter10WeightsPath = Path.Combine(
            FindRepoRoot(),
            "Analysis",
            "2026-04-03_pair_preference_same_step_g500_p50000_i10_top4p90_a200",
            "weights",
            "iter10.bin");
        int gameCount = 200;
        float temp = 0f;

        string repoRoot = FindRepoRoot();
        string analysisDir = Path.Combine(repoRoot, "Analysis", experimentName);
        Directory.CreateDirectory(analysisDir);
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");

        RunSavedCheckpointEvaluation(
            experimentName: experimentName,
            checkpointLabels: [0, 5, 10],
            checkpointWeightsPaths:
            [
                sourcePairWeightsPath,
                iter5WeightsPath,
                iter10WeightsPath,
            ],
            temp: temp,
            gameCount: gameCount,
            analysisCsvPath: analysisCsvPath);

        Console.WriteLine($"Experiment directory ready: {analysisDir}");
    }


    static void RunIterativeLatestOnlyTraining(
        string experimentName,
        int rolloutIterationCount,
        int rolloutGameCount,
        int trainingPairCount,
        int trainingPairIncrement,
        int trainingBatchSize,
        int trainingPairMicroBatchSize,
        float learningRate,
        float gameplaySamplingTemp,
        int analysisSampleSize,
        int weightSnapshotFrequency,
        string analysisCsvPath,
        string weightsRootDir)
    {
        CSVBuilder analysisOutput = new();
        Stopwatch experimentStopwatch = Stopwatch.StartNew();

        using PreferenceTrainingPipeline trainingPipeline = new(
            learningRate: learningRate,
            gameplaySamplingTemp: gameplaySamplingTemp);

        for (int rolloutIteration = 1; rolloutIteration <= rolloutIterationCount; ++rolloutIteration)
        {
            string iterationName = $"{experimentName}_iter{rolloutIteration}";
            string iterationWeightsDir = Path.Combine(weightsRootDir, $"iter{rolloutIteration}");
            Directory.CreateDirectory(iterationWeightsDir);

            Stopwatch iterationStopwatch = Stopwatch.StartNew();

            Stopwatch rolloutStopwatch = Stopwatch.StartNew();
            PreferenceGameRecord[] iterationGames = trainingPipeline.PlayTrainingGames(gameCount: rolloutGameCount);
            rolloutStopwatch.Stop();

            float rolloutRewardMean = GetMeanReward(iterationGames);
            float rolloutRewardStdDev = GetRewardStdDev(iterationGames);
            float rolloutGenerationSeconds = GetElapsedSeconds(rolloutStopwatch);
            float pairTrainingCumulativeSeconds = 0f;
            float analysisCumulativeSeconds = 0f;

            Console.WriteLine(
                $"[{iterationName}] Generated {rolloutGameCount} games in {rolloutGenerationSeconds:F2}s.");

            for (int trainedPairs = trainingPairIncrement; trainedPairs <= trainingPairCount; trainedPairs += trainingPairIncrement)
            {
                int[] pairAllocation = AllocatePairsToLatestDataset(
                    totalPairCount: trainingPairIncrement,
                    datasetCount: rolloutIterationCount,
                    latestIterationIndex: rolloutIteration);

                Stopwatch pairTrainingStopwatch = Stopwatch.StartNew();
                PreferenceTrainingMetrics metrics = trainingPipeline.TrainOnRandomPairs(
                    gameRecords: iterationGames,
                    pairCount: trainingPairIncrement,
                    batchSize: trainingBatchSize);
                pairTrainingStopwatch.Stop();
                pairTrainingCumulativeSeconds += GetElapsedSeconds(pairTrainingStopwatch);

                Stopwatch analysisStopwatch = Stopwatch.StartNew();
                GameState[] analysisGames = trainingPipeline.PlayAnalysisGames(gameCount: analysisSampleSize);
                analysisStopwatch.Stop();
                analysisCumulativeSeconds += GetElapsedSeconds(analysisStopwatch);

                int analysisPoint = trainedPairs / trainingPairIncrement;
                analysisOutput.NextRow()
                    .SetCell("experiment", experimentName)
                    .SetCell("iteration_name", iterationName)
                    .SetCell("rollout_iteration", rolloutIteration)
                    .SetCell("rollout_iteration_count", rolloutIterationCount)
                    .SetCell("replay_dataset_count", 1)
                    .SetCell("rollout_game_count", rolloutGameCount)
                    .SetCell("training_pair_limit", trainingPairCount)
                    .SetCell("training_pair_increment", trainingPairIncrement)
                    .SetCell("training_batch_size", trainingBatchSize)
                    .SetCell("training_pair_micro_batch_size", trainingPairMicroBatchSize)
                    .SetCell("learning_rate", learningRate)
                    .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
                    .SetCell("analysis_game_count", analysisSampleSize)
                    .SetCell("analysis_point", analysisPoint)
                    .SetCell("training_progress_frac", (float)trainedPairs / trainingPairCount)
                    .SetCell("trained_pairs", trainedPairs)
                    .SetCell("pair_batch_loss", metrics.MeanLoss)
                    .SetCell("rollout_reward_mean", rolloutRewardMean)
                    .SetCell("rollout_reward_stddev", rolloutRewardStdDev)
                    .SetCell("rollout_generation_seconds", rolloutGenerationSeconds)
                    .SetCell("pair_training_step_seconds", GetElapsedSeconds(pairTrainingStopwatch))
                    .SetCell("pair_training_cumulative_seconds", pairTrainingCumulativeSeconds)
                    .SetCell("analysis_step_seconds", GetElapsedSeconds(analysisStopwatch))
                    .SetCell("analysis_cumulative_seconds", analysisCumulativeSeconds)
                    .SetCell("iteration_elapsed_seconds", GetElapsedSeconds(iterationStopwatch))
                    .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch))
                    .SetCell("pair_allocation_summary", GetPairAllocationSummary(pairAllocation));

                for (int allocationIteration = 1; allocationIteration <= rolloutIterationCount; ++allocationIteration)
                {
                    analysisOutput.SetCell(
                        $"pairs_from_iter{allocationIteration}",
                        GetAllocatedPairs(pairAllocation, iterationIndex: allocationIteration));
                }

                AnalyzeGames(analysisGames, analysisOutput);

                File.WriteAllText(analysisCsvPath, analysisOutput.ToString());

                if (analysisPoint % weightSnapshotFrequency == 0)
                    trainingPipeline.Save(Path.Combine(iterationWeightsDir, $"{trainedPairs}.bin"));

                Console.WriteLine(
                    $"[{iterationName}] Trained {trainedPairs}/{trainingPairCount} pairs " +
                    $"({analysisPoint} analysis points). " +
                    $"Alloc {GetPairAllocationSummary(pairAllocation)}. " +
                    $"Train step {GetElapsedSeconds(pairTrainingStopwatch):F2}s, " +
                    $"analysis {GetElapsedSeconds(analysisStopwatch):F2}s.");
            }

            trainingPipeline.Save(Path.Combine(iterationWeightsDir, "latest.bin"));
            Console.WriteLine(
                $"[{iterationName}] Completed in {GetElapsedSeconds(iterationStopwatch):F2}s " +
                $"with rollout generation {rolloutGenerationSeconds:F2}s and pair training total {pairTrainingCumulativeSeconds:F2}s.");
        }
    }


    static int[] AllocatePairsToLatestDataset(int totalPairCount, int datasetCount, int latestIterationIndex)
    {
        int[] pairAllocation = new int[datasetCount];
        if (latestIterationIndex <= 0 || latestIterationIndex > datasetCount)
            return pairAllocation;

        pairAllocation[latestIterationIndex - 1] = totalPairCount;
        return pairAllocation;
    }


    static void AnalyzeGames(IEnumerable<GameState> analysisGames, CSVBuilder analysisOutput)
    {
        AnalyzeGames(
            analysisGames: analysisGames,
            analysisOutput: analysisOutput,
            analyzers: Analyzers);
    }


    static void AnalyzeGames(IEnumerable<GameState> analysisGames, CSVBuilder analysisOutput, ITrainingRunAnalyzer[] analyzers)
    {
        for (int analyzerIndex = 0; analyzerIndex < analyzers.Length; ++analyzerIndex)
            analyzers[analyzerIndex].Analyze(analysisGames, analysisOutput);
    }


    static int[] AllocatePairsEvenly(int totalPairCount, int datasetCount)
    {
        int[] pairAllocation = new int[datasetCount];
        if (datasetCount <= 0)
            return pairAllocation;

        int baseAllocation = totalPairCount / datasetCount;
        int remainder = totalPairCount % datasetCount;
        for (int datasetIndex = 0; datasetIndex < datasetCount; ++datasetIndex)
            pairAllocation[datasetIndex] = baseAllocation + (datasetIndex < remainder ? 1 : 0);

        return pairAllocation;
    }


    static int GetAllocatedPairs(int[] pairAllocation, int iterationIndex)
    {
        if (iterationIndex <= 0 || iterationIndex > pairAllocation.Length)
            return 0;

        return pairAllocation[iterationIndex - 1];
    }


    static string GetPairAllocationSummary(int[] pairAllocation)
    {
        if (pairAllocation.Length == 0)
            return string.Empty;

        string[] parts = new string[pairAllocation.Length];
        for (int datasetIndex = 0; datasetIndex < pairAllocation.Length; ++datasetIndex)
            parts[datasetIndex] = $"i{datasetIndex + 1}:{pairAllocation[datasetIndex]}";

        return string.Join(" | ", parts);
    }


    static string FindRepoRoot()
    {
        string currentPath = AppContext.BaseDirectory;
        DirectoryInfo currentDirectory = new(currentPath);

        while (currentDirectory != null)
        {
            string analysisPath = Path.Combine(currentDirectory.FullName, "Analysis");
            if (Directory.Exists(analysisPath))
                return currentDirectory.FullName;

            currentDirectory = currentDirectory.Parent;
        }

        throw new InvalidOperationException("Could not find repository root containing Analysis/.");
    }


    static float GetElapsedSeconds(Stopwatch stopwatch)
    {
        return (float)stopwatch.Elapsed.TotalSeconds;
    }


    static void RunPairPreferenceSameStepExperiment(
        string experimentName,
        string sourcePairWeightsPath,
        int rolloutIterationCount,
        int rolloutGameCount,
        int trainingPairCount,
        int trainingBatchSize,
        int analysisGameCount,
        float learningRate,
        int gameplayTopMoveCount,
        float gameplayTopMoveProbabilityMassTarget,
        float fallbackSamplingTemp,
        int weightSnapshotFrequency,
        string analysisCsvPath,
        string weightsRootDir)
    {
        CSVBuilder analysisOutput = new();
        Stopwatch experimentStopwatch = Stopwatch.StartNew();
        FastRandom random = FastRandom.SeededByClock();

        using PreferenceValueModel model = new();
        model.Load(sourcePairWeightsPath);

        AdamW optimizer = optim.AdamW(
            parameters: model.parameters(),
            lr: learningRate,
            weight_decay: 0f,
            beta1: 0.9f,
            beta2: 0.99f);

        try
        {
            AppendPairPreferenceAnalysisRow(
                analysisOutput: analysisOutput,
                experimentName: experimentName,
                sourcePairWeightsPath: sourcePairWeightsPath,
                rolloutIterationCount: rolloutIterationCount,
                rolloutGameCount: rolloutGameCount,
                trainingPairCount: trainingPairCount,
                trainingBatchSize: trainingBatchSize,
                analysisGameCount: analysisGameCount,
                learningRate: learningRate,
                gameplayTopMoveCount: gameplayTopMoveCount,
                gameplayTopMoveProbabilityMassTarget: gameplayTopMoveProbabilityMassTarget,
                fallbackSamplingTemp: fallbackSamplingTemp,
                weightSnapshotFrequency: weightSnapshotFrequency,
                rolloutIteration: 0,
                totalTrainedPairs: 0,
                pairLossMean: 0f,
                rolloutRewardMean: 0f,
                rolloutRewardStdDev: 0f,
                rolloutGenerationSeconds: 0f,
                pairGenerationSeconds: 0f,
                trainingStepSeconds: 0f,
                availableOrderedPairCount: 0,
                eligibleStepCount: 0,
                model: model,
                experimentStopwatch: experimentStopwatch);
            File.WriteAllText(analysisCsvPath, analysisOutput.ToString());

            for (int rolloutIteration = 1; rolloutIteration <= rolloutIterationCount; ++rolloutIteration)
            {
                using PreferenceSamplingAgent rolloutAgent = new(model, ownsModel: false);
                rolloutAgent.TopMoveProbabilityCount = gameplayTopMoveCount;
                rolloutAgent.TopMoveProbabilityMassTarget = gameplayTopMoveProbabilityMassTarget;

                Stopwatch rolloutStopwatch = Stopwatch.StartNew();
                GameState[] rolloutGames = PlayGames(
                    agent: rolloutAgent,
                    gameCount: rolloutGameCount,
                    temp: fallbackSamplingTemp,
                    annotatePolicy: false);
                rolloutStopwatch.Stop();

                PreferenceGameRecord[] gameRecords = BuildPreferenceGameRecords(rolloutGames);
                float rolloutRewardMean = GetMeanReward(gameRecords);
                float rolloutRewardStdDev = GetRewardStdDev(gameRecords);

                Stopwatch pairGenerationStopwatch = Stopwatch.StartNew();
                StepMatchedPairDataset pairDataset = GenerateStepMatchedPairDataset(
                    gameRecords: gameRecords,
                    pairCount: trainingPairCount,
                    random: random);
                pairGenerationStopwatch.Stop();

                Stopwatch trainingStopwatch = Stopwatch.StartNew();
                PreferenceTrainingMetrics metrics = TrainOnStepMatchedPairs(
                    gameRecords: gameRecords,
                    pairDataset: pairDataset,
                    model: model,
                    optimizer: optimizer,
                    batchSize: trainingBatchSize,
                    random: random);
                trainingStopwatch.Stop();

                int totalTrainedPairs = rolloutIteration * trainingPairCount;
                AppendPairPreferenceAnalysisRow(
                    analysisOutput: analysisOutput,
                    experimentName: experimentName,
                    sourcePairWeightsPath: sourcePairWeightsPath,
                    rolloutIterationCount: rolloutIterationCount,
                    rolloutGameCount: rolloutGameCount,
                    trainingPairCount: trainingPairCount,
                    trainingBatchSize: trainingBatchSize,
                    analysisGameCount: analysisGameCount,
                    learningRate: learningRate,
                    gameplayTopMoveCount: gameplayTopMoveCount,
                    gameplayTopMoveProbabilityMassTarget: gameplayTopMoveProbabilityMassTarget,
                    fallbackSamplingTemp: fallbackSamplingTemp,
                    weightSnapshotFrequency: weightSnapshotFrequency,
                    rolloutIteration: rolloutIteration,
                    totalTrainedPairs: totalTrainedPairs,
                    pairLossMean: metrics.MeanLoss,
                    rolloutRewardMean: rolloutRewardMean,
                    rolloutRewardStdDev: rolloutRewardStdDev,
                    rolloutGenerationSeconds: GetElapsedSeconds(rolloutStopwatch),
                    pairGenerationSeconds: GetElapsedSeconds(pairGenerationStopwatch),
                    trainingStepSeconds: GetElapsedSeconds(trainingStopwatch),
                    availableOrderedPairCount: pairDataset.AvailableOrderedPairCount,
                    eligibleStepCount: pairDataset.EligibleStepCount,
                    model: model,
                    experimentStopwatch: experimentStopwatch);

                File.WriteAllText(analysisCsvPath, analysisOutput.ToString());

                if (rolloutIteration % weightSnapshotFrequency == 0)
                    model.Save(Path.Combine(weightsRootDir, $"iter{rolloutIteration}.bin"));

                Console.WriteLine(
                    $"[{experimentName}] Iteration {rolloutIteration}/{rolloutIterationCount} generated {rolloutGameCount} games, " +
                    $"sampled {pairDataset.Pairs.Length} same-step pairs across {pairDataset.EligibleStepCount} steps, " +
                    $"and trained with mean loss {metrics.MeanLoss:F6}.");
            }

            model.Save(Path.Combine(weightsRootDir, "latest.bin"));
            File.WriteAllText(analysisCsvPath, analysisOutput.ToString());
        }
        finally
        {
            optimizer.Dispose();
        }
    }


    static void AppendPairPreferenceAnalysisRow(
        CSVBuilder analysisOutput,
        string experimentName,
        string sourcePairWeightsPath,
        int rolloutIterationCount,
        int rolloutGameCount,
        int trainingPairCount,
        int trainingBatchSize,
        int analysisGameCount,
        float learningRate,
        int gameplayTopMoveCount,
        float gameplayTopMoveProbabilityMassTarget,
        float fallbackSamplingTemp,
        int weightSnapshotFrequency,
        int rolloutIteration,
        int totalTrainedPairs,
        float pairLossMean,
        float rolloutRewardMean,
        float rolloutRewardStdDev,
        float rolloutGenerationSeconds,
        float pairGenerationSeconds,
        float trainingStepSeconds,
        int availableOrderedPairCount,
        int eligibleStepCount,
        PreferenceValueModel model,
        Stopwatch experimentStopwatch)
    {
        using PreferenceSamplingAgent analysisAgent = new(model, ownsModel: false);
        analysisAgent.TopMoveProbabilityCount = gameplayTopMoveCount;
        analysisAgent.TopMoveProbabilityMassTarget = gameplayTopMoveProbabilityMassTarget;

        Stopwatch analysisStopwatch = Stopwatch.StartNew();
        GameState[] analysisGames = PlayGames(
            agent: analysisAgent,
            gameCount: analysisGameCount,
            temp: fallbackSamplingTemp,
            annotatePolicy: true);
        analysisStopwatch.Stop();

        analysisOutput.NextRow()
            .SetCell("row_type", "analysis")
            .SetCell("experiment", experimentName)
            .SetCell("source_pair_weights_path", sourcePairWeightsPath)
            .SetCell("rollout_iteration", rolloutIteration)
            .SetCell("rollout_iteration_count", rolloutIterationCount)
            .SetCell("rollout_game_count", rolloutGameCount)
            .SetCell("training_pair_count", trainingPairCount)
            .SetCell("training_batch_size", trainingBatchSize)
            .SetCell("analysis_game_count", analysisGameCount)
            .SetCell("learning_rate", learningRate)
            .SetCell("sampling_mode", "target_top_probability_mass")
            .SetCell("gameplay_top_move_count", gameplayTopMoveCount)
            .SetCell("gameplay_top_move_probability_mass_target", gameplayTopMoveProbabilityMassTarget)
            .SetCell("fallback_sampling_temp", fallbackSamplingTemp)
            .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
            .SetCell("total_trained_pairs", totalTrainedPairs)
            .SetCell("pair_loss_mean", pairLossMean)
            .SetCell("rollout_reward_mean", rolloutRewardMean)
            .SetCell("rollout_reward_stddev", rolloutRewardStdDev)
            .SetCell("rollout_generation_seconds", rolloutGenerationSeconds)
            .SetCell("pair_generation_seconds", pairGenerationSeconds)
            .SetCell("training_step_seconds", trainingStepSeconds)
            .SetCell("analysis_step_seconds", GetElapsedSeconds(analysisStopwatch))
            .SetCell("available_ordered_pair_count", availableOrderedPairCount)
            .SetCell("eligible_step_count", eligibleStepCount)
            .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch));

        AnalyzeGames(
            analysisGames: analysisGames,
            analysisOutput: analysisOutput,
            analyzers: RewardAndEntropyAnalyzers);
    }


    static PreferenceGameRecord[] BuildPreferenceGameRecords(ReadOnlySpan<GameState> games)
    {
        PreferenceGameRecord[] gameRecords = new PreferenceGameRecord[games.Length];
        for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
        {
            GameState gameState = games[gameIndex];
            using MemoryStream stream = new();
            gameState.Serialize(stream);
            gameRecords[gameIndex] = new(
                serializedGame: stream.ToArray(),
                finalReward: GetReward(gameState),
                moveCount: gameState.MoveState.MoveHistory.Count);
        }

        return gameRecords;
    }


    static StepMatchedPairDataset GenerateStepMatchedPairDataset(
        IReadOnlyList<PreferenceGameRecord> gameRecords,
        int pairCount,
        FastRandom random)
    {
        StepBucket[] stepBuckets = BuildStepBuckets(gameRecords);
        int availableOrderedPairCount = 0;
        int eligibleStepCount = 0;
        for (int bucketIndex = 0; bucketIndex < stepBuckets.Length; ++bucketIndex)
        {
            int orderedPairCount = stepBuckets[bucketIndex].OrderedPairCount;
            if (orderedPairCount <= 0)
                continue;

            availableOrderedPairCount += orderedPairCount;
            eligibleStepCount++;
        }

        if (availableOrderedPairCount <= 0 || pairCount <= 0)
            return new(Pairs: [], AvailableOrderedPairCount: 0, EligibleStepCount: eligibleStepCount);

        int[] cumulativePairCounts = new int[stepBuckets.Length];
        int runningPairCount = 0;
        for (int bucketIndex = 0; bucketIndex < stepBuckets.Length; ++bucketIndex)
        {
            runningPairCount += stepBuckets[bucketIndex].OrderedPairCount;
            cumulativePairCounts[bucketIndex] = runningPairCount;
        }

        StepMatchedGamePair[] pairs = new StepMatchedGamePair[pairCount];
        for (int pairIndex = 0; pairIndex < pairCount; ++pairIndex)
            pairs[pairIndex] = SampleStepMatchedPair(gameRecords, stepBuckets, cumulativePairCounts, availableOrderedPairCount, random);

        return new(
            Pairs: pairs,
            AvailableOrderedPairCount: availableOrderedPairCount,
            EligibleStepCount: eligibleStepCount);
    }


    static StepBucket[] BuildStepBuckets(IReadOnlyList<PreferenceGameRecord> gameRecords)
    {
        int maxPositionCount = 0;
        for (int gameIndex = 0; gameIndex < gameRecords.Count; ++gameIndex)
            maxPositionCount = Math.Max(maxPositionCount, gameRecords[gameIndex].PositionCount);

        List<int>[] gameIndicesByStep = new List<int>[maxPositionCount];
        for (int moveStep = 0; moveStep < maxPositionCount; ++moveStep)
            gameIndicesByStep[moveStep] = [];

        for (int gameIndex = 0; gameIndex < gameRecords.Count; ++gameIndex)
        {
            PreferenceGameRecord gameRecord = gameRecords[gameIndex];
            for (int moveStep = 0; moveStep < gameRecord.PositionCount; ++moveStep)
                gameIndicesByStep[moveStep].Add(gameIndex);
        }

        StepBucket[] stepBuckets = new StepBucket[maxPositionCount];
        for (int moveStep = 0; moveStep < maxPositionCount; ++moveStep)
        {
            int[] gameIndices = [.. gameIndicesByStep[moveStep]];
            int orderedPairCount = gameIndices.Length < 2 ? 0 : gameIndices.Length * (gameIndices.Length - 1);
            stepBuckets[moveStep] = new(
                MoveStep: moveStep,
                GameIndices: gameIndices,
                OrderedPairCount: orderedPairCount);
        }

        return stepBuckets;
    }


    static StepMatchedGamePair SampleStepMatchedPair(
        IReadOnlyList<PreferenceGameRecord> gameRecords,
        StepBucket[] stepBuckets,
        int[] cumulativePairCounts,
        int availableOrderedPairCount,
        FastRandom random)
    {
        int sampledPairIndex = random.Next(availableOrderedPairCount) + 1;
        int bucketIndex = Array.BinarySearch(cumulativePairCounts, sampledPairIndex);
        if (bucketIndex < 0)
            bucketIndex = ~bucketIndex;

        StepBucket stepBucket = stepBuckets[bucketIndex];
        int leftBucketIndex = random.Next(stepBucket.GameIndices.Length);
        int rightBucketIndex = random.Next(stepBucket.GameIndices.Length - 1);
        if (rightBucketIndex >= leftBucketIndex)
            rightBucketIndex++;

        int leftGameIndex = stepBucket.GameIndices[leftBucketIndex];
        int rightGameIndex = stepBucket.GameIndices[rightBucketIndex];
        float target = GetPairTrainingTarget(
            leftReward: gameRecords[leftGameIndex].FinalReward,
            rightReward: gameRecords[rightGameIndex].FinalReward);

        return new(
            LeftGameIndex: leftGameIndex,
            RightGameIndex: rightGameIndex,
            MoveStep: stepBucket.MoveStep,
            Target: target);
    }


    static PreferenceTrainingMetrics TrainOnStepMatchedPairs(
        IReadOnlyList<PreferenceGameRecord> gameRecords,
        StepMatchedPairDataset pairDataset,
        PreferenceValueModel model,
        AdamW optimizer,
        int batchSize,
        FastRandom random)
    {
        if (pairDataset.Pairs.Length == 0)
            return new(MeanLoss: 0f, TrainedPairs: 0);

        StepMatchedGamePair[] shuffledPairs = [.. pairDataset.Pairs];
        ShufflePairs(shuffledPairs, random);

        int effectiveBatchSize = Math.Max(batchSize, 1);
        float totalLoss = 0f;
        int batchCount = 0;

        for (int batchStart = 0; batchStart < shuffledPairs.Length; batchStart += effectiveBatchSize)
        {
            int currentBatchSize = Math.Min(effectiveBatchSize, shuffledPairs.Length - batchStart);
            using var scope = NewDisposeScope();
            GameStateEmbedder leftGameStateEmbedder = new(currentBatchSize);
            GameStateEmbedder rightGameStateEmbedder = new(currentBatchSize);
            float[] targets = new float[currentBatchSize];

            for (int batchIndex = 0; batchIndex < currentBatchSize; ++batchIndex)
            {
                StepMatchedGamePair pair = shuffledPairs[batchStart + batchIndex];
                GameState leftState = MaterializePreferenceState(gameRecords[pair.LeftGameIndex], pair.MoveStep);
                GameState rightState = MaterializePreferenceState(gameRecords[pair.RightGameIndex], pair.MoveStep);
                leftGameStateEmbedder.AddGameState(leftState);
                rightGameStateEmbedder.AddGameState(rightState);
                targets[batchIndex] = pair.Target;
            }

            GameStateTensors leftTensors = leftGameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
            GameStateTensors rightTensors = rightGameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
            Tensor targetsTensor = tensor(targets, dtype: ScalarType.Float32, device: PreferenceValueModel.EvalDevice);

            optimizer.zero_grad();

            Tensor leftLogits = model.GetLogits(leftTensors);
            Tensor rightLogits = model.GetLogits(rightTensors);
            Tensor loss = CalculateBradleyTerryLoss(leftLogits, rightLogits, targetsTensor);
            loss.backward();
            optimizer.step();

            totalLoss += loss.item<float>();
            batchCount++;
        }

        return new(
            MeanLoss: totalLoss / Math.Max(1, batchCount),
            TrainedPairs: shuffledPairs.Length);
    }


    static void ShufflePairs(Span<StepMatchedGamePair> pairs, FastRandom random)
    {
        for (int pairIndex = pairs.Length - 1; pairIndex > 0; --pairIndex)
        {
            int swapIndex = random.Next(pairIndex + 1);
            (pairs[pairIndex], pairs[swapIndex]) = (pairs[swapIndex], pairs[pairIndex]);
        }
    }


    static GameState MaterializePreferenceState(PreferenceGameRecord gameRecord, int moveStep)
    {
        GameState gameState = new(GameData.Default);
        using MemoryStream stream = new(gameRecord.SerializedGame, writable: false);
        gameState.Deserialize(stream);
        gameState.MoveState.RevertToStep(moveStep);
        return gameState;
    }


    static Tensor CalculateBradleyTerryLoss(Tensor leftLogits, Tensor rightLogits, Tensor targets)
    {
        Tensor pairLogits = leftLogits - rightLogits;
        Tensor logProbLeft = TorchSharp.torch.nn.functional.logsigmoid(pairLogits);
        Tensor logProbRight = TorchSharp.torch.nn.functional.logsigmoid(-pairLogits);
        return -(targets * logProbLeft + (1f - targets) * logProbRight).mean();
    }


    static float GetPairTrainingTarget(float leftReward, float rightReward)
    {
        if (leftReward > rightReward)
            return 1f;
        if (leftReward < rightReward)
            return 0f;
        return 0.5f;
    }


    static void RunValueNetworkOutcomeExperiment(
        string experimentName,
        string sourceWeightsPath,
        int rolloutGameCount,
        int epochCount,
        int trainingBatchSize,
        float learningRate,
        float gameplaySamplingTemp,
        int calibrationBucketCount,
        int weightSnapshotFrequency,
        string analysisCsvPath,
        string weightsRootDir)
    {
        CSVBuilder analysisOutput = new();
        Stopwatch experimentStopwatch = Stopwatch.StartNew();

        using PreferenceValueModel sourceModel = new();
        sourceModel.Load(sourceWeightsPath);
        using PreferenceSamplingAgent sourceAgent = new(sourceModel, ownsModel: false);

        Stopwatch rolloutStopwatch = Stopwatch.StartNew();
        GameState[] rolloutGames = PlayGames(
            agent: sourceAgent,
            gameCount: rolloutGameCount,
            temp: gameplaySamplingTemp,
            annotatePolicy: false);
        rolloutStopwatch.Stop();

        ValueExperimentGameRecord[] gameRecords = BuildValueExperimentGameRecords(rolloutGames);
        string gameDatabaseName = $"{experimentName}_games";
        GameDatabase gameDatabase = new(gameDatabaseName, load: false, delete: true);
        for (int gameIndex = 0; gameIndex < rolloutGames.Length; ++gameIndex)
            gameDatabase.AddGame(rolloutGames[gameIndex]);

        using ValueNetwork model = new();
        using ValueNetworkTrainingPipeline trainingPipeline = new(
            learningRate: learningRate,
            model: model,
            ownsModel: false);

        AppendCalibrationRows(
            analysisOutput: analysisOutput,
            gameRecords: gameRecords,
            model: model,
            experimentName: experimentName,
            sourceWeightsPath: sourceWeightsPath,
            rolloutGameCount: rolloutGameCount,
            epochCount: epochCount,
            trainingBatchSize: trainingBatchSize,
            learningRate: learningRate,
            gameplaySamplingTemp: gameplaySamplingTemp,
            calibrationBucketCount: calibrationBucketCount,
            weightSnapshotFrequency: weightSnapshotFrequency,
            epoch: 0,
            experimentStopwatch: experimentStopwatch,
            rolloutStopwatch: rolloutStopwatch);
        File.WriteAllText(analysisCsvPath, analysisOutput.ToString());

        for (int epoch = 1; epoch <= epochCount; ++epoch)
        {
            Stopwatch trainingStopwatch = Stopwatch.StartNew();
            ValueNetworkTrainingMetrics metrics = trainingPipeline.TrainOnAllStates(
                gameDatabase: gameDatabase,
                epochCount: 1,
                batchSize: trainingBatchSize);
            trainingStopwatch.Stop();

            analysisOutput.NextRow()
                .SetCell("row_type", "loss")
                .SetCell("experiment", experimentName)
                .SetCell("source_weights_path", sourceWeightsPath)
                .SetCell("rollout_game_count", rolloutGameCount)
                .SetCell("epoch_count", epochCount)
                .SetCell("training_batch_size", trainingBatchSize)
                .SetCell("learning_rate", learningRate)
                .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
                .SetCell("calibration_bucket_count", calibrationBucketCount)
                .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
                .SetCell("epoch", epoch)
                .SetCell("mean_loss", metrics.MeanLoss)
                .SetCell("trained_states", metrics.TrainedStates)
                .SetCell("training_step_seconds", GetElapsedSeconds(trainingStopwatch))
                .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch));

            if (epoch % weightSnapshotFrequency == 0)
            {
                string checkpointPath = Path.Combine(weightsRootDir, $"epoch{epoch}.bin");
                model.Save(checkpointPath);

                AppendCalibrationRows(
                    analysisOutput: analysisOutput,
                    gameRecords: gameRecords,
                    model: model,
                    experimentName: experimentName,
                    sourceWeightsPath: sourceWeightsPath,
                    rolloutGameCount: rolloutGameCount,
                    epochCount: epochCount,
                    trainingBatchSize: trainingBatchSize,
                    learningRate: learningRate,
                    gameplaySamplingTemp: gameplaySamplingTemp,
                    calibrationBucketCount: calibrationBucketCount,
                    weightSnapshotFrequency: weightSnapshotFrequency,
                    epoch: epoch,
                    experimentStopwatch: experimentStopwatch,
                    rolloutStopwatch: rolloutStopwatch);
                File.WriteAllText(analysisCsvPath, analysisOutput.ToString());
            }

            Console.WriteLine(
                $"[{experimentName}] Epoch {epoch}/{epochCount} trained {metrics.TrainedStates} states " +
                $"with mean loss {metrics.MeanLoss:F6} in {GetElapsedSeconds(trainingStopwatch):F2}s.");
        }

        model.Save(Path.Combine(weightsRootDir, "latest.bin"));
        File.WriteAllText(analysisCsvPath, analysisOutput.ToString());
    }


    static void RunPairAgentDistillationExperiment(
        string experimentName,
        string sourcePairWeightsPath,
        string teacherValueWeightsPath,
        string sourceGameDatabaseName,
        int trainingEpochCount,
        int trainingBatchSize,
        int evaluationGameCount,
        float learningRate,
        float gameplaySamplingTemp,
        float calibrationBucketWidth,
        int weightSnapshotFrequency,
        string analysisCsvPath,
        string weightsRootDir)
    {
        Stopwatch experimentStopwatch = Stopwatch.StartNew();
        CSVBuilder analysisOutput = new();

        GameDatabase sourceGameDatabase = new(sourceGameDatabaseName);
        ValueExperimentGameRecord[] gameRecords = BuildValueExperimentGameRecords(sourceGameDatabase);

        using ValueNetwork teacherModel = new();
        teacherModel.Load(teacherValueWeightsPath);

        using PreferenceValueModel studentModel = new();
        studentModel.Load(sourcePairWeightsPath);
        AdamW optimizer = optim.AdamW(
            parameters: studentModel.parameters(),
            lr: learningRate,
            weight_decay: 0f,
            beta1: 0.9f,
            beta2: 0.99f);

        try
        {
            (float[] predictedLogits, float[] teacherValues) = EvaluateStudentAgainstTeacher(
                gameRecords: gameRecords,
                teacherModel: teacherModel,
                studentModel: studentModel,
                batchSize: trainingBatchSize);

            float calibrationMax = GetCalibrationMax(teacherValues, calibrationBucketWidth);

            AppendPairAgentRewardRow(
                analysisOutput: analysisOutput,
                studentModel: studentModel,
                experimentName: experimentName,
                sourcePairWeightsPath: sourcePairWeightsPath,
                teacherValueWeightsPath: teacherValueWeightsPath,
                sourceGameDatabaseName: sourceGameDatabaseName,
                trainingEpochCount: trainingEpochCount,
                trainingBatchSize: trainingBatchSize,
                evaluationGameCount: evaluationGameCount,
                learningRate: learningRate,
                gameplaySamplingTemp: gameplaySamplingTemp,
                calibrationBucketWidth: calibrationBucketWidth,
                calibrationBucketMax: calibrationMax,
                weightSnapshotFrequency: weightSnapshotFrequency,
                epoch: 0,
                experimentStopwatch: experimentStopwatch);

            AppendTeacherCalibrationRows(
                analysisOutput: analysisOutput,
                predictedAdvantages: predictedLogits,
                trueAdvantages: teacherValues,
                experimentName: experimentName,
                sourcePairWeightsPath: sourcePairWeightsPath,
                teacherValueWeightsPath: teacherValueWeightsPath,
                sourceGameDatabaseName: sourceGameDatabaseName,
                trainingEpochCount: trainingEpochCount,
                trainingBatchSize: trainingBatchSize,
                evaluationGameCount: evaluationGameCount,
                learningRate: learningRate,
                gameplaySamplingTemp: gameplaySamplingTemp,
                calibrationBucketWidth: calibrationBucketWidth,
                calibrationBucketMax: calibrationMax,
                weightSnapshotFrequency: weightSnapshotFrequency,
                epoch: 0,
                experimentStopwatch: experimentStopwatch);

            File.WriteAllText(analysisCsvPath, analysisOutput.ToString());

            for (int epoch = 1; epoch <= trainingEpochCount; ++epoch)
            {
                Stopwatch trainingStopwatch = Stopwatch.StartNew();
                PairAgentDistillationMetrics metrics = TrainPairAgentOnTeacherValues(
                    gameRecords: gameRecords,
                    teacherModel: teacherModel,
                    studentModel: studentModel,
                    optimizer: optimizer,
                    batchSize: trainingBatchSize);
                trainingStopwatch.Stop();

                analysisOutput.NextRow()
                    .SetCell("row_type", "loss")
                    .SetCell("experiment", experimentName)
                    .SetCell("source_pair_weights_path", sourcePairWeightsPath)
                    .SetCell("teacher_value_weights_path", teacherValueWeightsPath)
                    .SetCell("source_game_database_name", sourceGameDatabaseName)
                    .SetCell("training_epoch_count", trainingEpochCount)
                    .SetCell("training_batch_size", trainingBatchSize)
                    .SetCell("evaluation_game_count", evaluationGameCount)
                    .SetCell("learning_rate", learningRate)
                    .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
                    .SetCell("calibration_bucket_width", calibrationBucketWidth)
                    .SetCell("calibration_bucket_min", 0f)
                    .SetCell("calibration_bucket_max", calibrationMax)
                    .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
                    .SetCell("epoch", epoch)
                    .SetCell("mean_loss", metrics.MeanLoss)
                    .SetCell("trained_states", metrics.TrainedStates)
                    .SetCell("training_step_seconds", GetElapsedSeconds(trainingStopwatch))
                    .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch));

                AppendPairAgentRewardRow(
                    analysisOutput: analysisOutput,
                    studentModel: studentModel,
                    experimentName: experimentName,
                    sourcePairWeightsPath: sourcePairWeightsPath,
                    teacherValueWeightsPath: teacherValueWeightsPath,
                    sourceGameDatabaseName: sourceGameDatabaseName,
                    trainingEpochCount: trainingEpochCount,
                    trainingBatchSize: trainingBatchSize,
                    evaluationGameCount: evaluationGameCount,
                    learningRate: learningRate,
                    gameplaySamplingTemp: gameplaySamplingTemp,
                    calibrationBucketWidth: calibrationBucketWidth,
                    calibrationBucketMax: calibrationMax,
                    weightSnapshotFrequency: weightSnapshotFrequency,
                    epoch: epoch,
                    experimentStopwatch: experimentStopwatch);

                if (epoch % weightSnapshotFrequency == 0 || epoch == trainingEpochCount)
                {
                    string checkpointPath = Path.Combine(weightsRootDir, $"epoch{epoch}.bin");
                    studentModel.Save(checkpointPath);

                    (float[] checkpointPredictions, float[] checkpointTeacherValues) = EvaluateStudentAgainstTeacher(
                        gameRecords: gameRecords,
                        teacherModel: teacherModel,
                        studentModel: studentModel,
                        batchSize: trainingBatchSize);

                    AppendTeacherCalibrationRows(
                        analysisOutput: analysisOutput,
                        predictedAdvantages: checkpointPredictions,
                        trueAdvantages: checkpointTeacherValues,
                        experimentName: experimentName,
                        sourcePairWeightsPath: sourcePairWeightsPath,
                        teacherValueWeightsPath: teacherValueWeightsPath,
                        sourceGameDatabaseName: sourceGameDatabaseName,
                        trainingEpochCount: trainingEpochCount,
                        trainingBatchSize: trainingBatchSize,
                        evaluationGameCount: evaluationGameCount,
                        learningRate: learningRate,
                        gameplaySamplingTemp: gameplaySamplingTemp,
                        calibrationBucketWidth: calibrationBucketWidth,
                        calibrationBucketMax: calibrationMax,
                        weightSnapshotFrequency: weightSnapshotFrequency,
                        epoch: epoch,
                        experimentStopwatch: experimentStopwatch);
                }

                File.WriteAllText(analysisCsvPath, analysisOutput.ToString());

                Console.WriteLine(
                    $"[{experimentName}] Epoch {epoch}/{trainingEpochCount} trained {metrics.TrainedStates} states " +
                    $"with mean loss {metrics.MeanLoss:F6}.");
            }

            studentModel.Save(Path.Combine(weightsRootDir, "latest.bin"));
            File.WriteAllText(analysisCsvPath, analysisOutput.ToString());
        }
        finally
        {
            optimizer.Dispose();
        }
    }


    static void ContinueValueNetworkExperimentAndRebuildTrueBucketCalibration(
        string experimentName,
        string sourceWeightsPath,
        string startingValueWeightsPath,
        int rolloutGameCount,
        int totalEpochCount,
        int continuationEpochCount,
        int startingEpoch,
        int trainingBatchSize,
        float learningRate,
        float gameplaySamplingTemp,
        float calibrationBucketWidth,
        int weightSnapshotFrequency,
        string analysisCsvPath,
        string weightsRootDir)
    {
        Stopwatch experimentStopwatch = Stopwatch.StartNew();
        List<ValueExperimentLossRow> lossRows = [];
        float priorElapsedSeconds = 0f;
        string gameDatabaseName = $"{experimentName}_games";
        ValueExperimentGameRecord[] gameRecords = LoadOrCreateValueExperimentGameRecords(
            gameDatabaseName: gameDatabaseName,
            sourceWeightsPath: sourceWeightsPath,
            rolloutGameCount: rolloutGameCount,
            gameplaySamplingTemp: gameplaySamplingTemp);

        // Resume from the last saved value-network checkpoint and extend training.
        if (continuationEpochCount > 0)
        {
            using ValueNetwork model = new();
            model.Load(startingValueWeightsPath);

            using ValueNetworkTrainingPipeline trainingPipeline = new(
                learningRate: learningRate,
                model: model,
                ownsModel: false);

            GameDatabase gameDatabase = new(gameDatabaseName);
            for (int epoch = startingEpoch + 1; epoch <= startingEpoch + continuationEpochCount; ++epoch)
            {
                Stopwatch trainingStopwatch = Stopwatch.StartNew();
                ValueNetworkTrainingMetrics metrics = trainingPipeline.TrainOnAllStates(
                    gameDatabase: gameDatabase,
                    epochCount: 1,
                    batchSize: trainingBatchSize);
                trainingStopwatch.Stop();

                float experimentElapsedSeconds = priorElapsedSeconds + GetElapsedSeconds(experimentStopwatch);
                lossRows.Add(new(
                    Epoch: epoch,
                    MeanLoss: metrics.MeanLoss,
                    TrainedStates: metrics.TrainedStates,
                    TrainingStepSeconds: GetElapsedSeconds(trainingStopwatch),
                    ExperimentElapsedSeconds: experimentElapsedSeconds));

                if (epoch % weightSnapshotFrequency == 0)
                    model.Save(Path.Combine(weightsRootDir, $"epoch{epoch}.bin"));

                Console.WriteLine(
                    $"[{experimentName}] Epoch {epoch}/{startingEpoch + continuationEpochCount} trained {metrics.TrainedStates} states " +
                    $"with mean loss {metrics.MeanLoss:F6} in {GetElapsedSeconds(trainingStopwatch):F2}s.");
            }

            model.Save(Path.Combine(weightsRootDir, "latest.bin"));
        }

        int[] calibrationEpochs = GetContinuationCalibrationEpochs(
            startingEpoch: startingEpoch,
            endingEpoch: startingEpoch + continuationEpochCount,
            weightSnapshotFrequency: weightSnapshotFrequency);
        Dictionary<int, (float[] predictedAdvantages, float[] trueAdvantages)> predictionsByEpoch = [];
        float maxTrueAdvantage = 0f;

        // Evaluate each saved checkpoint once so all calibration plots share the same true-advantage bucket edges.
        for (int epochIndex = 0; epochIndex < calibrationEpochs.Length; ++epochIndex)
        {
            int epoch = calibrationEpochs[epochIndex];
            string weightsPath = GetValueExperimentWeightsPath(
                sourceWeightsPath: startingValueWeightsPath,
                weightsRootDir: weightsRootDir,
                epoch: epoch,
                startingEpoch: startingEpoch);

            using ValueNetwork model = new();
            model.Load(weightsPath);
            (float[] predictedAdvantages, float[] trueAdvantages) = EvaluateValueTargets(
                gameRecords: gameRecords,
                model: model,
                batchSize: trainingBatchSize);
            predictionsByEpoch[epoch] = (predictedAdvantages, trueAdvantages);

            for (int valueIndex = 0; valueIndex < trueAdvantages.Length; ++valueIndex)
                maxTrueAdvantage = MathF.Max(maxTrueAdvantage, trueAdvantages[valueIndex]);
        }

        float fixedBucketMin = 0f;
        float fixedBucketMax = MathF.Max(
            calibrationBucketWidth,
            MathF.Ceiling(maxTrueAdvantage / calibrationBucketWidth) * calibrationBucketWidth);

        CSVBuilder analysisOutput = new();

        if (predictionsByEpoch.TryGetValue(startingEpoch, out (float[] predictedAdvantages, float[] trueAdvantages) startingPredictionSet))
        {
            AppendTrueBucketCalibrationRows(
                analysisOutput: analysisOutput,
                predictedAdvantages: startingPredictionSet.predictedAdvantages,
                trueAdvantages: startingPredictionSet.trueAdvantages,
                experimentName: experimentName,
                sourceWeightsPath: sourceWeightsPath,
                rolloutGameCount: rolloutGameCount,
                epochCount: totalEpochCount,
                trainingBatchSize: trainingBatchSize,
                learningRate: learningRate,
                gameplaySamplingTemp: gameplaySamplingTemp,
                calibrationBucketWidth: calibrationBucketWidth,
                fixedBucketMin: fixedBucketMin,
                fixedBucketMax: fixedBucketMax,
                weightSnapshotFrequency: weightSnapshotFrequency,
                epoch: startingEpoch,
                experimentStopwatch: experimentStopwatch);
        }

        for (int lossIndex = 0; lossIndex < lossRows.Count; ++lossIndex)
        {
            ValueExperimentLossRow lossRow = lossRows[lossIndex];
            int epoch = lossRow.Epoch;

            analysisOutput.NextRow()
                .SetCell("row_type", "loss")
                .SetCell("experiment", experimentName)
                .SetCell("source_weights_path", sourceWeightsPath)
                .SetCell("rollout_game_count", rolloutGameCount)
                .SetCell("epoch_count", totalEpochCount)
                .SetCell("training_batch_size", trainingBatchSize)
                .SetCell("learning_rate", learningRate)
                .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
                .SetCell("calibration_bucket_width", calibrationBucketWidth)
                .SetCell("calibration_bucket_min", fixedBucketMin)
                .SetCell("calibration_bucket_max", fixedBucketMax)
                .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
                .SetCell("epoch", epoch)
                .SetCell("mean_loss", lossRow.MeanLoss)
                .SetCell("trained_states", lossRow.TrainedStates)
                .SetCell("training_step_seconds", lossRow.TrainingStepSeconds)
                .SetCell("experiment_elapsed_seconds", lossRow.ExperimentElapsedSeconds);

            if (predictionsByEpoch.TryGetValue(epoch, out (float[] predictedAdvantages, float[] trueAdvantages) predictionSet))
            {
                AppendTrueBucketCalibrationRows(
                    analysisOutput: analysisOutput,
                    predictedAdvantages: predictionSet.predictedAdvantages,
                    trueAdvantages: predictionSet.trueAdvantages,
                    experimentName: experimentName,
                    sourceWeightsPath: sourceWeightsPath,
                    rolloutGameCount: rolloutGameCount,
                    epochCount: totalEpochCount,
                    trainingBatchSize: trainingBatchSize,
                    learningRate: learningRate,
                    gameplaySamplingTemp: gameplaySamplingTemp,
                    calibrationBucketWidth: calibrationBucketWidth,
                    fixedBucketMin: fixedBucketMin,
                    fixedBucketMax: fixedBucketMax,
                    weightSnapshotFrequency: weightSnapshotFrequency,
                    epoch: epoch,
                    experimentStopwatch: experimentStopwatch);
            }
        }

        File.WriteAllText(analysisCsvPath, analysisOutput.ToString());
    }


    static void AppendCalibrationRows(
        CSVBuilder analysisOutput,
        IReadOnlyList<ValueExperimentGameRecord> gameRecords,
        ValueNetwork model,
        string experimentName,
        string sourceWeightsPath,
        int rolloutGameCount,
        int epochCount,
        int trainingBatchSize,
        float learningRate,
        float gameplaySamplingTemp,
        int calibrationBucketCount,
        int weightSnapshotFrequency,
        int epoch,
        Stopwatch experimentStopwatch,
        Stopwatch rolloutStopwatch)
    {
        (float[] predictedAdvantages, float[] trueAdvantages) = EvaluateValueTargets(
            gameRecords: gameRecords,
            model: model,
            batchSize: trainingBatchSize);
        CalibrationBucketStats[] bucketStats = BuildCalibrationBuckets(
            predictedAdvantages: predictedAdvantages,
            trueAdvantages: trueAdvantages,
            bucketCount: calibrationBucketCount);

        for (int bucketIndex = 0; bucketIndex < bucketStats.Length; ++bucketIndex)
        {
            CalibrationBucketStats bucket = bucketStats[bucketIndex];
            analysisOutput.NextRow()
                .SetCell("row_type", "calibration")
                .SetCell("experiment", experimentName)
                .SetCell("source_weights_path", sourceWeightsPath)
                .SetCell("rollout_game_count", rolloutGameCount)
                .SetCell("epoch_count", epochCount)
                .SetCell("training_batch_size", trainingBatchSize)
                .SetCell("learning_rate", learningRate)
                .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
                .SetCell("calibration_bucket_count", calibrationBucketCount)
                .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
                .SetCell("epoch", epoch)
                .SetCell("bucket_index", bucket.BucketIndex)
                .SetCell("bucket_lower", bucket.BucketLower)
                .SetCell("bucket_upper", bucket.BucketUpper)
                .SetCell("bucket_center", bucket.BucketCenter)
                .SetCell("bucket_count", bucket.Count)
                .SetCell("bucket_predicted_mean", bucket.PredictedMean)
                .SetCell("bucket_true_mean", bucket.TrueMean)
                .SetCell("bucket_true_stddev", bucket.TrueStdDev)
                .SetCell("rollout_generation_seconds", GetElapsedSeconds(rolloutStopwatch))
                .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch));
        }
    }


    static void AppendTrueBucketCalibrationRows(
        CSVBuilder analysisOutput,
        ReadOnlySpan<float> predictedAdvantages,
        ReadOnlySpan<float> trueAdvantages,
        string experimentName,
        string sourceWeightsPath,
        int rolloutGameCount,
        int epochCount,
        int trainingBatchSize,
        float learningRate,
        float gameplaySamplingTemp,
        float calibrationBucketWidth,
        float fixedBucketMin,
        float fixedBucketMax,
        int weightSnapshotFrequency,
        int epoch,
        Stopwatch experimentStopwatch)
    {
        CalibrationBucketStats[] bucketStats = BuildTrueBucketCalibrationBuckets(
            predictedAdvantages: predictedAdvantages,
            trueAdvantages: trueAdvantages,
            bucketMin: fixedBucketMin,
            bucketMax: fixedBucketMax,
            bucketWidth: calibrationBucketWidth);

        for (int bucketIndex = 0; bucketIndex < bucketStats.Length; ++bucketIndex)
        {
            CalibrationBucketStats bucket = bucketStats[bucketIndex];
            analysisOutput.NextRow()
                .SetCell("row_type", "calibration")
                .SetCell("experiment", experimentName)
                .SetCell("source_weights_path", sourceWeightsPath)
                .SetCell("rollout_game_count", rolloutGameCount)
                .SetCell("epoch_count", epochCount)
                .SetCell("training_batch_size", trainingBatchSize)
                .SetCell("learning_rate", learningRate)
                .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
                .SetCell("calibration_bucket_width", calibrationBucketWidth)
                .SetCell("calibration_bucket_min", fixedBucketMin)
                .SetCell("calibration_bucket_max", fixedBucketMax)
                .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
                .SetCell("epoch", epoch)
                .SetCell("bucket_index", bucket.BucketIndex)
                .SetCell("bucket_lower", bucket.BucketLower)
                .SetCell("bucket_upper", bucket.BucketUpper)
                .SetCell("bucket_center", bucket.BucketCenter)
                .SetCell("bucket_count", bucket.Count)
                .SetCell("bucket_predicted_mean", bucket.PredictedMean)
                .SetCell("bucket_predicted_stddev", bucket.PredictedStdDev)
                .SetCell("bucket_true_mean", bucket.TrueMean)
                .SetCell("bucket_true_stddev", bucket.TrueStdDev)
                .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch));
        }
    }


    static ValueExperimentGameRecord[] BuildValueExperimentGameRecords(ReadOnlySpan<GameState> games)
    {
        ValueExperimentGameRecord[] gameRecords = new ValueExperimentGameRecord[games.Length];
        for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
        {
            GameState gameState = games[gameIndex];
            using MemoryStream stream = new();
            gameState.Serialize(stream);
            gameRecords[gameIndex] = new(
                SerializedGame: stream.ToArray(),
                FinalReward: GetReward(gameState),
                MoveCount: gameState.MoveState.MoveHistory.Count);
        }

        return gameRecords;
    }


    static ValueExperimentGameRecord[] BuildValueExperimentGameRecords(GameDatabase gameDatabase)
    {
        List<ValueExperimentGameRecord> gameRecords = [];
        foreach (GameState gameState in gameDatabase)
        {
            using MemoryStream stream = new();
            gameState.Serialize(stream);
            gameRecords.Add(new(
                SerializedGame: stream.ToArray(),
                FinalReward: GetReward(gameState),
                MoveCount: gameState.MoveState.MoveHistory.Count));
        }

        return [.. gameRecords];
    }


    static ValueExperimentGameRecord[] LoadOrCreateValueExperimentGameRecords(
        string gameDatabaseName,
        string sourceWeightsPath,
        int rolloutGameCount,
        float gameplaySamplingTemp)
    {
        string gameDatabasePath = GameDatabase.GetGameDatabasePath(gameDatabaseName);
        if (File.Exists(gameDatabasePath))
        {
            GameDatabase existingGameDatabase = new(gameDatabaseName);
            return BuildValueExperimentGameRecords(existingGameDatabase);
        }

        using PreferenceValueModel sourceModel = new();
        sourceModel.Load(sourceWeightsPath);
        using PreferenceSamplingAgent sourceAgent = new(sourceModel, ownsModel: false);

        GameState[] rolloutGames = PlayGames(
            agent: sourceAgent,
            gameCount: rolloutGameCount,
            temp: gameplaySamplingTemp,
            annotatePolicy: false);

        GameDatabase gameDatabase = new(gameDatabaseName, load: false, delete: true);
        for (int gameIndex = 0; gameIndex < rolloutGames.Length; ++gameIndex)
            gameDatabase.AddGame(rolloutGames[gameIndex]);

        return BuildValueExperimentGameRecords(rolloutGames);
    }


    static (float[] predictedAdvantages, float[] trueAdvantages) EvaluateValueTargets(
        IReadOnlyList<ValueExperimentGameRecord> gameRecords,
        ValueNetwork model,
        int batchSize)
    {
        int effectiveBatchSize = Math.Max(batchSize, 1);
        List<float> predictedAdvantages = [];
        List<float> trueAdvantages = [];

        for (int gameIndex = 0; gameIndex < gameRecords.Count; ++gameIndex)
        {
            ValueExperimentGameRecord gameRecord = gameRecords[gameIndex];

            for (int batchStart = 0; batchStart < gameRecord.PositionCount; batchStart += effectiveBatchSize)
            {
                int currentBatchSize = Math.Min(effectiveBatchSize, gameRecord.PositionCount - batchStart);
                using var scope = NewDisposeScope();
                GameStateEmbedder gameStateEmbedder = new(currentBatchSize);

                for (int stateIndex = 0; stateIndex < currentBatchSize; ++stateIndex)
                {
                    GameState state = MaterializeValueExperimentState(gameRecord, moveStep: batchStart + stateIndex);
                    gameStateEmbedder.AddGameState(state);
                }

                GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(ValueNetwork.EvalDevice);
                using var noGrad = no_grad();
                Tensor predictionsTensor = model.GetAdvantages(gameStateTensors).to(CPU);
                float[] batchPredictions = predictionsTensor.data<float>().ToArray();
                for (int predictionIndex = 0; predictionIndex < batchPredictions.Length; ++predictionIndex)
                {
                    predictedAdvantages.Add(batchPredictions[predictionIndex]);
                    trueAdvantages.Add(gameRecord.FinalReward);
                }
            }
        }

        return ([.. predictedAdvantages], [.. trueAdvantages]);
    }


    static PairAgentDistillationMetrics TrainPairAgentOnTeacherValues(
        IReadOnlyList<ValueExperimentGameRecord> gameRecords,
        ValueNetwork teacherModel,
        PreferenceValueModel studentModel,
        AdamW optimizer,
        int batchSize)
    {
        int effectiveBatchSize = Math.Max(batchSize, 1);
        float totalLoss = 0f;
        int batchCount = 0;
        int totalTrainedStates = 0;

        for (int gameIndex = 0; gameIndex < gameRecords.Count; ++gameIndex)
        {
            ValueExperimentGameRecord gameRecord = gameRecords[gameIndex];
            for (int batchStart = 0; batchStart < gameRecord.PositionCount; batchStart += effectiveBatchSize)
            {
                int currentBatchSize = Math.Min(effectiveBatchSize, gameRecord.PositionCount - batchStart);
                using var scope = NewDisposeScope();
                GameStateEmbedder gameStateEmbedder = new(currentBatchSize);

                for (int stateIndex = 0; stateIndex < currentBatchSize; ++stateIndex)
                {
                    GameState state = MaterializeValueExperimentState(gameRecord, moveStep: batchStart + stateIndex);
                    gameStateEmbedder.AddGameState(state);
                }

                GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
                optimizer.zero_grad();

                Tensor teacherValues;
                using (var noGrad = no_grad())
                    teacherValues = teacherModel.GetAdvantages(gameStateTensors);

                Tensor studentLogits = studentModel.GetLogits(gameStateTensors);
                Tensor loss = TorchSharp.torch.nn.functional.mse_loss(studentLogits, teacherValues);
                loss.backward();
                optimizer.step();

                totalLoss += loss.item<float>();
                totalTrainedStates += currentBatchSize;
                batchCount++;
            }
        }

        return new(
            MeanLoss: totalLoss / Math.Max(1, batchCount),
            TrainedStates: totalTrainedStates);
    }


    static (float[] predictedAdvantages, float[] trueAdvantages) EvaluateStudentAgainstTeacher(
        IReadOnlyList<ValueExperimentGameRecord> gameRecords,
        ValueNetwork teacherModel,
        PreferenceValueModel studentModel,
        int batchSize)
    {
        int effectiveBatchSize = Math.Max(batchSize, 1);
        List<float> predictedAdvantages = [];
        List<float> trueAdvantages = [];

        for (int gameIndex = 0; gameIndex < gameRecords.Count; ++gameIndex)
        {
            ValueExperimentGameRecord gameRecord = gameRecords[gameIndex];
            for (int batchStart = 0; batchStart < gameRecord.PositionCount; batchStart += effectiveBatchSize)
            {
                int currentBatchSize = Math.Min(effectiveBatchSize, gameRecord.PositionCount - batchStart);
                using var scope = NewDisposeScope();
                GameStateEmbedder gameStateEmbedder = new(currentBatchSize);

                for (int stateIndex = 0; stateIndex < currentBatchSize; ++stateIndex)
                {
                    GameState state = MaterializeValueExperimentState(gameRecord, moveStep: batchStart + stateIndex);
                    gameStateEmbedder.AddGameState(state);
                }

                GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
                using var noGrad = no_grad();
                Tensor teacherValuesTensor = teacherModel.GetAdvantages(gameStateTensors).to(CPU);
                Tensor studentLogitsTensor = studentModel.GetLogits(gameStateTensors).to(CPU);
                float[] batchTeacherValues = teacherValuesTensor.data<float>().ToArray();
                float[] batchStudentLogits = studentLogitsTensor.data<float>().ToArray();

                for (int valueIndex = 0; valueIndex < batchTeacherValues.Length; ++valueIndex)
                {
                    predictedAdvantages.Add(batchStudentLogits[valueIndex]);
                    trueAdvantages.Add(batchTeacherValues[valueIndex]);
                }
            }
        }

        return ([.. predictedAdvantages], [.. trueAdvantages]);
    }


    static void AppendPairAgentRewardRow(
        CSVBuilder analysisOutput,
        PreferenceValueModel studentModel,
        string experimentName,
        string sourcePairWeightsPath,
        string teacherValueWeightsPath,
        string sourceGameDatabaseName,
        int trainingEpochCount,
        int trainingBatchSize,
        int evaluationGameCount,
        float learningRate,
        float gameplaySamplingTemp,
        float calibrationBucketWidth,
        float calibrationBucketMax,
        int weightSnapshotFrequency,
        int epoch,
        Stopwatch experimentStopwatch)
    {
        using PreferenceSamplingAgent agent = new(studentModel, ownsModel: false);
        Stopwatch rewardStopwatch = Stopwatch.StartNew();
        GameState[] evaluationGames = PlayGames(
            agent: agent,
            gameCount: evaluationGameCount,
            temp: gameplaySamplingTemp,
            annotatePolicy: false);
        rewardStopwatch.Stop();

        analysisOutput.NextRow()
            .SetCell("row_type", "reward")
            .SetCell("experiment", experimentName)
            .SetCell("source_pair_weights_path", sourcePairWeightsPath)
            .SetCell("teacher_value_weights_path", teacherValueWeightsPath)
            .SetCell("source_game_database_name", sourceGameDatabaseName)
            .SetCell("training_epoch_count", trainingEpochCount)
            .SetCell("training_batch_size", trainingBatchSize)
            .SetCell("evaluation_game_count", evaluationGameCount)
            .SetCell("learning_rate", learningRate)
            .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
            .SetCell("calibration_bucket_width", calibrationBucketWidth)
            .SetCell("calibration_bucket_min", 0f)
            .SetCell("calibration_bucket_max", calibrationBucketMax)
            .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
            .SetCell("epoch", epoch)
            .SetCell("reward_mean", GetMeanReward(evaluationGames))
            .SetCell("reward_stddev", GetRewardStdDev(evaluationGames))
            .SetCell("reward_eval_seconds", GetElapsedSeconds(rewardStopwatch))
            .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch));
    }


    static void AppendTeacherCalibrationRows(
        CSVBuilder analysisOutput,
        ReadOnlySpan<float> predictedAdvantages,
        ReadOnlySpan<float> trueAdvantages,
        string experimentName,
        string sourcePairWeightsPath,
        string teacherValueWeightsPath,
        string sourceGameDatabaseName,
        int trainingEpochCount,
        int trainingBatchSize,
        int evaluationGameCount,
        float learningRate,
        float gameplaySamplingTemp,
        float calibrationBucketWidth,
        float calibrationBucketMax,
        int weightSnapshotFrequency,
        int epoch,
        Stopwatch experimentStopwatch)
    {
        CalibrationBucketStats[] bucketStats = BuildTrueBucketCalibrationBuckets(
            predictedAdvantages: predictedAdvantages,
            trueAdvantages: trueAdvantages,
            bucketMin: 0f,
            bucketMax: calibrationBucketMax,
            bucketWidth: calibrationBucketWidth);

        for (int bucketIndex = 0; bucketIndex < bucketStats.Length; ++bucketIndex)
        {
            CalibrationBucketStats bucket = bucketStats[bucketIndex];
            analysisOutput.NextRow()
                .SetCell("row_type", "calibration")
                .SetCell("experiment", experimentName)
                .SetCell("source_pair_weights_path", sourcePairWeightsPath)
                .SetCell("teacher_value_weights_path", teacherValueWeightsPath)
                .SetCell("source_game_database_name", sourceGameDatabaseName)
                .SetCell("training_epoch_count", trainingEpochCount)
                .SetCell("training_batch_size", trainingBatchSize)
                .SetCell("evaluation_game_count", evaluationGameCount)
                .SetCell("learning_rate", learningRate)
                .SetCell("gameplay_sampling_temp", gameplaySamplingTemp)
                .SetCell("calibration_bucket_width", calibrationBucketWidth)
                .SetCell("calibration_bucket_min", 0f)
                .SetCell("calibration_bucket_max", calibrationBucketMax)
                .SetCell("weight_snapshot_frequency", weightSnapshotFrequency)
                .SetCell("epoch", epoch)
                .SetCell("bucket_index", bucket.BucketIndex)
                .SetCell("bucket_lower", bucket.BucketLower)
                .SetCell("bucket_upper", bucket.BucketUpper)
                .SetCell("bucket_center", bucket.BucketCenter)
                .SetCell("bucket_count", bucket.Count)
                .SetCell("bucket_predicted_mean", bucket.PredictedMean)
                .SetCell("bucket_predicted_stddev", bucket.PredictedStdDev)
                .SetCell("bucket_true_mean", bucket.TrueMean)
                .SetCell("bucket_true_stddev", bucket.TrueStdDev)
                .SetCell("experiment_elapsed_seconds", GetElapsedSeconds(experimentStopwatch));
        }
    }


    static float GetCalibrationMax(ReadOnlySpan<float> trueAdvantages, float calibrationBucketWidth)
    {
        float maxTrueAdvantage = 0f;
        for (int valueIndex = 0; valueIndex < trueAdvantages.Length; ++valueIndex)
            maxTrueAdvantage = MathF.Max(maxTrueAdvantage, trueAdvantages[valueIndex]);

        return MathF.Max(
            calibrationBucketWidth,
            MathF.Ceiling(maxTrueAdvantage / calibrationBucketWidth) * calibrationBucketWidth);
    }


    static CalibrationBucketStats[] BuildCalibrationBuckets(
        ReadOnlySpan<float> predictedAdvantages,
        ReadOnlySpan<float> trueAdvantages,
        int bucketCount)
    {
        int effectiveBucketCount = Math.Max(bucketCount, 1);
        CalibrationBucketAccumulator[] accumulators = new CalibrationBucketAccumulator[effectiveBucketCount];
        for (int bucketIndex = 0; bucketIndex < accumulators.Length; ++bucketIndex)
            accumulators[bucketIndex] = new(bucketIndex);

        float minPrediction = float.PositiveInfinity;
        float maxPrediction = float.NegativeInfinity;
        for (int valueIndex = 0; valueIndex < predictedAdvantages.Length; ++valueIndex)
        {
            float prediction = predictedAdvantages[valueIndex];
            minPrediction = MathF.Min(minPrediction, prediction);
            maxPrediction = MathF.Max(maxPrediction, prediction);
        }

        if (predictedAdvantages.Length == 0)
            return CreateEmptyCalibrationBuckets(minPrediction: 0f, maxPrediction: 1f, effectiveBucketCount);

        if (minPrediction == maxPrediction)
        {
            minPrediction -= 0.5f;
            maxPrediction += 0.5f;
        }

        float bucketWidth = (maxPrediction - minPrediction) / effectiveBucketCount;

        for (int valueIndex = 0; valueIndex < predictedAdvantages.Length; ++valueIndex)
        {
            float prediction = predictedAdvantages[valueIndex];
            float target = trueAdvantages[valueIndex];
            int bucketIndex = (int)((prediction - minPrediction) / bucketWidth);
            if (bucketIndex < 0)
                bucketIndex = 0;
            if (bucketIndex >= effectiveBucketCount)
                bucketIndex = effectiveBucketCount - 1;

            CalibrationBucketAccumulator accumulator = accumulators[bucketIndex];
            accumulator.Count++;
            accumulator.PredictedSum += prediction;
            accumulator.TrueSum += target;
            accumulator.TrueSquaredSum += target * target;
            accumulators[bucketIndex] = accumulator;
        }

        CalibrationBucketStats[] bucketStats = new CalibrationBucketStats[effectiveBucketCount];
        for (int bucketIndex = 0; bucketIndex < effectiveBucketCount; ++bucketIndex)
        {
            float bucketLower = minPrediction + bucketWidth * bucketIndex;
            float bucketUpper = bucketIndex == effectiveBucketCount - 1 ? maxPrediction : bucketLower + bucketWidth;
            CalibrationBucketAccumulator accumulator = accumulators[bucketIndex];

            float predictedMean = accumulator.Count == 0 ? 0f : accumulator.PredictedSum / accumulator.Count;
            float trueMean = accumulator.Count == 0 ? 0f : accumulator.TrueSum / accumulator.Count;
            float trueVariance = 0f;
            if (accumulator.Count > 1)
            {
                trueVariance = (accumulator.TrueSquaredSum - accumulator.TrueSum * trueMean) / (accumulator.Count - 1);
                if (trueVariance < 0f)
                    trueVariance = 0f;
            }

            bucketStats[bucketIndex] = new(
                BucketIndex: bucketIndex,
                BucketLower: bucketLower,
                BucketUpper: bucketUpper,
                BucketCenter: (bucketLower + bucketUpper) * 0.5f,
                Count: accumulator.Count,
                PredictedMean: predictedMean,
                PredictedStdDev: 0f,
                TrueMean: trueMean,
                TrueStdDev: MathF.Sqrt(trueVariance));
        }

        return bucketStats;
    }


    static CalibrationBucketStats[] BuildTrueBucketCalibrationBuckets(
        ReadOnlySpan<float> predictedAdvantages,
        ReadOnlySpan<float> trueAdvantages,
        float bucketMin,
        float bucketMax,
        float bucketWidth)
    {
        float effectiveBucketWidth = MathF.Max(bucketWidth, 1e-6f);
        float effectiveBucketMax = MathF.Max(bucketMax, bucketMin + effectiveBucketWidth);
        int bucketCount = (int)MathF.Ceiling((effectiveBucketMax - bucketMin) / effectiveBucketWidth);
        CalibrationBucketAccumulator[] accumulators = new CalibrationBucketAccumulator[bucketCount];
        for (int bucketIndex = 0; bucketIndex < bucketCount; ++bucketIndex)
            accumulators[bucketIndex] = new(bucketIndex);

        for (int valueIndex = 0; valueIndex < predictedAdvantages.Length; ++valueIndex)
        {
            float prediction = predictedAdvantages[valueIndex];
            float target = trueAdvantages[valueIndex];
            float clampedTarget = MathF.Min(MathF.Max(target, bucketMin), effectiveBucketMax - 1e-6f);
            int bucketIndex = (int)((clampedTarget - bucketMin) / effectiveBucketWidth);
            if (bucketIndex < 0)
                bucketIndex = 0;
            if (bucketIndex >= bucketCount)
                bucketIndex = bucketCount - 1;

            CalibrationBucketAccumulator accumulator = accumulators[bucketIndex];
            accumulator.Count++;
            accumulator.PredictedSum += prediction;
            accumulator.PredictedSquaredSum += prediction * prediction;
            accumulator.TrueSum += target;
            accumulator.TrueSquaredSum += target * target;
            accumulators[bucketIndex] = accumulator;
        }

        CalibrationBucketStats[] bucketStats = new CalibrationBucketStats[bucketCount];
        for (int bucketIndex = 0; bucketIndex < bucketCount; ++bucketIndex)
        {
            float bucketLower = bucketMin + effectiveBucketWidth * bucketIndex;
            float bucketUpper = bucketLower + effectiveBucketWidth;
            CalibrationBucketAccumulator accumulator = accumulators[bucketIndex];
            float predictedMean = accumulator.Count == 0 ? 0f : accumulator.PredictedSum / accumulator.Count;
            float predictedVariance = 0f;
            if (accumulator.Count > 1)
            {
                predictedVariance = (accumulator.PredictedSquaredSum - accumulator.PredictedSum * predictedMean) / (accumulator.Count - 1);
                if (predictedVariance < 0f)
                    predictedVariance = 0f;
            }

            float trueMean = accumulator.Count == 0 ? 0f : accumulator.TrueSum / accumulator.Count;
            float trueVariance = 0f;
            if (accumulator.Count > 1)
            {
                trueVariance = (accumulator.TrueSquaredSum - accumulator.TrueSum * trueMean) / (accumulator.Count - 1);
                if (trueVariance < 0f)
                    trueVariance = 0f;
            }

            bucketStats[bucketIndex] = new(
                BucketIndex: bucketIndex,
                BucketLower: bucketLower,
                BucketUpper: bucketUpper,
                BucketCenter: bucketLower + effectiveBucketWidth * 0.5f,
                Count: accumulator.Count,
                PredictedMean: predictedMean,
                PredictedStdDev: MathF.Sqrt(predictedVariance),
                TrueMean: trueMean,
                TrueStdDev: MathF.Sqrt(trueVariance));
        }

        return bucketStats;
    }


    static CalibrationBucketStats[] CreateEmptyCalibrationBuckets(float minPrediction, float maxPrediction, int bucketCount)
    {
        CalibrationBucketStats[] bucketStats = new CalibrationBucketStats[bucketCount];
        float bucketWidth = (maxPrediction - minPrediction) / bucketCount;
        for (int bucketIndex = 0; bucketIndex < bucketCount; ++bucketIndex)
        {
            float bucketLower = minPrediction + bucketWidth * bucketIndex;
            float bucketUpper = bucketIndex == bucketCount - 1 ? maxPrediction : bucketLower + bucketWidth;
            bucketStats[bucketIndex] = new(
                BucketIndex: bucketIndex,
                BucketLower: bucketLower,
                BucketUpper: bucketUpper,
                BucketCenter: (bucketLower + bucketUpper) * 0.5f,
                Count: 0,
                PredictedMean: 0f,
                PredictedStdDev: 0f,
                TrueMean: 0f,
                TrueStdDev: 0f);
        }

        return bucketStats;
    }


    static int[] GetContinuationCalibrationEpochs(int startingEpoch, int endingEpoch, int weightSnapshotFrequency)
    {
        List<int> epochs = [];

        epochs.Add(startingEpoch);
        for (int epoch = startingEpoch + weightSnapshotFrequency; epoch <= endingEpoch; epoch += weightSnapshotFrequency)
            epochs.Add(epoch);

        if (epochs[^1] != endingEpoch)
            epochs.Add(endingEpoch);

        return [.. epochs];
    }


    static string GetValueExperimentWeightsPath(string sourceWeightsPath, string weightsRootDir, int epoch, int startingEpoch)
    {
        if (epoch == startingEpoch)
            return sourceWeightsPath;

        return Path.Combine(weightsRootDir, $"epoch{epoch}.bin");
    }


    static GameState MaterializeValueExperimentState(ValueExperimentGameRecord gameRecord, int moveStep)
    {
        GameState gameState = new(GameData.Default);
        using MemoryStream stream = new(gameRecord.SerializedGame, writable: false);
        gameState.Deserialize(stream);
        gameState.MoveState.RevertToStep(moveStep);
        return gameState;
    }


    static void RunSavedCheckpointEvaluation(
        string experimentName,
        IReadOnlyList<int> checkpointLabels,
        IReadOnlyList<string> checkpointWeightsPaths,
        float temp,
        int gameCount,
        string analysisCsvPath)
    {
        if (checkpointLabels.Count != checkpointWeightsPaths.Count)
            throw new ArgumentException("Checkpoint labels and paths must have the same length.");

        CSVBuilder output = new();

        for (int checkpointIndex = 0; checkpointIndex < checkpointWeightsPaths.Count; ++checkpointIndex)
        {
            string sourceWeightsPath = checkpointWeightsPaths[checkpointIndex];
            int checkpointLabel = checkpointLabels[checkpointIndex];

            using PreferenceValueModel model = new();
            model.Load(sourceWeightsPath);
            using PreferenceSamplingAgent agent = new(model, ownsModel: false);

            Stopwatch stopwatch = Stopwatch.StartNew();
            GameState[] games = PlayGames(agent: agent, gameCount: gameCount, temp: temp, annotatePolicy: true);
            stopwatch.Stop();

            output.NextRow()
                .SetCell("checkpoint_label", checkpointLabel)
                .SetCell("experiment", experimentName)
                .SetCell("source_weights_path", sourceWeightsPath)
                .SetCell("temp", temp)
                .SetCell("game_count", gameCount)
                .SetCell("elapsed_seconds", GetElapsedSeconds(stopwatch))
                .SetCell("reward_mean", GetMeanReward(games))
                .SetCell("reward_stddev", GetRewardStdDev(games));

            AnalyzeGames(
                analysisGames: games,
                analysisOutput: output,
                analyzers: RewardAndEntropyAnalyzers);

            Console.WriteLine(
                $"[{experimentName}] Checkpoint {checkpointLabel} evaluated {gameCount} games at temp {temp:F4}. " +
                $"Reward mean {GetMeanReward(games):F4}, stddev {GetRewardStdDev(games):F4}.");
        }

        File.WriteAllText(analysisCsvPath, output.ToString());
    }


    static void RunSnapshotEvaluationSweep(
        string snapshotExperimentName,
        IReadOnlyList<string> sourceWeightsPaths,
        float temp,
        int gameCount)
    {
        string repoRoot = FindRepoRoot();
        string snapshotDir = Path.Combine(repoRoot, "Analysis", snapshotExperimentName);
        Directory.CreateDirectory(snapshotDir);

        CSVBuilder output = new();
        List<string> readmeLines =
        [
            $"# {snapshotExperimentName}",
            string.Empty,
            $"- Temperature: `{temp}`",
            $"- Games per snapshot: `{gameCount}`",
            string.Empty,
        ];

        for (int snapshotIndex = 0; snapshotIndex < sourceWeightsPaths.Count; ++snapshotIndex)
        {
            string sourceWeightsPath = sourceWeightsPaths[snapshotIndex];
            int iteration = snapshotIndex + 1;
            string snapshotWeightsPath = Path.Combine(snapshotDir, $"iter{iteration}.bin");
            File.Copy(sourceWeightsPath, snapshotWeightsPath, overwrite: true);

            using PreferenceValueModel model = new();
            model.Load(snapshotWeightsPath);
            using PreferenceSamplingAgent agent = new(model, ownsModel: false);

            Stopwatch stopwatch = Stopwatch.StartNew();
            GameState[] games = PlayGames(agent: agent, gameCount: gameCount, temp: temp, annotatePolicy: true);
            stopwatch.Stop();

            output.NextRow()
                .SetCell("iteration", iteration)
                .SetCell("source_weights_path", sourceWeightsPath)
                .SetCell("snapshot_weights_path", snapshotWeightsPath)
                .SetCell("temp", temp)
                .SetCell("game_count", gameCount)
                .SetCell("elapsed_seconds", GetElapsedSeconds(stopwatch))
                .SetCell("reward_mean", GetMeanReward(games))
                .SetCell("reward_stddev", GetRewardStdDev(games));

            AnalyzeGames(games, output);

            readmeLines.Add(
                $"- Iteration {iteration}: reward mean `{GetMeanReward(games):F4}`, " +
                $"stddev `{GetRewardStdDev(games):F4}`, elapsed `{GetElapsedSeconds(stopwatch):F2}s`");

            Console.WriteLine(
                $"[{snapshotExperimentName}] Iteration {iteration} evaluated {gameCount} games at temp {temp:F3}. " +
                $"Reward mean {GetMeanReward(games):F4}, stddev {GetRewardStdDev(games):F4}.");
        }

        File.WriteAllText(Path.Combine(snapshotDir, "analysis.csv"), output.ToString());
        File.WriteAllText(Path.Combine(snapshotDir, "README.md"), string.Join(Environment.NewLine, readmeLines));
    }


    static float GetMeanReward(ReadOnlySpan<PreferenceGameRecord> trainingGames)
    {
        if (trainingGames.Length == 0)
            return 0f;

        float totalReward = 0f;
        for (int gameIndex = 0; gameIndex < trainingGames.Length; ++gameIndex)
            totalReward += trainingGames[gameIndex].FinalReward;

        return totalReward / trainingGames.Length;
    }


    static float GetRewardStdDev(ReadOnlySpan<PreferenceGameRecord> trainingGames)
    {
        if (trainingGames.Length <= 1)
            return 0f;

        float meanReward = GetMeanReward(trainingGames);
        float sqErrorTotal = 0f;
        for (int gameIndex = 0; gameIndex < trainingGames.Length; ++gameIndex)
        {
            float error = trainingGames[gameIndex].FinalReward - meanReward;
            sqErrorTotal += error * error;
        }

        return MathF.Sqrt(sqErrorTotal / (trainingGames.Length - 1));
    }


    static GameState[] PlayGames(PreferenceSamplingAgent agent, int gameCount, float temp, bool annotatePolicy)
    {
        GameState[] games = new GameState[Math.Max(0, gameCount)];
        for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
            games[gameIndex] = new(GameData.Default);

        while (true)
        {
            bool allGamesDone = true;
            for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
            {
                if (!agent.IsGameDone(games[gameIndex]))
                {
                    allGamesDone = false;
                    break;
                }
            }

            if (allGamesDone)
                break;

            agent.MakeMove(temp: temp, annotatePolicy: annotatePolicy, games);
        }

        return games;
    }


    static float GetMeanReward(ReadOnlySpan<GameState> games)
    {
        if (games.Length == 0)
            return 0f;

        float totalReward = 0f;
        for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
            totalReward += GetReward(games[gameIndex]);

        return totalReward / games.Length;
    }


    static float GetRewardStdDev(ReadOnlySpan<GameState> games)
    {
        if (games.Length <= 1)
            return 0f;

        float meanReward = GetMeanReward(games);
        float sqErrorTotal = 0f;
        for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
        {
            float error = GetReward(games[gameIndex]) - meanReward;
            sqErrorTotal += error * error;
        }

        return MathF.Sqrt(sqErrorTotal / (games.Length - 1));
    }


    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalChips >= 300)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalChips / 1000f;
    }


    readonly record struct ValueExperimentGameRecord(byte[] SerializedGame, float FinalReward, int MoveCount)
    {
        public int PositionCount => MoveCount + 1;
    }


    readonly record struct ValueExperimentLossRow(
        int Epoch,
        float MeanLoss,
        int TrainedStates,
        float TrainingStepSeconds,
        float ExperimentElapsedSeconds);


    readonly record struct PairAgentDistillationMetrics(float MeanLoss, int TrainedStates);


    readonly record struct StepMatchedPairDataset(
        StepMatchedGamePair[] Pairs,
        int AvailableOrderedPairCount,
        int EligibleStepCount);


    readonly record struct StepMatchedGamePair(
        int LeftGameIndex,
        int RightGameIndex,
        int MoveStep,
        float Target);


    readonly record struct StepBucket(
        int MoveStep,
        int[] GameIndices,
        int OrderedPairCount);


    struct CalibrationBucketAccumulator
    {
        public readonly int BucketIndex;

        public int Count;

        public float PredictedSum;

        public float PredictedSquaredSum;

        public float TrueSum;

        public float TrueSquaredSum;

        public CalibrationBucketAccumulator(int bucketIndex)
        {
            BucketIndex = bucketIndex;
        }
    }


    readonly record struct CalibrationBucketStats(
        int BucketIndex,
        float BucketLower,
        float BucketUpper,
        float BucketCenter,
        int Count,
        float PredictedMean,
        float PredictedStdDev,
        float TrueMean,
        float TrueStdDev);
}
