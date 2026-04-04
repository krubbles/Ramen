namespace Ramen.ConsoleApp;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
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

    public static void Main()
    {
        // Do not change START
        set_default_device(mps_is_available() ? MPS : CPU);
        Ramen.AI.TensorManager.Init();
        Console.WriteLine("=== START ===");
        // Do not change END

        string experimentName = "2026-04-03_bt_value_iterative_g1000_p50000_i10_latest_only_t0p01_a100";
        int rolloutIterationCount = 10;
        int rolloutGameCount = 1000;
        int trainingPairCount = 50000;
        int trainingPairIncrement = 2000;
        int trainingBatchSize = 256;
        int trainingPairMicroBatchSize = 256;
        float learningRate = 3e-4f;
        float gameplaySamplingTemp = 0.01f;
        int analysisSampleSize = 100;
        int weightSnapshotFrequency = 5;
        bool shouldRunTraining = true;
        bool shouldEvaluateSnapshot = false;
        string snapshotSourceWeightsPath = Path.Combine(
            FindRepoRoot(),
            "Analysis",
            "2026-04-03_bt_value_iterative_g2000_p100000_i5_uniform_replay_interleaved_a100",
            "weights",
            "iter5",
            "latest.bin");
        string snapshotExperimentName = "2026-04-03_bt_value_iterative_g2000_p100000_i5_uniform_replay_interleaved_a100_iter_eval_temp0001_g500";
        int snapshotGameCount = 500;
        float snapshotTemp = 0.001f;

        string repoRoot = FindRepoRoot();
        string analysisDir = Path.Combine(repoRoot, "Analysis", experimentName);
        Directory.CreateDirectory(analysisDir);
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string weightsDir = Path.Combine(analysisDir, "weights");
        Directory.CreateDirectory(weightsDir);

        if (shouldRunTraining)
        {
            RunIterativeLatestOnlyTraining(
                experimentName: experimentName,
                rolloutIterationCount: rolloutIterationCount,
                rolloutGameCount: rolloutGameCount,
                trainingPairCount: trainingPairCount,
                trainingPairIncrement: trainingPairIncrement,
                trainingBatchSize: trainingBatchSize,
                trainingPairMicroBatchSize: trainingPairMicroBatchSize,
                learningRate: learningRate,
                gameplaySamplingTemp: gameplaySamplingTemp,
                analysisSampleSize: analysisSampleSize,
                weightSnapshotFrequency: weightSnapshotFrequency,
                analysisCsvPath: analysisCsvPath,
                weightsRootDir: weightsDir);
        }

        if (shouldEvaluateSnapshot)
        {
            RunSnapshotEvaluationSweep(
                snapshotExperimentName: snapshotExperimentName,
                sourceWeightsPaths:
                [
                    Path.Combine(
                        FindRepoRoot(),
                        "Analysis",
                        "2026-04-03_bt_value_iterative_g2000_p100000_i5_uniform_replay_interleaved_a100",
                        "weights",
                        "iter1",
                        "latest.bin"),
                    Path.Combine(
                        FindRepoRoot(),
                        "Analysis",
                        "2026-04-03_bt_value_iterative_g2000_p100000_i5_uniform_replay_interleaved_a100",
                        "weights",
                        "iter2",
                        "latest.bin"),
                    Path.Combine(
                        FindRepoRoot(),
                        "Analysis",
                        "2026-04-03_bt_value_iterative_g2000_p100000_i5_uniform_replay_interleaved_a100",
                        "weights",
                        "iter3",
                        "latest.bin"),
                    Path.Combine(
                        FindRepoRoot(),
                        "Analysis",
                        "2026-04-03_bt_value_iterative_g2000_p100000_i5_uniform_replay_interleaved_a100",
                        "weights",
                        "iter4",
                        "latest.bin"),
                    Path.Combine(
                        FindRepoRoot(),
                        "Analysis",
                        "2026-04-03_bt_value_iterative_g2000_p100000_i5_uniform_replay_interleaved_a100",
                        "weights",
                        "iter5",
                        "latest.bin"),
                ],
                temp: snapshotTemp,
                gameCount: snapshotGameCount);
        }

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
        for (int analyzerIndex = 0; analyzerIndex < Analyzers.Length; ++analyzerIndex)
            Analyzers[analyzerIndex].Analyze(analysisGames, analysisOutput);
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
}
