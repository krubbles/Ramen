namespace Ramen.ConsoleApp;

using System;
using System.IO;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public static class Program
{
    public static void Main()
    {
        // Do not change START
        set_default_device(MPS);
        Ramen.AI.TensorManager.Init();
        Console.WriteLine("=== START ===");
        // Do not change END

        string experimentName = "2026-04-02_bt_value_g5000_p500000_a2000_s100";
        int rolloutGameCount = 5000;
        int trainingPairCount = 500000;
        int trainingPairIncrement = 2000;
        int trainingBatchSize = 256;
        float learningRate = 3e-4f;
        int analysisSampleSize = 100;
        int weightSnapshotFrequency = 5;
        bool shouldRunTraining = true;

        string repoRoot = FindRepoRoot();
        string analysisDir = Path.Combine(repoRoot, "Analysis", experimentName);
        Directory.CreateDirectory(analysisDir);
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string weightsDir = Path.Combine(analysisDir, "weights");
        Directory.CreateDirectory(weightsDir);

        if (shouldRunTraining)
        {
            RunTraining(
                experimentName: experimentName,
                rolloutGameCount: rolloutGameCount,
                trainingPairCount: trainingPairCount,
                trainingPairIncrement: trainingPairIncrement,
                trainingBatchSize: trainingBatchSize,
                learningRate: learningRate,
                analysisSampleSize: analysisSampleSize,
                weightSnapshotFrequency: weightSnapshotFrequency,
                analysisCsvPath: analysisCsvPath,
                weightsDir: weightsDir);
        }

        Console.WriteLine($"Experiment directory ready: {analysisDir}");
    }


    static void RunTraining(
        string experimentName,
        int rolloutGameCount,
        int trainingPairCount,
        int trainingPairIncrement,
        int trainingBatchSize,
        float learningRate,
        int analysisSampleSize,
        int weightSnapshotFrequency,
        string analysisCsvPath,
        string weightsDir)
    {
        using PreferenceTrainingPipeline trainingPipeline = new(learningRate: learningRate);
        PreferenceGameRecord[] trainingGames = trainingPipeline.PlayTrainingGames(gameCount: rolloutGameCount);

        RewardStatsTrainingRunAnalyzer rewardAnalyzer = new();
        PolicyEntropyTrainingRunAnalyzer entropyAnalyzer = new();
        HandTypePresenceTrainingRunAnalyzer handTypeAnalyzer = new();
        EndStateHandCountTrainingRunAnalyzer endStateAnalyzer = new();
        CSVBuilder analysisOutput = new();

        for (int trainedPairs = trainingPairIncrement; trainedPairs <= trainingPairCount; trainedPairs += trainingPairIncrement)
        {
            PreferenceTrainingMetrics metrics = trainingPipeline.TrainOnRandomPairs(
                gameRecords: trainingGames,
                pairCount: trainingPairIncrement,
                batchSize: trainingBatchSize);

            GameState[] analysisGames = trainingPipeline.PlayAnalysisGames(gameCount: analysisSampleSize);

            analysisOutput.NextRow()
                .SetCell("experiment", experimentName)
                .SetCell("trained_pairs", trainedPairs)
                .SetCell("pair_batch_loss", metrics.MeanLoss)
                .SetCell("rollout_game_count", rolloutGameCount)
                .SetCell("analysis_game_count", analysisSampleSize)
                .SetCell("rollout_reward_mean", GetMeanReward(trainingGames))
                .SetCell("rollout_reward_stddev", GetRewardStdDev(trainingGames));

            rewardAnalyzer.Analyze(analysisGames, analysisOutput);
            entropyAnalyzer.Analyze(analysisGames, analysisOutput);
            handTypeAnalyzer.Analyze(analysisGames, analysisOutput);
            endStateAnalyzer.Analyze(analysisGames, analysisOutput);

            File.WriteAllText(analysisCsvPath, analysisOutput.ToString());

            int analysisPoint = trainedPairs / trainingPairIncrement;
            if (analysisPoint % weightSnapshotFrequency == 0)
                trainingPipeline.Save(Path.Combine(weightsDir, $"{trainedPairs}.bin"));

            Console.WriteLine($"[{experimentName}] Trained {trainedPairs}/{trainingPairCount} pairs.");
        }

        trainingPipeline.Save(Path.Combine(weightsDir, "latest.bin"));
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
}
