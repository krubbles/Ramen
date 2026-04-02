namespace Ramen.ConsoleApp;

using System;
using System.IO;
using System.Threading;
using Ramen.Agents;
using Ramen.AI;
using Ramen.Training;
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

        string experimentName = "2026-02-14_grpo_s1000_r5000_lr1e-5_ent0p02";
        int trainingSteps = 1000;
        int rolloutSize = 5000;
        float learningRate = 1e-5f;
        float entropyCoeff = 0.02f;
        int snapshotFrequency = 5;
        int analysisSampleSize = 512;
        bool shouldRunTraining = false;

        string repoRoot = FindRepoRoot();
        string analysisDir = Path.Combine(repoRoot, "Analysis", experimentName);
        Directory.CreateDirectory(analysisDir);

        if (shouldRunTraining)
        {
            RunTraining(
                experimentName: experimentName,
                trainingSteps: trainingSteps,
                rolloutSize: rolloutSize,
                learningRate: learningRate,
                entropyCoeff: entropyCoeff,
                snapshotFrequency: snapshotFrequency);
        }

        WriteAnalyzerCsvs(
            analysisDir: analysisDir,
            experimentName: experimentName,
            analysisSampleSize: analysisSampleSize);
    }


    static void RunTraining(string experimentName, int trainingSteps, int rolloutSize, float learningRate, float entropyCoeff, int snapshotFrequency)
    {
        BasicGRPOTrainingRun trainingRun = new()
        {
            RolloutSize = rolloutSize,
            LearningRate = learningRate,
            Entropy = entropyCoeff,
        };

        using CancellationTokenSource cancellationTokenSource = new();
        ITrainingRun.Run(
            trainingRun: trainingRun,
            runName: experimentName,
            steps: trainingSteps,
            samplingFrequency: snapshotFrequency,
            cancellationTokenSource: cancellationTokenSource).Wait();
    }


    static void WriteAnalyzerCsvs(string analysisDir, string experimentName, int analysisSampleSize)
    {
        // Entropy over snapshots.
        CSVBuilder entropyOutput = TrainingRunAnalysis.Analyze(
            runName: experimentName,
            agentLoader: LoadSnapshotAgent,
            sampleSize: analysisSampleSize,
            new PolicyEntropyTrainingRunAnalyzer());
        string entropyPath = Path.Combine(analysisDir, "entropy.csv");
        File.WriteAllText(entropyPath, entropyOutput.ToString());

        // Average reward over snapshots.
        CSVBuilder avgRewardOutput = TrainingRunAnalysis.Analyze(
            runName: experimentName,
            agentLoader: LoadSnapshotAgent,
            sampleSize: analysisSampleSize,
            new RewardStatsTrainingRunAnalyzer());
        string avgRewardPath = Path.Combine(analysisDir, "avg_reward.csv");
        File.WriteAllText(avgRewardPath, avgRewardOutput.ToString());

        // End-state outcome distribution over snapshots.
        CSVBuilder outcomeDistOutput = TrainingRunAnalysis.Analyze(
            runName: experimentName,
            agentLoader: LoadSnapshotAgent,
            sampleSize: analysisSampleSize,
            new EndStateHandCountTrainingRunAnalyzer());
        string outcomeDistPath = Path.Combine(analysisDir, "outcome_dist.csv");
        File.WriteAllText(outcomeDistPath, outcomeDistOutput.ToString());

        Console.WriteLine($"Wrote entropy CSV: {entropyPath}");
        Console.WriteLine($"Wrote average reward CSV: {avgRewardPath}");
        Console.WriteLine($"Wrote outcome distribution CSV: {outcomeDistPath}");
    }


    static IAgent LoadSnapshotAgent(string filePath)
    {
        PolicyModel model = new();
        model.Load(filePath);
        return new PolicyOnlyAgent(model, ownsModel: true);
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
}
