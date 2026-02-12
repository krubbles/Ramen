namespace Ramen.ConsoleApp;

using System;
using System.IO;
using System.Threading;
using Ramen.AI;
using Ramen.Training;
using static TorchSharp.torch;

public static class Program
{
    public static void Main()
    {
        // Do not change START
        TorchSharp.torch.set_default_device(MPS);
        Ramen.AI.TensorManager.Init(); // gc system for tensors.
        Console.WriteLine("=== START ===");
        // Do not change END

        string baseRunName = "2026-02-12_grpo_s200_r5000_lr2e-4_ent0p01";
        string experimentName = "2026-02-12_grpo_continue_s230_from200_r5000_lr5e-5_ent0";

        int startingStep = 230;
        int additionalSteps = 100;
        int rolloutSize = 5000;
        int samplingFrequency = 5;
        int analysisSampleSize = 512;
        float learningRate = 5e-5f;
        float entropyCoeff = 0f;

        string repoRoot = FindRepoRoot();
        string analysisDir = Path.Combine(repoRoot, "Analysis", experimentName);
        Directory.CreateDirectory(analysisDir);

        string baseWeightsDir = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "Ramen",
            "Weights",
            baseRunName);
        string resumeSnapshotPath = Path.Combine(baseWeightsDir, $"{startingStep}.bin");

        if (!File.Exists(resumeSnapshotPath))
            throw new FileNotFoundException($"Missing resume snapshot: {resumeSnapshotPath}");

        RunContinuedTraining(
            runName: baseRunName,
            resumeSnapshotPath: resumeSnapshotPath,
            startingStep: startingStep,
            additionalSteps: additionalSteps,
            rolloutSize: rolloutSize,
            samplingFrequency: samplingFrequency,
            learningRate: learningRate,
            entropyCoeff: entropyCoeff);

        CSVBuilder analysisOutput = TrainingRunAnalysis.Analyze(
            runName: baseRunName,
            sampleSize: analysisSampleSize,
            new RewardStatsTrainingRunAnalyzer(),
            new PolicyEntropyTrainingRunAnalyzer(),
            new HandTypePresenceTrainingRunAnalyzer(),
            new EndStateHandCountTrainingRunAnalyzer());

        string analysisPath = Path.Combine(analysisDir, "training_analysis.csv");
        File.WriteAllText(analysisPath, analysisOutput.ToString());
        Console.WriteLine($"Wrote analysis CSV: {analysisPath}");
    }

    static void RunContinuedTraining(string runName, string resumeSnapshotPath, int startingStep, int additionalSteps, int rolloutSize, int samplingFrequency, float learningRate, float entropyCoeff)
    {
        BasicGRPOTrainingRun baseRun = new()
        {
            RolloutSize = rolloutSize,
            LearningRate = learningRate,
            Entropy = entropyCoeff,
        };

        ResumeFromSnapshotTrainingRun resumedRun = new(baseRun, resumeSnapshotPath);

        using CancellationTokenSource cancellationTokenSource = new();
        ITrainingRun.Run(
            trainingRun: resumedRun,
            runName: runName,
            steps: additionalSteps,
            samplingFrequency: samplingFrequency,
            cancellationTokenSource: cancellationTokenSource,
            startingStep: startingStep).Wait();
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

    sealed class ResumeFromSnapshotTrainingRun : ITrainingRun
    {
        readonly ITrainingRun _innerRun;
        readonly string _resumeSnapshotPath;
        bool _hasLoadedSnapshot;

        public ResumeFromSnapshotTrainingRun(ITrainingRun innerRun, string resumeSnapshotPath)
        {
            _innerRun = innerRun;
            _resumeSnapshotPath = resumeSnapshotPath;
            _hasLoadedSnapshot = false;
        }

        public void Step(IPolicyModel model)
        {
            if (!_hasLoadedSnapshot)
            {
                model.Load(_resumeSnapshotPath);
                _hasLoadedSnapshot = true;
            }

            _innerRun.Step(model);
        }
    }
}
