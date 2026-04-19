namespace Ramen.ConsoleApp;

using System.Diagnostics;
using System.Globalization;
using System.Text;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using Ramen.Training;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TensorGroups = Ramen.AgentTools.TensorGroupExtentions;

public static class Program
{
    static readonly ExperimentConfig Config = new(
        ExperimentName: "2026-04-16_grpo_policy_unplayed_mean_suit256x64_scoreb4w16_clip0p3_ent0p01_b64k_lr1e4_e4_s200",
        CommitHash: "c6d1eaa",
        RolloutSampleCount: 65536,
        SampledSoftmaxCount: 10,
        GroupSize: 32,
        BatchSize: 65536,
        LearningRate: 1e-4f,
        Epochs: 4,
        Steps: 200,
        SnapshotFrequency: 5,
        ClipRange: 0.3f,
        EntropyCoeff: 0.01f,
        KldCoeff: 0f,
        AnalysisSampleSize: 300);

    static readonly ITrainingRunAnalyzer[] AnalysisAnalyzers =
    [
        new RewardStatsTrainingRunAnalyzer(),
        new PolicyEntropyTrainingRunAnalyzer(),
    ];

    public static void Main()
    {
        set_default_device(mps_is_available() ? MPS : CPU);
        Ramen.AI.TensorManager.Init();
        Profiling.CollectData = true;
        Console.WriteLine("=== START ===");

        string repoRoot = FindRepoRoot();
        string analysisDir = Path.Combine(repoRoot, "Analysis", Config.ExperimentName);
        Directory.CreateDirectory(analysisDir);

        string weightsRunName = $"{Config.ExperimentName}_weights";
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string notebookPath = Path.Combine(analysisDir, "analysis.ipynb");
        string readmePath = Path.Combine(analysisDir, "README.md");
        string copiedProgramPath = Path.Combine(analysisDir, "Program.cs");

        RunExperiment(
            config: Config,
            weightsRunName: weightsRunName,
            analysisCsvPath: analysisCsvPath);

        File.Copy(
            sourceFileName: Path.Combine(repoRoot, "ConsoleApp", "Program.cs"),
            destFileName: copiedProgramPath,
            overwrite: true);

        WriteReadme(
            filePath: readmePath,
            config: Config,
            weightsRunName: weightsRunName);

        WriteNotebook(
            filePath: notebookPath,
            csvPath: analysisCsvPath,
            experimentName: Config.ExperimentName);

        Console.WriteLine($"Experiment directory ready: {analysisDir}");
    }


    static void RunExperiment(ExperimentConfig config, string weightsRunName, string analysisCsvPath)
    {
        string weightsDir = GetTrainingRunWeightsDir(weightsRunName);
        Directory.CreateDirectory(weightsDir);
        DeleteExistingWeights(weightsDir);

        using UnplayedCardsPolicyModel model = new();
        AdamW optimizer = optim.AdamW(
            parameters: model.parameters(),
            lr: config.LearningRate,
            weight_decay: 0f,
            beta1: 0.9f,
            beta2: 0.99f);

        List<StepMetrics> stepMetrics = [];
        Dictionary<int, SnapshotMetrics> snapshotMetricsByStep = [];

        for (int step = 1; step <= config.Steps; ++step)
        {
            using ProfileScope trainStepProfile = ProfileScope.New("ExperimentTrainStep");

            Stopwatch rolloutStopwatch = Stopwatch.StartNew();
            IReadOnlyList<PolicyTrainingSample> trainingData = GenerateRolloutBatch(
                model: model,
                config: config);
            rolloutStopwatch.Stop();

            Stopwatch trainingStopwatch = Stopwatch.StartNew();
            PolicyOptimizationMetrics optimizationMetrics = TrainPolicyModelGrpo(
                model: model,
                optimizer: optimizer,
                trainingData: trainingData,
                config: config);
            trainingStopwatch.Stop();

            StepMetrics currentStepMetrics = new(
                Step: step,
                RolloutSeconds: GetElapsedSeconds(rolloutStopwatch),
                TrainingSeconds: GetElapsedSeconds(trainingStopwatch),
                TrainingLossMean: optimizationMetrics.TrainingLossMean,
                PolicyRewardMean: optimizationMetrics.PolicyRewardMean,
                TrainingEntropyMean: optimizationMetrics.TrainingEntropyMean,
                TrainingKldMean: optimizationMetrics.TrainingKldMean);
            stepMetrics.Add(currentStepMetrics);

            if (step % config.SnapshotFrequency == 0)
            {
                string filePath = Path.Combine(weightsDir, $"{step}.bin");
                model.Save(filePath);
                snapshotMetricsByStep[step] = SummarizeSnapshotWindow(
                    recentMetrics: stepMetrics,
                    windowSize: config.SnapshotFrequency);
            }

            Console.WriteLine(
                $"step {step}/{config.Steps} | rollout {currentStepMetrics.RolloutSeconds:F2}s | " +
                $"train {currentStepMetrics.TrainingSeconds:F2}s | loss {currentStepMetrics.TrainingLossMean:F4} | " +
                $"policy {currentStepMetrics.PolicyRewardMean:F4} | entropy {currentStepMetrics.TrainingEntropyMean:F4} | " +
                $"kld {currentStepMetrics.TrainingKldMean:F4}");

            Ramen.AI.TensorManager.DisposeAll();
            GC.Collect();
        }

        CSVBuilder baseAnalysis = TrainingRunAnalysis.Analyze(
            runName: weightsRunName,
            agentLoader: filePath => CreateAgent(filePath),
            sampleSize: config.AnalysisSampleSize,
            analyzers: AnalysisAnalyzers);

        CSVBuilder mergedOutput = MergeAnalysis(
            baseAnalysis: baseAnalysis,
            snapshotMetricsByStep: snapshotMetricsByStep);

        File.WriteAllText(analysisCsvPath, mergedOutput.ToString());
    }


    static IReadOnlyList<PolicyTrainingSample> GenerateRolloutBatch(UnplayedCardsPolicyModel model, ExperimentConfig config)
    {
        using ProfileScope rolloutProfile = ProfileScope.New("ExperimentRollout");
        return GRPOTrainingData.GenerateTrainingData(
            model: model,
            trainingSampleCount: config.RolloutSampleCount,
            sampledSoftmaxCount: config.SampledSoftmaxCount,
            groupSize: config.GroupSize);
    }


    static PolicyOptimizationMetrics TrainPolicyModelGrpo(
        UnplayedCardsPolicyModel model,
        AdamW optimizer,
        IReadOnlyList<PolicyTrainingSample> trainingData,
        ExperimentConfig config)
    {
        using var disposeScope = NewDisposeScope();
        using ProfileScope optimizeProfile = ProfileScope.New("ExperimentOptimize");

        PolicyTrainingSample stacked = TensorGroups.Stack(trainingData, disposeInputs: false, concat: true);
        DisposeTrainingSamples(trainingData);

        stacked.Advantage = stacked.Advantage.to(UnplayedCardsPolicyModel.EvalDevice);
        stacked.EntropyScalar = stacked.EntropyScalar.to(UnplayedCardsPolicyModel.EvalDevice);

        float trainingLossSum = 0f;
        float policyRewardSum = 0f;
        float trainingEntropySum = 0f;
        float trainingKldSum = 0f;
        int batchCount = 0;
        int sampleCount = trainingData.Count;

        for (int epoch = 0; epoch < config.Epochs; ++epoch)
        {
            using ProfileScope epochProfile = ProfileScope.New("ExperimentOptimizeEpoch");

            for (int start = 0; start < sampleCount; start += config.BatchSize)
            {
                using var batchScope = NewDisposeScope();
                using ProfileScope batchProfile = ProfileScope.New("ExperimentOptimizeBatch");

                optimizer.zero_grad();

                int end = Math.Min(start + config.BatchSize, sampleCount);
                PolicyTrainingSample batch = stacked.GetBatch(start, end);
                Tensor logits = model.GetPolicyLogits(batch.StateTensors, batch.UseHandTensors, batch.MoveIndices);

                float batchPolicyReward = 0f;
                float batchEntropy = 0f;
                float batchKld = 0f;
                Tensor loss = CalculateGrpoLoss(
                    logits: logits,
                    oldProbs: batch.SamplingProb,
                    advantage: batch.Advantage,
                    entropyScalar: batch.EntropyScalar,
                    config: config,
                    policyRewardMean: ref batchPolicyReward,
                    entropyMean: ref batchEntropy,
                    kldMean: ref batchKld);

                loss.backward();
                optimizer.step();

                trainingLossSum += loss.item<float>();
                policyRewardSum += batchPolicyReward;
                trainingEntropySum += batchEntropy;
                trainingKldSum += batchKld;
                batchCount++;

                batch.Dispose();
            }
        }

        stacked.Dispose();

        float averageDivisor = Math.Max(batchCount, 1);
        return new(
            TrainingLossMean: trainingLossSum / averageDivisor,
            PolicyRewardMean: policyRewardSum / averageDivisor,
            TrainingEntropyMean: trainingEntropySum / averageDivisor,
            TrainingKldMean: trainingKldSum / averageDivisor);
    }


    static Tensor CalculateGrpoLoss(
        Tensor logits,
        Tensor oldProbs,
        Tensor advantage,
        Tensor entropyScalar,
        ExperimentConfig config,
        ref float policyRewardMean,
        ref float entropyMean,
        ref float kldMean)
    {
        Tensor normalizedOldProbs = oldProbs / oldProbs.sum(dim: 1, keepdim: true).clamp_min(1e-9f);
        Tensor oldLogProbs = log(normalizedOldProbs.clamp_min(0f) + 1e-9f);
        Tensor newLogProbs = functional.log_softmax(logits, dim: 1);

        Tensor chosenOldLogProb = oldLogProbs.select(dim: 1, index: 0);
        Tensor chosenNewLogProb = newLogProbs.select(dim: 1, index: 0);
        Tensor ratioMinusOne = exp(chosenNewLogProb - chosenOldLogProb) - 1f;

        Tensor probs = exp(newLogProbs);
        Tensor entropy = (-(probs * newLogProbs).sum(dim: 1) * entropyScalar).mean();
        Tensor kld = (normalizedOldProbs * (oldLogProbs - newLogProbs)).sum(dim: 1).mean();

        Tensor unclippedReward = ratioMinusOne * advantage;
        Tensor clippedReward = clamp(ratioMinusOne, min: -config.ClipRange, max: config.ClipRange) * advantage;
        Tensor surrogateReward = min(unclippedReward, clippedReward).mean();
        Tensor loss = config.KldCoeff * kld - surrogateReward - config.EntropyCoeff * entropy;

        policyRewardMean = surrogateReward.item<float>();
        entropyMean = entropy.item<float>();
        kldMean = kld.item<float>();
        loss.MoveToOuterDisposeScope();
        return loss;
    }


    static SnapshotMetrics SummarizeSnapshotWindow(IReadOnlyList<StepMetrics> recentMetrics, int windowSize)
    {
        int startIndex = Math.Max(0, recentMetrics.Count - windowSize);
        float rolloutSecondsSum = 0f;
        float trainingSecondsSum = 0f;
        float trainingLossSum = 0f;
        float policyRewardSum = 0f;
        float trainingEntropySum = 0f;
        float trainingKldSum = 0f;
        int count = 0;

        for (int metricIndex = startIndex; metricIndex < recentMetrics.Count; ++metricIndex)
        {
            StepMetrics metric = recentMetrics[metricIndex];
            rolloutSecondsSum += metric.RolloutSeconds;
            trainingSecondsSum += metric.TrainingSeconds;
            trainingLossSum += metric.TrainingLossMean;
            policyRewardSum += metric.PolicyRewardMean;
            trainingEntropySum += metric.TrainingEntropyMean;
            trainingKldSum += metric.TrainingKldMean;
            count++;
        }

        float divisor = Math.Max(count, 1);
        return new(
            RolloutSecondsWindowMean: rolloutSecondsSum / divisor,
            TrainingSecondsWindowMean: trainingSecondsSum / divisor,
            TrainingLossWindowMean: trainingLossSum / divisor,
            PolicyRewardWindowMean: policyRewardSum / divisor,
            TrainingEntropyWindowMean: trainingEntropySum / divisor,
            TrainingKldWindowMean: trainingKldSum / divisor);
    }


    static CSVBuilder MergeAnalysis(CSVBuilder baseAnalysis, IReadOnlyDictionary<int, SnapshotMetrics> snapshotMetricsByStep)
    {
        List<Dictionary<string, string>> parsedRows = ParseCsv(baseAnalysis.ToString());
        parsedRows.Sort(static (left, right) => GetRowStep(left).CompareTo(GetRowStep(right)));

        CSVBuilder output = new();
        for (int rowIndex = 0; rowIndex < parsedRows.Count; ++rowIndex)
        {
            Dictionary<string, string> row = parsedRows[rowIndex];
            int step = GetRowStep(row);

            output.NextRow().SetCell("step", step);
            CopyParsedCsvCells(output, row);

            if (snapshotMetricsByStep.TryGetValue(step, out SnapshotMetrics snapshotMetrics))
            {
                output
                    .SetCell("rollout_seconds_window_mean", snapshotMetrics.RolloutSecondsWindowMean)
                    .SetCell("training_seconds_window_mean", snapshotMetrics.TrainingSecondsWindowMean)
                    .SetCell("training_loss_window_mean", snapshotMetrics.TrainingLossWindowMean)
                    .SetCell("policy_reward_window_mean", snapshotMetrics.PolicyRewardWindowMean)
                    .SetCell("training_entropy_window_mean", snapshotMetrics.TrainingEntropyWindowMean)
                    .SetCell("training_kld_window_mean", snapshotMetrics.TrainingKldWindowMean);
            }
        }

        return output;
    }


    static void CopyParsedCsvCells(CSVBuilder output, IReadOnlyDictionary<string, string> row)
    {
        foreach (KeyValuePair<string, string> pair in row)
        {
            if (pair.Key == "step")
                continue;

            if (float.TryParse(pair.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out float floatValue))
            {
                output.SetCell(pair.Key, floatValue);
                continue;
            }

            output.SetCell(pair.Key, pair.Value);
        }
    }


    static List<Dictionary<string, string>> ParseCsv(string csvText)
    {
        List<Dictionary<string, string>> rows = [];
        string[] lines = csvText
            .Split(
                separator: ['\r', '\n'],
                options: StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (lines.Length == 0)
            return rows;

        List<string> header = ParseCsvLine(lines[0]);
        for (int lineIndex = 1; lineIndex < lines.Length; ++lineIndex)
        {
            List<string> cells = ParseCsvLine(lines[lineIndex]);
            Dictionary<string, string> row = [];
            for (int columnIndex = 0; columnIndex < header.Count; ++columnIndex)
            {
                string value = columnIndex < cells.Count ? cells[columnIndex] : string.Empty;
                row[header[columnIndex]] = value;
            }
            rows.Add(row);
        }

        return rows;
    }


    static List<string> ParseCsvLine(string line)
    {
        List<string> cells = [];
        StringBuilder currentCell = new();
        bool inQuotes = false;

        for (int charIndex = 0; charIndex < line.Length; ++charIndex)
        {
            char current = line[charIndex];

            if (current == '"')
            {
                if (inQuotes && charIndex + 1 < line.Length && line[charIndex + 1] == '"')
                {
                    currentCell.Append('"');
                    charIndex++;
                }
                else
                    inQuotes = !inQuotes;

                continue;
            }

            if (current == ',' && !inQuotes)
            {
                cells.Add(currentCell.ToString());
                currentCell.Clear();
                continue;
            }

            currentCell.Append(current);
        }

        cells.Add(currentCell.ToString());
        return cells;
    }


    static int GetRowStep(IReadOnlyDictionary<string, string> row)
    {
        if (!row.TryGetValue("step", out string stepText))
            return 0;

        if (int.TryParse(stepText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int step))
            return step;

        return 0;
    }


    static PolicyOnlyAgent CreateAgent(string filePath)
    {
        UnplayedCardsPolicyModel model = new();
        model.Load(filePath);
        return new(model, ownsModel: true);
    }


    static void DisposeTrainingSamples(IReadOnlyList<PolicyTrainingSample> trainingData)
    {
        for (int sampleIndex = 0; sampleIndex < trainingData.Count; ++sampleIndex)
            trainingData[sampleIndex].Dispose();
    }


    static string GetTrainingRunWeightsDir(string runName)
    {
        return Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "Ramen",
            "Weights",
            runName);
    }


    static void DeleteExistingWeights(string weightsDir)
    {
        if (!Directory.Exists(weightsDir))
            return;

        string[] weightFiles = Directory.GetFiles(weightsDir, "*.bin");
        for (int fileIndex = 0; fileIndex < weightFiles.Length; ++fileIndex)
            File.Delete(weightFiles[fileIndex]);
    }


    static float GetElapsedSeconds(Stopwatch stopwatch)
    {
        return (float)stopwatch.Elapsed.TotalSeconds;
    }


    static string FindRepoRoot()
    {
        string currentDirectory = Directory.GetCurrentDirectory();
        DirectoryInfo directory = new(currentDirectory);

        while (directory is not null)
        {
            string solutionPath = Path.Combine(directory.FullName, "Ramen.sln");
            if (File.Exists(solutionPath))
                return directory.FullName;

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException("Could not find repo root containing Ramen.sln.");
    }


    static void WriteReadme(string filePath, ExperimentConfig config, string weightsRunName)
    {
        string weightsDir = GetTrainingRunWeightsDir(weightsRunName);
        string readme = $"""
Date: 2026-04-16
Commit Hash: {config.CommitHash}

# Training Params
1. Model: `UnplayedCardsPolicyModel`
2. Rollout training samples per step: `{config.RolloutSampleCount}`
3. Sampled-softmax move count per sample: `{config.SampledSoftmaxCount}`
4. Rollout group size: `{config.GroupSize}`
5. Optimizer batch size: `{config.BatchSize}`
6. Learning rate: `{config.LearningRate.ToString("0.0e0", CultureInfo.InvariantCulture)}`
7. Epochs per step: `{config.Epochs}`
8. Training steps: `{config.Steps}`
9. Weight snapshot frequency: every `{config.SnapshotFrequency}` steps
10. PPO clip range: `{config.ClipRange}`
11. Entropy coefficient: `{config.EntropyCoeff}`
12. KLD coefficient: `{config.KldCoeff}`
13. Training-run analysis sample size: `{config.AnalysisSampleSize}`
14. Output CSV: `analysis.csv`
15. Snapshot weights directory: `{weightsDir}`

# Description
- This run trains a policy-only GRPO model that embeds the original full hand, remaining deck, and per-move unplayed cards as suit-wise rank-count tensors processed by `13 -> 256 -> GELU -> 64`, then mean-pooled across suits.
- Score features use the existing threshold score embedder with `4` buckets and width `16`, using the repo's normalized score scale where `1.0` corresponds to the `300` chip target.
- Hands and discards are represented as one-hot counts for hands and discards separately; the move trunk includes both action-specific post-count one-hots so the shared residual stream can emit both hand and discard logits from one `Nx2` head.
- The GRPO rollout path reuses the existing sampled-softmax data generation, while the optimization loop in `Program.cs` mirrors the repo trainer but changes clipping to `0.3` and keeps `kld=0`, `entropy=0.01`.
- `analysis.csv` contains one row per saved checkpoint with reward/entropy from a single `TrainingRunAnalysis.Analyze(...)` call, merged with rolling 5-step timing and optimization metrics for rollout and training stages.
- Assumption: the requested second suit-count projection was implemented as `256 -> 64`; the prompt said `128 -> 64`, which would not match the preceding `13 -> 256` layer without another hidden projection.
""";
        File.WriteAllText(filePath, readme);
    }


    static void WriteNotebook(string filePath, string csvPath, string experimentName)
    {
        string notebook = $$"""
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# {{experimentName}}\n",
    "\n",
    "Plots reward, entropy, and training-stage timings from `analysis.csv`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "import csv\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "csv_path = Path(r\"{{csvPath}}\")\n",
    "with csv_path.open(newline=\"\") as f:\n",
    "    rows = list(csv.DictReader(f))\n",
    "\n",
    "steps = [int(row[\"step\"]) for row in rows]\n",
    "reward_mean = [float(row[\"reward_mean\"]) for row in rows]\n",
    "policy_entropy = [float(row[\"policy_entropy_mean\"]) for row in rows]\n",
    "rollout_seconds = [float(row[\"rollout_seconds_window_mean\"]) for row in rows]\n",
    "training_seconds = [float(row[\"training_seconds_window_mean\"]) for row in rows]\n",
    "training_loss = [float(row[\"training_loss_window_mean\"]) for row in rows]\n",
    "training_entropy = [float(row[\"training_entropy_window_mean\"]) for row in rows]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "figure, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
    "\n",
    "axes[0, 0].plot(steps, reward_mean, marker=\"o\")\n",
    "axes[0, 0].set_title(\"Average reward\")\n",
    "axes[0, 0].set_xlabel(\"Step\")\n",
    "axes[0, 0].set_ylabel(\"Reward\")\n",
    "\n",
    "axes[0, 1].plot(steps, policy_entropy, marker=\"o\", color=\"tab:orange\")\n",
    "axes[0, 1].set_title(\"Policy entropy\")\n",
    "axes[0, 1].set_xlabel(\"Step\")\n",
    "axes[0, 1].set_ylabel(\"Entropy\")\n",
    "\n",
    "axes[1, 0].plot(steps, rollout_seconds, marker=\"o\", label=\"rollout\")\n",
    "axes[1, 0].plot(steps, training_seconds, marker=\"o\", label=\"training\")\n",
    "axes[1, 0].set_title(\"Mean stage seconds over the last 5 steps\")\n",
    "axes[1, 0].set_xlabel(\"Step\")\n",
    "axes[1, 0].set_ylabel(\"Seconds\")\n",
    "axes[1, 0].legend()\n",
    "\n",
    "axes[1, 1].plot(steps, training_loss, marker=\"o\", label=\"loss\")\n",
    "axes[1, 1].plot(steps, training_entropy, marker=\"o\", label=\"train entropy\")\n",
    "axes[1, 1].set_title(\"Optimization metrics\")\n",
    "axes[1, 1].set_xlabel(\"Step\")\n",
    "axes[1, 1].legend()\n",
    "\n",
    "figure.tight_layout()\n",
    "figure"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
""";
        File.WriteAllText(filePath, notebook);
    }
}

public readonly record struct ExperimentConfig
(
    string ExperimentName,
    string CommitHash,
    int RolloutSampleCount,
    int SampledSoftmaxCount,
    int GroupSize,
    int BatchSize,
    float LearningRate,
    int Epochs,
    int Steps,
    int SnapshotFrequency,
    float ClipRange,
    float EntropyCoeff,
    float KldCoeff,
    int AnalysisSampleSize
);

public readonly record struct StepMetrics
(
    int Step,
    float RolloutSeconds,
    float TrainingSeconds,
    float TrainingLossMean,
    float PolicyRewardMean,
    float TrainingEntropyMean,
    float TrainingKldMean
);

public readonly record struct SnapshotMetrics
(
    float RolloutSecondsWindowMean,
    float TrainingSecondsWindowMean,
    float TrainingLossWindowMean,
    float PolicyRewardWindowMean,
    float TrainingEntropyWindowMean,
    float TrainingKldWindowMean
);

public readonly record struct PolicyOptimizationMetrics
(
    float TrainingLossMean,
    float PolicyRewardMean,
    float TrainingEntropyMean,
    float TrainingKldMean
);

public sealed class SuitCountMeanEmbedding : Module<Tensor, Tensor>
{
    readonly Linear _inputProjection = Linear(13, 256, device: UnplayedCardsPolicyModel.EvalDevice);
    readonly GELU _activation = GELU();
    readonly Linear _outputProjection = Linear(256, 64, device: UnplayedCardsPolicyModel.EvalDevice);

    public SuitCountMeanEmbedding() : base(nameof(SuitCountMeanEmbedding))
    {
        RegisterComponents();
    }


    public override Tensor forward(Tensor suitRankCounts)
    {
        using var scope = NewDisposeScope();
        Tensor projected = _inputProjection.forward(suitRankCounts);
        Tensor activated = _activation.forward(projected);
        Tensor output = _outputProjection.forward(activated);
        Tensor pooled = output.mean([output.Dimensions - 2]);
        pooled.MoveToOuterDisposeScope();
        return pooled;
    }
}

public sealed class InterpolatedScoreEmbedding : Module<Tensor, Tensor>
{
    readonly Embedding _bucketEmbeddings;
    readonly float _threshold;
    readonly int _bucketCount;

    public InterpolatedScoreEmbedding(float threshold, int bucketCount, int embeddingWidth) : base(nameof(InterpolatedScoreEmbedding))
    {
        _threshold = threshold;
        _bucketCount = bucketCount;
        _bucketEmbeddings = Embedding(bucketCount, embeddingWidth, device: UnplayedCardsPolicyModel.EvalDevice);
        RegisterComponents();
    }


    public override Tensor forward(Tensor score)
    {
        using var scope = NewDisposeScope();

        Tensor relativeScore = (score.to_type(ScalarType.Float32) / _threshold).clamp(0f, 1f);
        Tensor bucketPosition = relativeScore * (_bucketCount - 1);
        Tensor lowerIndex = bucketPosition.floor().to_type(ScalarType.Int64);
        Tensor upperIndex = (lowerIndex + 1).clamp_max(_bucketCount - 1);
        Tensor upperWeight = (bucketPosition - lowerIndex.to_type(ScalarType.Float32)).unsqueeze(-1);
        Tensor lowerWeight = 1f - upperWeight;

        Tensor lowerEmbedding = _bucketEmbeddings.forward(lowerIndex);
        Tensor upperEmbedding = _bucketEmbeddings.forward(upperIndex);
        Tensor result = lowerEmbedding * lowerWeight + upperEmbedding * upperWeight;
        result.MoveToOuterDisposeScope();
        return result;
    }
}

public sealed class GeluResidualBlock : Module<Tensor, Tensor>
{
    readonly GELU _inputActivation = GELU();
    readonly Linear _upProjection;
    readonly GELU _hiddenActivation = GELU();
    readonly Linear _downProjection;

    public GeluResidualBlock(int width) : base(nameof(GeluResidualBlock))
    {
        _upProjection = Linear(width, width, device: UnplayedCardsPolicyModel.EvalDevice);
        _downProjection = Linear(width, width, device: UnplayedCardsPolicyModel.EvalDevice);
        RegisterComponents();
    }


    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();
        Tensor hidden = _inputActivation.forward(input);
        hidden = _upProjection.forward(hidden);
        hidden = _hiddenActivation.forward(hidden);
        Tensor output = input + _downProjection.forward(hidden);
        output.MoveToOuterDisposeScope();
        return output;
    }
}

public sealed class UnplayedCardsPolicyModel : Module, IPolicyModel
{
    public static readonly Device EvalDevice = PolicyModel.EvalDevice;

    public const int CardSetEmbeddingWidth = 64;
    public const int ScoreEmbeddingWidth = 16;
    public const int ScoreBucketCount = 4;
    public const int CountOneHotWidth = 5;
    public const int MoveFeatureWidth = ScoreEmbeddingWidth * 2 + CountOneHotWidth * 6 + CardSetEmbeddingWidth * 3;
    public const int UseableHandCount = 218;

    readonly InterpolatedScoreEmbedding _scoreEmbedding = new(
        threshold: 1f,
        bucketCount: ScoreBucketCount,
        embeddingWidth: ScoreEmbeddingWidth);
    readonly SuitCountMeanEmbedding _cardSetEmbedding = new();
    readonly GeluResidualBlock _residualBlock = new(MoveFeatureWidth);
    readonly GELU _finalActivation = GELU();
    readonly Linear _outputProjection = Linear(MoveFeatureWidth, 2, device: EvalDevice);
    readonly Tensor _remainingHandMask;

    public UnplayedCardsPolicyModel() : base(nameof(UnplayedCardsPolicyModel))
    {
        _remainingHandMask = tensor(
            BuildRemainingHandMaskData(),
            dtype: ScalarType.Float32,
            device: EvalDevice);
        _remainingHandMask.DetachFromScope();
        Ramen.AI.TensorManager.PersistForever(_remainingHandMask);
        RegisterComponents();
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        using var scope = NewDisposeScope();

        Profiling.Enter("ExperimentPolicyFeatureBuild");
        Tensor moveFeatures = BuildMoveFeatures(gameStateTensors, useHandTensors);
        Profiling.Exit("ExperimentPolicyFeatureBuild");

        Profiling.Enter("ExperimentPolicyHead");
        Tensor residualStream = _residualBlock.forward(moveFeatures);
        Tensor moveLogits = _outputProjection.forward(_finalActivation.forward(residualStream));
        Tensor flattenedLogits = moveLogits.view([moveLogits.size(0), moveLogits.size(1) * moveLogits.size(2)]);
        Profiling.Exit("ExperimentPolicyHead");

        flattenedLogits.MoveToOuterDisposeScope();
        return flattenedLogits;
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        Tensor selectedMoveIndices = moveIndices.to_type(ScalarType.Int64).to(EvalDevice);
        Tensor handIndices = selectedMoveIndices.div(2).to_type(ScalarType.Int64);
        Tensor actionIndices = selectedMoveIndices.remainder(2).to_type(ScalarType.Int64);

        Profiling.Enter("ExperimentPolicyFeatureBuild");
        Tensor moveFeatures = BuildMoveFeatures(
            gameStateTensors: gameStateTensors,
            useHandTensors: useHandTensors,
            selectedHandIndices: handIndices);
        Profiling.Exit("ExperimentPolicyFeatureBuild");

        Profiling.Enter("ExperimentPolicyHead");
        Tensor residualStream = _residualBlock.forward(moveFeatures);
        Tensor moveLogits = _outputProjection.forward(_finalActivation.forward(residualStream));
        Tensor gatheredLogits = moveLogits.gather(
            dim: 2,
            index: actionIndices.unsqueeze(-1)).squeeze(-1);
        Profiling.Exit("ExperimentPolicyHead");

        gatheredLogits.MoveToOuterDisposeScope();
        return gatheredLogits;
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }


    Tensor BuildMoveFeatures(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        using var scope = NewDisposeScope();

        Tensor fullHand = gameStateTensors.FullHand.to(EvalDevice);
        Tensor remainingDeck = gameStateTensors.RemainingDeck.to(EvalDevice);
        Tensor preScore = gameStateTensors.Score.to(EvalDevice);
        Tensor postPlayScore = useHandTensors.Score.to(EvalDevice);
        Tensor handsAndDiscards = gameStateTensors.HandsAndDiscards.to(EvalDevice).to_type(ScalarType.Int64);

        Tensor fullHandPerCardCounts = GetPerCardSuitRankCounts(fullHand);
        Tensor fullHandEmbedding = _cardSetEmbedding.forward(fullHandPerCardCounts.sum(dim: 1));
        Tensor remainingDeckEmbedding = _cardSetEmbedding.forward(GetPerCardSuitRankCounts(remainingDeck).sum(dim: 1));
        Tensor remainingHandCounts = einsum("bcsr,mc->bmsr", fullHandPerCardCounts, _remainingHandMask);
        Tensor remainingHandEmbedding = _cardSetEmbedding.forward(remainingHandCounts);

        int moveCount = (int)postPlayScore.size(1);
        Tensor preScoreEmbedding = ExpandAcrossMoves(_scoreEmbedding.forward(preScore).squeeze(1), moveCount);
        Tensor postPlayScoreEmbedding = _scoreEmbedding.forward(postPlayScore);

        Tensor preHands = handsAndDiscards.div(CountOneHotWidth);
        Tensor preDiscards = handsAndDiscards.remainder(CountOneHotWidth);
        Tensor postPlayHands = (preHands - 1).clamp_min(0);
        Tensor postDiscardDiscards = (preDiscards - 1).clamp_min(0);

        Tensor moveFeatures = concat(
            [
                preScoreEmbedding,
                postPlayScoreEmbedding,
                ExpandAcrossMoves(ToOneHot(preHands, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(preDiscards, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(postPlayHands, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(preDiscards, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(preHands, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(postDiscardDiscards, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(fullHandEmbedding, moveCount),
                remainingHandEmbedding,
                ExpandAcrossMoves(remainingDeckEmbedding, moveCount),
            ],
            dim: -1);
        moveFeatures.MoveToOuterDisposeScope();
        return moveFeatures;
    }


    Tensor BuildMoveFeatures(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor selectedHandIndices)
    {
        using var scope = NewDisposeScope();

        Tensor fullHand = gameStateTensors.FullHand.to(EvalDevice);
        Tensor remainingDeck = gameStateTensors.RemainingDeck.to(EvalDevice);
        Tensor preScore = gameStateTensors.Score.to(EvalDevice);
        Tensor postPlayScore = useHandTensors.Score.to(EvalDevice);
        Tensor handsAndDiscards = gameStateTensors.HandsAndDiscards.to(EvalDevice).to_type(ScalarType.Int64);

        Tensor fullHandPerCardCounts = GetPerCardSuitRankCounts(fullHand);
        Tensor fullHandEmbedding = _cardSetEmbedding.forward(fullHandPerCardCounts.sum(dim: 1));
        Tensor remainingDeckEmbedding = _cardSetEmbedding.forward(GetPerCardSuitRankCounts(remainingDeck).sum(dim: 1));

        int batchSize = (int)fullHand.size(0);
        int moveCount = (int)selectedHandIndices.size(1);

        Tensor selectedMask = _remainingHandMask
            .index_select(dim: 0, index: selectedHandIndices.view(-1))
            .view(batchSize, moveCount, GameData.HandSize);
        Tensor remainingHandCounts = einsum("bcsr,bmc->bmsr", fullHandPerCardCounts, selectedMask);
        Tensor remainingHandEmbedding = _cardSetEmbedding.forward(remainingHandCounts);

        Tensor selectedPostPlayScore = postPlayScore.gather(dim: 1, index: selectedHandIndices);
        Tensor preScoreEmbedding = ExpandAcrossMoves(_scoreEmbedding.forward(preScore).squeeze(1), moveCount);
        Tensor postPlayScoreEmbedding = _scoreEmbedding.forward(selectedPostPlayScore);

        Tensor preHands = handsAndDiscards.div(CountOneHotWidth);
        Tensor preDiscards = handsAndDiscards.remainder(CountOneHotWidth);
        Tensor postPlayHands = (preHands - 1).clamp_min(0);
        Tensor postDiscardDiscards = (preDiscards - 1).clamp_min(0);

        Tensor moveFeatures = concat(
            [
                preScoreEmbedding,
                postPlayScoreEmbedding,
                ExpandAcrossMoves(ToOneHot(preHands, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(preDiscards, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(postPlayHands, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(preDiscards, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(preHands, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(ToOneHot(postDiscardDiscards, CountOneHotWidth), moveCount),
                ExpandAcrossMoves(fullHandEmbedding, moveCount),
                remainingHandEmbedding,
                ExpandAcrossMoves(remainingDeckEmbedding, moveCount),
            ],
            dim: -1);
        moveFeatures.MoveToOuterDisposeScope();
        return moveFeatures;
    }


    Tensor GetPerCardSuitRankCounts(Tensor cardSet)
    {
        using var scope = NewDisposeScope();

        Tensor cardIndices = cardSet.to_type(ScalarType.Int64);
        Tensor validCards = cardIndices.gt(0).to_type(ScalarType.Float32).unsqueeze(-1).unsqueeze(-1);
        Tensor rankIndices = (cardIndices - 1).clamp_min(0).remainder(Card.RankCount).to_type(ScalarType.Int64);
        Tensor suitIndices = (cardIndices - 1).clamp_min(0).div(Card.RankCount).to_type(ScalarType.Int64);

        Tensor rankOneHot = functional.one_hot(rankIndices, Card.RankCount).to_type(ScalarType.Float32);
        Tensor suitOneHot = functional.one_hot(suitIndices, Card.SuitCount).to_type(ScalarType.Float32);

        Tensor perCardSuitRankCounts = suitOneHot.unsqueeze(-1) * rankOneHot.unsqueeze(-2) * validCards;
        perCardSuitRankCounts.MoveToOuterDisposeScope();
        return perCardSuitRankCounts;
    }


    Tensor ExpandAcrossMoves(Tensor tensorToExpand, int moveCount)
    {
        using var scope = NewDisposeScope();

        Tensor expanded = tensorToExpand
            .unsqueeze(1)
            .expand(tensorToExpand.size(0), moveCount, tensorToExpand.size(1));
        expanded.MoveToOuterDisposeScope();
        return expanded;
    }


    Tensor ToOneHot(Tensor indices, int width)
    {
        using var scope = NewDisposeScope();

        Tensor oneHot = functional.one_hot(indices.to_type(ScalarType.Int64), width).to_type(ScalarType.Float32);
        oneHot.MoveToOuterDisposeScope();
        return oneHot;
    }


    static float[,] BuildRemainingHandMaskData()
    {
        float[,] mask = new float[UseableHandCount, GameData.HandSize];
        int[][] combinations = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: GameData.MaxPlayedHandSize);

        for (int handIndex = 0; handIndex < combinations.Length; ++handIndex)
        {
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                mask[handIndex, cardIndex] = 1f;

            int[] playedCards = combinations[handIndex];
            for (int playedCardIndex = 0; playedCardIndex < playedCards.Length; ++playedCardIndex)
                mask[handIndex, playedCards[playedCardIndex]] = 0f;
        }

        return mask;
    }
}
