namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public static class PolicyValueNetworkTraining
{
    public const int RolloutBatchSize = 64;

    static (List<GameState> trajectories, List<PolicyTrainingSample> trainingSamples) GenerateRollout(IPolicyNetwork network, PpoTrainingSettings settings)
    {
        using PolicyNetworkAgent agent = new(network, ownsNetwork: false);
        List<PolicyTrainingSample> completedSamples = [];
        List<GameState> trajectoryGameStates = [];
        GameState[] gameStates = new GameState[RolloutBatchSize];
        List<PolicyTrainingSample>[] activeSamples = new List<PolicyTrainingSample>[RolloutBatchSize];

        for (int slot = 0; slot < RolloutBatchSize; ++slot)
        {
            gameStates[slot] = new(settings.GameData);
            activeSamples[slot] = [];
        }

        while (completedSamples.Count < settings.RolloutStateCount)
        {
            PolicyTrainingSample[] stepSamples = agent.MakeMoveAndTrainingSample(gameStates, settings.SampledSoftmaxCount);
            for (int slot = 0; slot < RolloutBatchSize; ++slot)
            {
                if (completedSamples.Count >= settings.RolloutStateCount)
                {
                    stepSamples[slot].Dispose();
                    continue;
                }

                activeSamples[slot].Add(stepSamples[slot]);

                if (!IsTrajectoryDone(gameStates[slot]))
                {
                    gameStates[slot].AdvanceToNextPlayerChoice();
                    if (!IsTrajectoryDone(gameStates[slot]))
                        continue;
                }

                float reward = GetReward(gameStates[slot]);
                SetTargetsAndAdvantages(activeSamples[slot], reward, settings.AdvantageFalloff);
                int remainingSampleCount = settings.RolloutStateCount - completedSamples.Count;
                int samplesToAdd = Math.Min(remainingSampleCount, activeSamples[slot].Count);
                for (int sampleIndex = 0; sampleIndex < samplesToAdd; ++sampleIndex)
                    completedSamples.Add(activeSamples[slot][sampleIndex]);
                for (int sampleIndex = samplesToAdd; sampleIndex < activeSamples[slot].Count; ++sampleIndex)
                    activeSamples[slot][sampleIndex].Dispose();
                trajectoryGameStates.Add(gameStates[slot]);

                gameStates[slot] = new(settings.GameData);
                activeSamples[slot] = [];
            }
        }

        for (int slot = 0; slot < RolloutBatchSize; ++slot)
        {
            for (int sampleIndex = 0; sampleIndex < activeSamples[slot].Count; ++sampleIndex)
                activeSamples[slot][sampleIndex].Dispose();
        }

        return (trajectoryGameStates, completedSamples);
    }

    static void SetTargetsAndAdvantages(List<PolicyTrainingSample> samples, float finalReward, float advantageFalloff)
    {
        float weightedSubsequentValueSum = finalReward;
        float subsequentWeightSum = 1f;

        for (int index = samples.Count - 1; index >= 0; --index)
        {
            PolicyTrainingSample sample = samples[index];
            float predictedValue = sample.Value.item<float>();
            float valueTarget = weightedSubsequentValueSum / subsequentWeightSum;
            float policyAdvantage = valueTarget - predictedValue;

            sample.ValueTarget?.Dispose();
            sample.PolicyAdvantage?.Dispose();
            sample.ValueTarget = tensor([valueTarget], device: CPU).DetachFromScope();
            sample.PolicyAdvantage = tensor([policyAdvantage], device: CPU).DetachFromScope();

            weightedSubsequentValueSum = predictedValue + advantageFalloff * weightedSubsequentValueSum;
            subsequentWeightSum = 1f + advantageFalloff * subsequentWeightSum;
        }
    }

    public static float GetReward(GameState gameState)
    {
        float winReward = IsFirstRoundWin(gameState) ? 1f : 0f;
        return winReward + 0.1f * gameState.HandState.RemainingHands;
    }

    static bool IsTrajectoryDone(GameState gameState)
    {
        return gameState.GameIsDone || IsFirstRoundWin(gameState);
    }

    static bool IsFirstRoundWin(GameState gameState)
    {
        return gameState.Round == 1 &&
            gameState.ScoringState.CurrentRoundTotalScore >= gameState.ScoringState.CurrentRoundThresholdScore;
    }

    public static RolloutData DoPPORollout(IPolicyNetwork network, PpoTrainingSettings settings)
    {
        using AdamW optimizer = BuildAdamWOptimizer(network, settings);
        return DoPPORollout(network, settings, optimizer);
    }

    public static RolloutData DoPPORollout(IPolicyNetwork network, PpoTrainingSettings settings, AdamW optimizer)
    {
        using var scope = NewDisposeScope();

        if (network is not Module networkModule)
            throw new InvalidOperationException($"{nameof(network)} must be a TorchSharp module to train.");

        (List<GameState> trajectories, List<PolicyTrainingSample> trainingSamples) = GenerateRollout(network, settings);
        List<float> gradNorms = [];
        double clippedSampleCount = 0;
        double totalRatioSampleCount = 0;

        PolicyTrainingSample stackedSamples = TensorGroupExtentions.Stack(trainingSamples, disposeInputs: true, concat: true);
        int sampleCount = trainingSamples.Count;
        for (int epoch = 0; epoch < settings.EpochCount; ++epoch)
        {
            Tensor shuffledIndices = randperm(sampleCount, dtype: ScalarType.Int64, device: CPU);
            PolicyTrainingSample shuffled = stackedSamples.IndexSelect(dim: 0, indices: shuffledIndices);

            for (int batchStart = 0; batchStart < sampleCount; batchStart += settings.BatchSize)
            {
                using var batchScope = NewDisposeScope();

                int batchEnd = Math.Min(batchStart + settings.BatchSize, sampleCount);
                PolicyTrainingSample batch = shuffled.GetBatch(batchStart, batchEnd);
                optimizer.zero_grad();



                (Tensor logits, Tensor values) = network.GetPolicyValue(batch.StateTensors, batch.MoveIndices);

                Tensor safeNegLogitSampleProbs = batch.SamplingProb[TensorIndex.Colon, 1..].max(1e-9f);

                // index zero always contains the selected move
                Tensor positiveLogit = logits[TensorIndex.Colon, 0];
                Tensor negativeLogits = logits[TensorIndex.Colon, 1..];

                Tensor adjustedNegativeLogits = negativeLogits
                    - log(safeNegLogitSampleProbs)
                    - log(negativeLogits.size(dim: 1));
                Tensor adjustedLogits = cat([positiveLogit.unsqueeze(1), adjustedNegativeLogits], dim: 1);

                Tensor logProbs = functional.log_softmax(adjustedLogits, dim: 1);
                Tensor logPiNew = logProbs.select(dim: 1, index: 0);
                Tensor logPiOld = log(batch.SamplingProb[TensorIndex.Colon, 0]);
                Tensor ratio = exp(logPiNew - logPiOld);

                Tensor advantages = batch.PolicyAdvantage.to(logits.device).reshape([-1]);
                Tensor clippedRatio = clamp(ratio, 1f - settings.PpoEpsilon, 1f + settings.PpoEpsilon);
                Tensor clipMask = (ratio - clippedRatio).abs().gt(0f);
                Tensor clipCount = clipMask.to_type(ScalarType.Float32).sum();
                clippedSampleCount += clipCount.item<float>();
                totalRatioSampleCount += batchEnd - batchStart;
                Tensor policyReward = min(ratio * advantages, clippedRatio * advantages).mean();
                Tensor entropy = -(exp(logProbs) * logProbs).sum(dim: 1).mean();
                Tensor policyLoss = -policyReward - settings.EntropyCoefficient * entropy;

                Tensor valueTargets = batch.ValueTarget.to(values.device).reshape([-1]);
                Tensor valueLoss = functional.mse_loss(values.reshape([-1]), valueTargets);
                Tensor loss = policyLoss + settings.ValueLossCoefficient * valueLoss;

                loss.backward();
                float gradNorm = GetGradNorm(networkModule);
                if (settings.GradNormClip > 0f && gradNorm > settings.GradNormClip)
                {
                    float gradScale = settings.GradNormClip / (gradNorm + 1e-6f);
                    foreach (Parameter parameter in networkModule.parameters())
                    {
                        Tensor grad = parameter.grad;
                        if (grad is not null)
                            grad.mul_(gradScale);
                    }
                }

                gradNorms.Add(gradNorm);
                optimizer.step();
            }

            shuffled.Dispose();
            shuffledIndices.Dispose();
        }

        stackedSamples.Dispose();
        float clipRate = totalRatioSampleCount == 0 ? 0f : (float)(clippedSampleCount / totalRatioSampleCount);
        return new() { Trajectories = trajectories, GradNorms = gradNorms, ClipRate = clipRate };
    }

    static float GetGradNorm(Module networkModule)
    {
        using var dScope = NewDisposeScope();

        double squaredNormSum = 0;
        foreach (Parameter parameter in networkModule.parameters())
        {
            Tensor grad = parameter.grad;
            if (grad is null)
                continue;

            Tensor gradFloat = grad.detach().to_type(ScalarType.Float32);
            Tensor gradSquaredSum = (gradFloat * gradFloat).sum();
            squaredNormSum += gradSquaredSum.item<float>();
        }

        return (float)Math.Sqrt(squaredNormSum);
    }

    public static AdamW BuildAdamWOptimizer(IPolicyNetwork network, PpoTrainingSettings settings)
    {
        if (network is not Module networkModule)
            throw new InvalidOperationException($"{nameof(network)} must be a TorchSharp module to train.");

        HashSet<Parameter> linearWeightDecayParameters = [];
        foreach ((string _, Module module) in networkModule.named_modules())
        {
            if (module is not TorchSharp.Modules.Linear)
                continue;

            foreach (Parameter parameter in module.parameters())
            {
                if (parameter.dim() > 1)
                    linearWeightDecayParameters.Add(parameter);
            }
        }

        List<Parameter> weightDecayParameters = [];
        List<Parameter> noWeightDecayParameters = [];
        foreach (Parameter parameter in networkModule.parameters())
        {
            if (linearWeightDecayParameters.Contains(parameter))
                weightDecayParameters.Add(parameter);
            else
                noWeightDecayParameters.Add(parameter);
        }

        return optim.AdamW(
            [
                new AdamW.ParamGroup(
                    weightDecayParameters,
                    lr: settings.LearningRate,
                    weight_decay: settings.WeightDecay,
                    beta1: settings.AdamBeta1,
                    beta2: settings.AdamBeta2),
                new AdamW.ParamGroup(
                    noWeightDecayParameters,
                    lr: settings.LearningRate,
                    weight_decay: 0,
                    beta1: settings.AdamBeta1,
                    beta2: settings.AdamBeta2),
            ],
            lr: settings.LearningRate,
            weight_decay: settings.WeightDecay,
            beta1: settings.AdamBeta1,
            beta2: settings.AdamBeta2);
    }
}

public struct PpoTrainingSettings
{
    public int RolloutStateCount = 1 << 16;
    public int SampledSoftmaxCount = 40;
    public int EpochCount = 3;
    public int BatchSize = 256;
    public float LearningRate = 1e-5f;
    public float AdamBeta1 = 0.9f;
    public float AdamBeta2 = 0.97f;
    public float WeightDecay = 0.01f;
    public float PpoEpsilon = 0.2f;
    public float GradNormClip = 30f;
    public float EntropyCoefficient = 0f;
    public float ValueLossCoefficient = 1f;
    public float AdvantageFalloff = 1f;

    public GameData GameData = GameData.Default;

    public PpoTrainingSettings() { }
}

public readonly record struct RolloutData(
    IReadOnlyList<GameState> Trajectories,
    IReadOnlyList<float> GradNorms,
    float ClipRate);
