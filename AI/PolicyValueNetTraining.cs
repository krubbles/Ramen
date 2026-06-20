namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public static class PolicyValueNetworkTraining
{
    public const int RolloutBatchSize = 64;

    static (List<GameState> trajectories, List<PolicyTrainingSample> trainingSamples, List<PpoStateData> stateData) GenerateRollout(IPolicyNetwork network, PpoTrainingSettings settings)
    {
        using var profileScope = ProfileScope.New(nameof(GenerateRollout));
        using PolicyNetworkAgent agent = new(network, ownsNetwork: false);
        List<PolicyTrainingSample> completedSamples = [];
        List<PpoStateData> stateData = [];
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
            PolicyTrainingSample[] stepSamples = agent.MakeMoveAndTrainingSample(gameStates, settings.UseSampledSoftmax, settings.SampledSoftmaxCount);
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
                int rolloutGameIndex = trajectoryGameStates.Count;
                int remainingSampleCount = settings.RolloutStateCount - completedSamples.Count;
                int samplesToAdd = Math.Min(remainingSampleCount, activeSamples[slot].Count);
                for (int sampleIndex = 0; sampleIndex < samplesToAdd; ++sampleIndex)
                {
                    PolicyTrainingSample sample = activeSamples[slot][sampleIndex];
                    stateData.Add(new(
                        GameInRolloutIndex: rolloutGameIndex,
                        MoveIndex: sampleIndex,
                        Advantage: sample.PolicyAdvantage.item<float>(),
                        ChosenMoveProb: MathF.Exp(sample.SamplingLogProb[0, 0].item<float>())));
                    completedSamples.Add(sample);
                }
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

        return (trajectoryGameStates, completedSamples, stateData);
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
        using var profileScope = ProfileScope.New(nameof(DoPPORollout));
        using var scope = NewDisposeScope();

        if (network is not Module networkModule)
            throw new InvalidOperationException($"{nameof(network)} must be a TorchSharp module to train.");

        (List<GameState> trajectories, List<PolicyTrainingSample> trainingSamples, List<PpoStateData> stateData) = GenerateRollout(network, settings);
        List<PpoStepData> stepData = [];
        List<Tensor> gradNormTensors = [];
        List<Tensor> entropyTensors = [];
        List<float> kldValues = [];
        double clippedSampleCount = 0;
        double totalRatioSampleCount = 0;
        double kldSum = 0;
        int kldCount = 0;
        bool stoppedEarly = false;

        using (ProfileScope.New("PpoUpdate"))
        {
            PolicyTrainingSample stackedSamples;
            using (ProfileScope.New("StackSamples"))
            {
                stackedSamples = TensorGroupExtentions.Stack(trainingSamples, disposeInputs: true, concat: true);
            }
            int sampleCount = trainingSamples.Count;
            for (int epoch = 0; epoch < settings.EpochCount && !stoppedEarly; ++epoch)
            {
                Tensor shuffledIndices;
                PolicyTrainingSample shuffled;
                using (ProfileScope.New("ShuffleSamples"))
                {
                    shuffledIndices = randperm(sampleCount, dtype: ScalarType.Int64, device: CPU);
                    shuffled = stackedSamples.IndexSelect(dim: 0, indices: shuffledIndices);
                }

                for (int batchStart = 0; batchStart < sampleCount && !stoppedEarly; batchStart += settings.BatchSize)
                {
                    using var batchScope = NewDisposeScope();

                    int batchEnd = Math.Min(batchStart + settings.BatchSize, sampleCount);
                    PolicyTrainingSample batch;
                    using (ProfileScope.New("GetPpoBatch"))
                    {
                        batch = shuffled.GetBatch(batchStart, batchEnd);
                    }
                    optimizer.zero_grad();


                    Tensor logProbs;
                    Tensor logPiNew;
                    Tensor logPiOld;
                    Tensor values;
                    using (ProfileScope.New("PpoForward"))
                    {
                        if (settings.UseSampledSoftmax)
                        {
                            Tensor logits;
                            (logits, values) = network.GetPolicyValue(batch.StateTensors, batch.MoveIndices);

                            Tensor chosenOldProb = batch.SamplingProb[TensorIndex.Colon, 0].to(logits.device).clamp(1e-9f, 1f - 1e-6f);
                            Tensor safeNegLogitSampleProbs = batch.SamplingProb[TensorIndex.Colon, 1..].to(logits.device).max(1e-9f);
                            Tensor oldNegativeProbabilityMass = (1f - chosenOldProb).max(1e-9f);

                            // Index zero always contains the selected move.
                            Tensor positiveLogit = logits[TensorIndex.Colon, 0];
                            Tensor negativeLogits = logits[TensorIndex.Colon, 1..];
                            Tensor adjustedNegativeLogits = negativeLogits
                                - log(safeNegLogitSampleProbs)
                                + log(oldNegativeProbabilityMass.unsqueeze(1))
                                - log(negativeLogits.size(dim: 1));
                            Tensor adjustedLogits = cat([positiveLogit.unsqueeze(1), adjustedNegativeLogits], dim: 1);

                            logProbs = functional.log_softmax(adjustedLogits, dim: 1);
                            logPiNew = logProbs.select(dim: 1, index: 0);
                            logPiOld = log(chosenOldProb);
                        }
                        else
                        {
                            Tensor logits;
                            (logits, values) = network.GetPolicyValue(batch.StateTensors);

                            logProbs = functional.log_softmax(logits, dim: 1);
                            Tensor chosenMoveIndices = batch.MoveIndices[TensorIndex.Colon, 0]
                                .to(logits.device)
                                .to_type(ScalarType.Int64)
                                .unsqueeze(1);
                            logPiNew = logProbs.gather(dim: 1, index: chosenMoveIndices).squeeze(1);
                            logPiOld = batch.SamplingLogProb[TensorIndex.Colon, 0].to(logPiNew.device);
                        }
                    }
                    Tensor kld;
                    Tensor clipCount;
                    Tensor entropy;
                    Tensor loss;
                    using (ProfileScope.New("PpoLoss"))
                    {
                        Tensor ratio = exp(logPiNew - logPiOld);
                        kld = (logPiOld - logPiNew).mean();

                        Tensor advantages = batch.PolicyAdvantage.to(logPiNew.device).reshape([-1]);
                        Tensor clippedRatio = clamp(ratio, 1f - settings.PpoEpsilon, 1f + settings.PpoEpsilon);
                        Tensor clipMask = (ratio - clippedRatio).abs().gt(0f);
                        clipCount = clipMask.to_type(ScalarType.Float32).sum();
                        totalRatioSampleCount += batchEnd - batchStart;
                        Tensor policyReward = min(ratio * advantages, clippedRatio * advantages).mean();
                        entropy = -(exp(logProbs) * logProbs).sum(dim: 1).mean();
                        Tensor policyLoss = -policyReward - settings.EntropyCoefficient * entropy;

                        Tensor valueTargets = batch.ValueTarget.to(values.device).reshape([-1]);
                        Tensor valueLoss = functional.mse_loss(values.reshape([-1]), valueTargets);
                        loss = policyLoss + settings.ValueLossCoefficient * valueLoss;
                    }

                    float kldValue;
                    using (ProfileScope.New("PpoScalarReadbacks"))
                    {
                        kldValue = kld.item<float>();
                        clippedSampleCount += clipCount.item<float>();
                    }
                    kldSum += kldValue;
                    kldCount++;

                    using (ProfileScope.New("PpoBackward"))
                    {
                        loss.backward();
                    }
                    Tensor gradNormTensor;
                    using (ProfileScope.New("GetGradNorm"))
                    {
                        gradNormTensor = GetGradNormTensor(networkModule).ToOuterScope();
                    }
                    gradNormTensors.Add(gradNormTensor);
                    entropyTensors.Add(entropy.ToOuterScope());
                    kldValues.Add(kldValue);

                    if (settings.KldEarlyStopThreshold > 0f && kldValue > settings.KldEarlyStopThreshold)
                    {
                        stoppedEarly = true;
                        continue;
                    }

                    if (settings.GradNormClip > 0f)
                    {
                        using (ProfileScope.New("GradClip"))
                        {
                            Tensor gradScale = min(ones_like(gradNormTensor), settings.GradNormClip / (gradNormTensor + 1e-6f));
                            foreach (Parameter parameter in networkModule.parameters())
                            {
                                Tensor grad = parameter.grad;
                                if (grad is not null)
                                    grad.mul_(gradScale);
                            }
                        }
                    }

                    using (ProfileScope.New("OptimizerStep"))
                    {
                        optimizer.step();
                    }
                }

                shuffled.Dispose();
                shuffledIndices.Dispose();
            }

            stackedSamples.Dispose();
        }

        if (gradNormTensors.Count > 0)
        {
            using (ProfileScope.New("GradNormReadback"))
            {
                Tensor stackedGradNorms = stack([.. gradNormTensors], dim: 0).to(CPU);
                Tensor stackedEntropies = stack([.. entropyTensors], dim: 0).to(CPU);
                float[] gradNormValues = [.. stackedGradNorms.data<float>()];
                float[] entropyValues = [.. stackedEntropies.data<float>()];
                for (int stepIndex = 0; stepIndex < gradNormValues.Length; ++stepIndex)
                {
                    stepData.Add(new(
                        GradNorm: gradNormValues[stepIndex],
                        Kld: kldValues[stepIndex],
                        Entropy: entropyValues[stepIndex]));
                }
            }
        }

        float clipRate = totalRatioSampleCount == 0 ? 0f : (float)(clippedSampleCount / totalRatioSampleCount);
        float averageKld = kldCount == 0 ? 0f : (float)(kldSum / kldCount);
        return new()
        {
            Trajectories = trajectories,
            StateData = stateData,
            StepData = stepData,
            ClipRate = clipRate,
            StoppedEarly = stoppedEarly,
            AverageKld = averageKld
        };
    }

    static Tensor GetGradNormTensor(Module networkModule)
    {
        using var dScope = NewDisposeScope();

        Tensor squaredNormSum = null;
        foreach (Parameter parameter in networkModule.parameters())
        {
            Tensor grad = parameter.grad;
            if (grad is null)
                continue;

            Tensor gradFloat = grad.detach().to_type(ScalarType.Float32);
            Tensor gradSquaredSum = (gradFloat * gradFloat).sum();
            squaredNormSum = squaredNormSum is null ? gradSquaredSum : squaredNormSum + gradSquaredSum;
        }

        if (squaredNormSum is null)
            squaredNormSum = tensor(0f);

        Tensor gradNorm = sqrt(squaredNormSum);
        gradNorm.ToOuterScope();
        return gradNorm;
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
    public bool UseSampledSoftmax = false;
    public int SampledSoftmaxCount = 40;
    public int EpochCount = 3;
    public int BatchSize = 256;
    public float LearningRate = 1e-5f;
    public float AdamBeta1 = 0.9f;
    public float AdamBeta2 = 0.97f;
    public float WeightDecay = 0.01f;
    public float PpoEpsilon = 0.2f;
    public float GradNormClip = 1.5f;
    public float KldEarlyStopThreshold = 0.5f;
    public float EntropyCoefficient = 0f;
    public float ValueLossCoefficient = 1f;
    public float AdvantageFalloff = 1f;

    public GameData GameData = GameData.Default;

    public PpoTrainingSettings() { }
}

public readonly record struct RolloutData(
    IReadOnlyList<GameState> Trajectories,
    IReadOnlyList<PpoStateData> StateData,
    IReadOnlyList<PpoStepData> StepData,
    float ClipRate,
    bool StoppedEarly,
    float AverageKld);

public readonly record struct PpoStateData(
    int GameInRolloutIndex,
    int MoveIndex,
    float Advantage,
    float ChosenMoveProb);

public readonly record struct PpoStepData(
    float GradNorm,
    float Kld,
    float Entropy);
