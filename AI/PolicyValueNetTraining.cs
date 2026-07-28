namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public static class PolicyValueNetworkTraining
{
    public const int RolloutBatchSize = 128;

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

        Profiling.PhaseSuffix = "_Rollout";
        (List<GameState> trajectories, List<PolicyTrainingSample> trainingSamples, List<PpoStateData> stateData) = GenerateRollout(network, settings);
        Profiling.PhaseSuffix = "";
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
            int totalBatchCount = (sampleCount + settings.BatchSize - 1) / settings.BatchSize;
            int accumulationSteps = Math.Max(1, settings.GradientAccumulationSteps);
            // Highest KLD seen among the batches of the current accumulation group, so the
            // per-step diagnostic still reflects every batch that fed into it.
            float groupMaxKld = 0f;
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

                    // Gradients accumulate across a group of batches and are applied once
                    // at its end. The final group of an epoch may be short, so the group's
                    // real size is used to scale the loss rather than the configured one.
                    int batchIndex = batchStart / settings.BatchSize;
                    int groupStartBatch = batchIndex - (batchIndex % accumulationSteps);
                    int groupBatchCount = Math.Min(accumulationSteps, totalBatchCount - groupStartBatch);
                    bool isFirstBatchInGroup = batchIndex == groupStartBatch;
                    bool isLastBatchInGroup = batchIndex == groupStartBatch + groupBatchCount - 1;

                    if (isFirstBatchInGroup)
                    {
                        optimizer.zero_grad();
                        groupMaxKld = 0f;
                    }


                    Tensor logProbs;
                    Tensor sampledLogitsForDistillation = null;
                    Tensor logPiNew;
                    Tensor logPiOld;
                    Tensor values;
                    using (ProfileScope.New("PpoForward"))
                    {
                        if (network is IAuxiliaryLossFreeLoadBalancedNetwork loadBalancedNetwork)
                            loadBalancedNetwork.UpdateExpertLoadBalance = true;
                        Profiling.PhaseSuffix = "_Train";
                        if (settings.UseSampledSoftmax && batch.ExactMass is not null)
                        {
                            // Two-tier layout: [candidates | chosen-if-outside | negatives].
                            // The exact block needs no correction; the negatives are drawn from
                            // its complement with q(a) = pi_old(a) / (1 - ExactMass), so their
                            // correction is -log pi_old(a) + log(1 - ExactMass) - log(count).
                            NewPolicyNetwork cascadeNetwork = (NewPolicyNetwork)network;
                            int candidateCount = cascadeNetwork.CascadeCandidateCount;
                            int exactCount = candidateCount + 1;

                            Tensor slotLogits;
                            (slotLogits, values) = cascadeNetwork.GetCascadePolicyValue(
                                batch.StateTensors, batch.MoveIndices, candidateCount);
                            sampledLogitsForDistillation = slotLogits;

                            int negativeCount = (int)slotLogits.size(1) - exactCount;
                            Tensor slotValid = batch.SlotValidMask.to(slotLogits.device);
                            Tensor exactMass = batch.ExactMass.to(slotLogits.device).clamp(1e-9f, 1f - 1e-9f);
                            Tensor slotProbs = batch.SamplingProb.to(slotLogits.device);

                            Tensor exactLogits = slotLogits.narrow(1, 0, exactCount);
                            // A slot carrying no move must not contribute to the normalizer.
                            exactLogits = where(
                                slotValid.narrow(1, 0, exactCount).gt(0.5f),
                                exactLogits,
                                full_like(exactLogits, PolicyLogitMask.IllegalMoveLogit));

                            Tensor negativeLogits = slotLogits.narrow(1, exactCount, negativeCount);
                            Tensor negativeSampleProbs = slotProbs.narrow(1, exactCount, negativeCount).max(1e-9f);
                            Tensor adjustedNegativeLogits = negativeLogits
                                - log(negativeSampleProbs)
                                + log(1f - exactMass)
                                - MathF.Log(negativeCount);

                            Tensor adjustedLogits = cat([exactLogits, adjustedNegativeLogits], dim: 1);
                            Tensor slotLogProbs = functional.log_softmax(adjustedLogits, dim: 1);

                            Tensor chosenSlot = batch.ChosenSlotIndex.to(slotLogits.device).to_type(ScalarType.Int64);
                            logPiNew = slotLogProbs.gather(dim: 1, index: chosenSlot).select(dim: 1, index: 0);
                            logPiOld = log(slotProbs.gather(dim: 1, index: chosenSlot).clamp(1e-9f, 1f).select(dim: 1, index: 0));
                            logProbs = null;
                        }
                        else if (settings.UseSampledSoftmax)
                        {
                            Tensor sampledLogits;
                            (sampledLogits, values) = network.GetPolicyValue(batch.StateTensors, batch.MoveIndices);
                            sampledLogitsForDistillation = sampledLogits;

                            Tensor chosenOldProb = batch.SamplingProb[TensorIndex.Colon, 0].to(sampledLogits.device).clamp(1e-9f, 1f - 1e-6f);
                            Tensor safeNegLogitSampleProbs = batch.SamplingProb[TensorIndex.Colon, 1..].to(sampledLogits.device).max(1e-9f);
                            Tensor oldNegativeProbabilityMass = (1f - chosenOldProb).max(1e-9f);

                            // Index zero always contains the selected move.
                            Tensor positiveLogit = sampledLogits[TensorIndex.Colon, 0];
                            Tensor negativeLogits = sampledLogits[TensorIndex.Colon, 1..];
                            Tensor adjustedNegativeLogits = negativeLogits
                                - log(safeNegLogitSampleProbs)
                                + log(oldNegativeProbabilityMass.unsqueeze(1))
                                - log(negativeLogits.size(dim: 1));
                            Tensor adjustedLogits = cat([positiveLogit.unsqueeze(1), adjustedNegativeLogits], dim: 1);

                            Tensor sampledLogProbs = functional.log_softmax(adjustedLogits, dim: 1);
                            logPiNew = sampledLogProbs.select(dim: 1, index: 0);
                            logPiOld = log(chosenOldProb);
                            logProbs = null;
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
                        Profiling.PhaseSuffix = "";
                        if (network is IAuxiliaryLossFreeLoadBalancedNetwork loadBalancedNetworkAfterForward)
                            loadBalancedNetworkAfterForward.UpdateExpertLoadBalance = false;
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
                        Tensor clippedRatio = clamp(ratio, 1f - settings.PpoClipLowEpsilon, 1f + settings.PpoClipHighEpsilon);
                        Tensor clipMask = ratio.lt(1f - settings.PpoClipLowEpsilon).logical_or(ratio.gt(1f + settings.PpoClipHighEpsilon));
                        clipCount = clipMask.to_type(ScalarType.Float32).sum();
                        totalRatioSampleCount += batchEnd - batchStart;
                        Tensor policyReward = min(ratio * advantages, clippedRatio * advantages).mean();
                        entropy = settings.UseSampledSoftmax
                            ? -logPiNew.mean()
                            : -(exp(logProbs) * logProbs).sum(dim: 1).mean();
                        Tensor policyLoss = -policyReward;
                        if (settings.EntropyCoefficient != 0f)
                            policyLoss -= settings.EntropyCoefficient * entropy;

                        Tensor valueTargets = batch.ValueTarget.to(values.device).reshape([-1]);
                        Tensor valueLoss = functional.mse_loss(values.reshape([-1]), valueTargets);
                        loss = policyLoss + settings.ValueLossCoefficient * valueLoss;

                        // Distil the secondary tower toward the policy tower over the sampled
                        // moves. The target is detached, so this trains only the small tower;
                        // its trunk input is detached inside the network for the same reason.
                        if (settings.DistillationCoefficient > 0f &&
                            network is NewPolicyNetwork dualNetwork &&
                            dualNetwork.HasSecondaryLeaf &&
                            settings.UseSampledSoftmax)
                        {
                            Tensor smallLogits = dualNetwork.GetSecondaryPolicyLogits(
                                batch.StateTensors, batch.MoveIndices);
                            Tensor smallLogProbs = functional.log_softmax(smallLogits, dim: 1);
                            Tensor largeLogProbs = functional.log_softmax(
                                sampledLogitsForDistillation, dim: 1).detach();
                            Tensor largeProbs = exp(largeLogProbs);
                            Tensor distillationLoss =
                                (largeProbs * (largeLogProbs - smallLogProbs)).sum(dim: 1).mean();
                            loss = loss + settings.DistillationCoefficient * distillationLoss;
                        }

                        // Each batch contributes its share, so the accumulated gradient is
                        // the mean over the group rather than the sum.
                        if (groupBatchCount > 1)
                            loss = loss / groupBatchCount;
                    }

                    float kldValue;
                    using (ProfileScope.New("PpoScalarReadbacks"))
                    {
                        kldValue = kld.item<float>();
                        clippedSampleCount += clipCount.item<float>();
                    }
                    kldSum += kldValue;
                    kldCount++;
                    groupMaxKld = Math.Max(groupMaxKld, kldValue);

                    using (ProfileScope.New("PpoBackward"))
                    {
                        loss.backward();
                    }
                    if (settings.KldEarlyStopThreshold > 0f && kldValue > settings.KldEarlyStopThreshold)
                    {
                        // Record the batch that tripped the threshold before bailing, so the
                        // value responsible for the stop is visible in the step data rather
                        // than being dropped along with the abandoned group.
                        gradNormTensors.Add(zeros(1, device: CPU).ToOuterScope());
                        entropyTensors.Add(entropy.ToOuterScope());
                        kldValues.Add(kldValue);

                        // Abandon the partly accumulated group; the next rollout zeroes it.
                        stoppedEarly = true;
                        continue;
                    }

                    // Clipping and the step belong to the whole group, so both wait until
                    // every batch in it has contributed its gradient.
                    if (!isLastBatchInGroup)
                        continue;

                    Tensor gradNormTensor;
                    if (settings.GradNormClip > 0f)
                    {
                        using (ProfileScope.New("GradNormAndClip"))
                        {
                            double gradNorm = utils.clip_grad_norm_(
                                networkModule.parameters(),
                                settings.GradNormClip);
                            gradNormTensor = tensor((float)gradNorm).ToOuterScope();
                        }
                    }
                    else
                    {
                        using (ProfileScope.New("GetGradNorm"))
                        {
                            gradNormTensor = GetGradNormTensor(networkModule).ToOuterScope();
                        }
                    }
                    gradNormTensors.Add(gradNormTensor);
                    entropyTensors.Add(entropy.ToOuterScope());
                    kldValues.Add(groupMaxKld);

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
        MoELoadBalanceSummary? loadBalance = network is IAuxiliaryLossFreeLoadBalancedNetwork trainingLoadBalancedNetwork
            ? SummarizeRoutingStats(trainingLoadBalancedNetwork.DrainRoutingStats())
            : null;
        return new()
        {
            Trajectories = trajectories,
            StateData = stateData,
            StepData = stepData,
            ClipRate = clipRate,
            StoppedEarly = stoppedEarly,
            AverageKld = averageKld,
            LoadBalance = loadBalance,
            LeafCoverage = LeafCoverageStats.Drain()
        };
    }

    static MoELoadBalanceSummary? SummarizeRoutingStats(List<MoERoutingStats> stats)
    {
        if (stats.Count == 0)
            return null;

        // Blocks may have different routed expert counts, so utilization is taken against
        // each block's own count, and the per-expert fractions are averaged only over the
        // blocks that share the first block's count. Fractions from blocks of a different
        // width are not comparable index-by-index and are left out.
        int expertCount = stats[0].ExpertTokenFractions.Length;
        float[] fractionSums = new float[expertCount];
        int fractionStatCount = 0;
        double lossSum = 0;
        double utilizationSum = 0;
        float minUtilization = 1f;
        for (int statIndex = 0; statIndex < stats.Count; ++statIndex)
        {
            MoERoutingStats stat = stats[statIndex];
            lossSum += stat.LoadBalancingLoss;
            float utilization = stat.ActiveExpertCount / (float)stat.ExpertTokenFractions.Length;
            utilizationSum += utilization;
            minUtilization = Math.Min(minUtilization, utilization);

            if (stat.ExpertTokenFractions.Length != expertCount)
                continue;
            fractionStatCount++;
            for (int expertIndex = 0; expertIndex < expertCount; ++expertIndex)
                fractionSums[expertIndex] += stat.ExpertTokenFractions[expertIndex];
        }

        float[] meanFractions = new float[expertCount];
        for (int expertIndex = 0; expertIndex < expertCount; ++expertIndex)
            meanFractions[expertIndex] = fractionSums[expertIndex] / fractionStatCount;

        return new(
            AverageLoadBalancingLoss: (float)(lossSum / stats.Count),
            MinExpertUtilization: minUtilization,
            MeanExpertUtilization: (float)(utilizationSum / stats.Count),
            ExpertTokenFractions: meanFractions);
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
    /// <summary>
    /// Number of batches whose gradients are accumulated before an optimizer step, so an
    /// effective step covers BatchSize * this many samples. One steps every batch.
    /// </summary>
    public int GradientAccumulationSteps = 1;
    /// <summary>
    /// Weight on the KL that distils the secondary move tower toward the policy tower.
    /// Zero disables it, leaving the secondary tower untrained.
    /// </summary>
    public float DistillationCoefficient = 0f;
    public float LearningRate = 1e-5f;
    public float AdamBeta1 = 0.9f;
    public float AdamBeta2 = 0.97f;
    public float WeightDecay = 0.01f;
    public float PpoClipLowEpsilon = 0.2f;
    public float PpoClipHighEpsilon = 0.2f;
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
    float AverageKld,
    MoELoadBalanceSummary? LoadBalance = null,
    LeafCoverageSummary? LeafCoverage = null);

public readonly record struct MoELoadBalanceSummary(
    float AverageLoadBalancingLoss,
    float MinExpertUtilization,
    float MeanExpertUtilization,
    float[] ExpertTokenFractions);

public readonly record struct PpoStateData(
    int GameInRolloutIndex,
    int MoveIndex,
    float Advantage,
    float ChosenMoveProb);

public readonly record struct PpoStepData(
    float GradNorm,
    float Kld,
    float Entropy);
