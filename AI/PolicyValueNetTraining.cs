namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public static class PolicyValueNetworkTraining
{
    public const int RolloutBatchSize = 64;

    public static List<PolicyTrainingSample> GenerateRollout(
        IPolicyNetwork network,
        PpoTrainingSettings settings,
        IReadOnlyList<IRolloutAnalyzer> analyzers = null)
    {
        using PolicyNetworkAgent agent = new(network, ownsNetwork: false);
        List<PolicyTrainingSample> completedSamples = [];
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
                activeSamples[slot].Add(stepSamples[slot]);

                if (!IsTrajectoryDone(gameStates[slot]))
                    gameStates[slot].AdvanceToNextPlayerChoice();

                if (!IsTrajectoryDone(gameStates[slot]))
                    continue;

                float reward = GetReward(gameStates[slot]);
                SetTargetsAndAdvantages(activeSamples[slot], reward, settings.AdvantageFalloff);
                if (analyzers is not null)
                {
                    for (int analyzerIndex = 0; analyzerIndex < analyzers.Count; ++analyzerIndex)
                        analyzers[analyzerIndex].ObserveCompletedTrajectory(activeSamples[slot], reward);
                }
                completedSamples.AddRange(activeSamples[slot]);

                gameStates[slot] = new(settings.GameData);
                activeSamples[slot] = [];

                if (completedSamples.Count >= settings.RolloutStateCount)
                    break;
            }
        }

        if (completedSamples.Count > settings.RolloutStateCount)
            completedSamples.RemoveRange(settings.RolloutStateCount, completedSamples.Count - settings.RolloutStateCount);

        return completedSamples;
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
        return gameState.Round == 1 && gameState.Stage == StageOfGame.EndRound;
    }

    public static RolloutAnalysis DoPPORollout(IPolicyNetwork network, PpoTrainingSettings settings)
    {
        using var scope = NewDisposeScope();

        AverageRewardRolloutAnalyzer averageRewardAnalyzer = new();
        AverageEntropyRolloutAnalyzer averageEntropyAnalyzer = new();
        IRolloutAnalyzer[] analyzers = [averageRewardAnalyzer, averageEntropyAnalyzer];
        List<PolicyTrainingSample> rollout = GenerateRollout(network, settings, analyzers);

        if (network is not Module networkModule)
            throw new InvalidOperationException($"{nameof(network)} must be a TorchSharp module to train.");

        using AdamW optimizer = BuildAdamWOptimizer(networkModule, settings);

        PolicyTrainingSample stacked = TensorGroupExtentions.Stack(rollout, disposeInputs: false, concat: true);
        int sampleCount = rollout.Count;
        for (int epoch = 0; epoch < settings.EpochCount; ++epoch)
        {
            Tensor shuffledIndices = randperm(sampleCount, dtype: ScalarType.Int64, device: CPU);
            PolicyTrainingSample shuffled = stacked.IndexSelect(dim: 0, indices: shuffledIndices);

            for (int batchStart = 0; batchStart < sampleCount; batchStart += settings.BatchSize)
            {
                using var batchScope = NewDisposeScope();

                int batchEnd = Math.Min(batchStart + settings.BatchSize, sampleCount);
                PolicyTrainingSample batch = shuffled.GetBatch(batchStart, batchEnd);
                optimizer.zero_grad();

                (Tensor sampledLogits, Tensor values) = network.GetPolicyValue(batch.StateTensors, batch.MoveIndices);
                Tensor oldProbs = batch.SamplingProb.to(sampledLogits.device);
                oldProbs /= oldProbs.sum(dim: 1, keepdim: true).clamp_min(1e-9f);

                Tensor logProbsOld = log(oldProbs.clamp_min(1e-9f));
                Tensor logProbs = functional.log_softmax(sampledLogits, dim: 1);
                Tensor logPiNew = logProbs.select(dim: 1, index: 0);
                Tensor logPiOld = logProbsOld.select(dim: 1, index: 0);
                Tensor ratio = exp(logPiNew - logPiOld);

                Tensor advantages = batch.PolicyAdvantage.to(sampledLogits.device).reshape([-1]);
                Tensor clippedRatio = clamp(ratio, 1f - settings.PpoEpsilon, 1f + settings.PpoEpsilon);
                Tensor policyReward = min(ratio * advantages, clippedRatio * advantages).mean();
                Tensor entropy = -(exp(logProbs) * logProbs).sum(dim: 1).mean();
                Tensor policyLoss = -policyReward - settings.EntropyCoefficient * entropy;

                Tensor valueTargets = batch.ValueTarget.to(values.device).reshape([-1]);
                Tensor valueLoss = functional.mse_loss(values.reshape([-1]), valueTargets);
                Tensor loss = policyLoss + settings.ValueLossCoefficient * valueLoss;

                loss.backward();
                optimizer.step();
            }

            shuffled.Dispose();
            shuffledIndices.Dispose();
        }

        stacked.Dispose();
        for (int sampleIndex = 0; sampleIndex < rollout.Count; ++sampleIndex)
            rollout[sampleIndex].Dispose();

        return new(
            AverageReward: averageRewardAnalyzer.Value,
            AverageEntropy: averageEntropyAnalyzer.Value);
    }

    static AdamW BuildAdamWOptimizer(Module networkModule, PpoTrainingSettings settings)
    {
        List<Parameter> weightDecayParameters = [];
        List<Parameter> noWeightDecayParameters = [];
        foreach (Parameter parameter in networkModule.parameters())
        {
            if (parameter.dim() <= 1)
                noWeightDecayParameters.Add(parameter);
            else
                weightDecayParameters.Add(parameter);
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
    public float EntropyCoefficient = 0f;
    public float ValueLossCoefficient = 1f;
    public float AdvantageFalloff = 1f;

    public GameData GameData = GameData.Default;

    public PpoTrainingSettings() { }
}
