namespace Ramen.Training;

using System;
using Ramen.AgentTools;
using Ramen.Game;
using Ramen.AI;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TensorGroups = Ramen.AgentTools.TensorGroupExtentions;

public static class PolicyValueNetworkTraining
{
    public const int RolloutBatchSize = 64;

    public static List<PolicyTrainingSample> GenerateRollout(IPolicyNetwork network, PpoTrainingSettings settings)
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
                gameStates[slot].AdvanceToNextPlayerChoice();

                if (!agent.IsGameDone(gameStates[slot]))
                    continue;

                SetTargetsAndAdvantages(activeSamples[slot], GetReward(gameStates[slot]));
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

    static void SetTargetsAndAdvantages(List<PolicyTrainingSample> samples, float finalReward)
    {
        float subsequentValueSum = finalReward;

        for (int index = samples.Count - 1; index >= 0; --index)
        {
            PolicyTrainingSample sample = samples[index];
            float predictedValue = sample.Value.item<float>();
            int subsequentStateCount = samples.Count - index;
            float valueTarget = subsequentValueSum / subsequentStateCount;
            float policyAdvantage = valueTarget - predictedValue;

            sample.ValueTarget?.Dispose();
            sample.PolicyAdvantage?.Dispose();
            sample.ValueTarget = tensor([valueTarget], device: CPU).DetachFromScope();
            sample.PolicyAdvantage = tensor([policyAdvantage], device: CPU).DetachFromScope();

            subsequentValueSum += predictedValue;
        }
    }

    public static float GetReward(GameState gameState)
    {
        float roundsSurvived = gameState.Round / 3f;
        return roundsSurvived * roundsSurvived;
    }

    public static void DoPPORollout(IPolicyNetwork network, PpoTrainingSettings settings)
    {
        using var scope = NewDisposeScope();

        List<PolicyTrainingSample> rollout = GenerateRollout(network, settings);

        if (network is not Module networkModule)
            throw new InvalidOperationException($"{nameof(network)} must be a TorchSharp module to train.");

        using AdamW optimizer = optim.AdamW(
            networkModule.parameters(),
            lr: settings.LearningRate,
            weight_decay: settings.WeightDecay,
            beta1: settings.AdamBeta1,
            beta2: settings.AdamBeta2);

        PolicyTrainingSample stacked = TensorGroups.Stack(rollout, disposeInputs: false, concat: true);
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

                (Tensor sampledLogits, Tensor values) = network.GetPolicyLogits(batch.StateTensors, batch.MoveIndices);
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

    public GameData GameData = GameData.Default;

    public PpoTrainingSettings() { }
}
