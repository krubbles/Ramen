#define STACKLESS
namespace Ramen.Training;

using System.Linq;
using Ramen.AI;
using Ramen.AgentTools;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TensorGroups = Ramen.AgentTools.TensorGroupExtentions;

public record struct TrainingParams
(
    int epochs = 5,
    int batchSize = 128,
    float learningRate = 1e-5f,
    float entropyCoeff = 0.00f, // only used in GRPO
    float kldCoeff = 0.0f // only used in GRPO
);

public static class Training
{
    public const float epsilonLow = 0.2f, epsilonHigh = 0.2f;
    public static AdamW Optimizer;
    static IPolicyModel _model;

    static void SetModelAndOptimizer(IPolicyModel model, TrainingParams tp)
    {
        Module modelModule = model as Module;
        if (_model == model)
            return;
        _model = model;
        Optimizer = optim.AdamW(modelModule.parameters(),
            lr: tp.learningRate,
            weight_decay: 0.00f,
            beta1: 0.9f,
            beta2: 0.99f
        );
        Optimizer.zero_grad();
    }

    public static void TrainPolicyModelGRPO(IPolicyModel model, IReadOnlyList<PolicyTrainingSample> trainingData, TrainingParams tp, CancellationToken cancel, bool validate = false)
    {
        using var dscope = NewDisposeScope();

        PolicyTrainingSample stacked = TensorGroups.Stack(trainingData, false, true);
        using (var nograd = no_grad())
        {
            _ = validate; // GRPO uses a surrogate objective, so validation loss is not meaningful.

            lock (trainingData)
            {
                stacked = TensorGroups.Stack(trainingData, false, true);
            }

            Tensor advantageGPU = stacked.Advantage.to(MPS);
            Tensor entScalarGPU = stacked.EntropyScalar.to(MPS);
            stacked.Advantage.Dispose();
            stacked.Advantage = advantageGPU;
            stacked.EntropyScalar.Dispose();
            stacked.EntropyScalar = entScalarGPU;

            stacked = stacked.IndexSelect(0, randperm(stacked.SamplingProb.size(0)));
        }

        SetModelAndOptimizer(model, tp);

        int samples = trainingData.Count;

        int trainCount = samples;

        for (int epoch = 0; epoch < tp.epochs; ++epoch)
        {
            float trainLossAvg = 0f;
            int trainBatchCount = 0;
            float kldTotal = 0f;

            for (int i = 0; i < trainCount; i += tp.batchSize)
            {
                using var dscopeInner = NewDisposeScope();
                
                Optimizer.zero_grad();

                int end = Math.Min(i + tp.batchSize, samples);
                PolicyTrainingSample inputs = stacked.GetBatch(i, end);
                Tensor logits = model.GetPolicyLogits(inputs.StateTensors, inputs.UseHandTensors, inputs.MoveIndices);
                Tensor loss = CalculatePPOLoss(logits, inputs.SamplingProb, inputs.Advantage, inputs.EntropyScalar, tp.entropyCoeff, tp.kldCoeff, useIndex0: true, ref kldTotal);
                loss.backward();
                Optimizer.step();

                trainLossAvg += loss.item<float>();
                trainBatchCount++;

                if (cancel.IsCancellationRequested)
                {
                    stacked.Dispose();
                    return;
                }
            }

            trainLossAvg /= Math.Max(1, trainBatchCount);
            if (epoch == tp.epochs - 1)
                Console.WriteLine($"KLD: {kldTotal / Math.Max(1, trainBatchCount)}");

            GC.Collect();
        }

        stacked.Dispose();
    }

    static Tensor CalculatePPOLoss(Tensor logits, Tensor oldProbs, Tensor advantage, Tensor entScalar, float ec, float kc, bool useIndex0, ref float kldAccumulate, Tensor moveIndex = null)
    {
        oldProbs /= oldProbs.sum(dim: 1, keepdim: true).max(1e-9f);
        var logProbsOld = log(oldProbs.clamp_min(0f) + 1e-9f);
        var logProbs = functional.log_softmax(logits, dim: 1);

        var logPiNew = useIndex0 ?
            logProbs.select(1, 0) :
            logProbs.gather(1, moveIndex);

        var probs = exp(logProbs);
        var entropy = (-(probs * logProbs).sum(1) * entScalar).mean();

        var logPiOld = useIndex0 ?
            logProbsOld.select(1, 0) :
            logProbsOld.gather(1, moveIndex);

        var ratio = exp(logPiNew - logPiOld) - 1;

        var kld = (oldProbs * (logProbsOld - logProbs)).sum(dim: 1).mean();
        kldAccumulate += kld.item<float>();

        var surr1 = ratio * advantage;
        var surr2 = clamp(ratio, -epsilonLow, epsilonHigh) * advantage;
        var surrMin = min(surr1, surr2);

        var policyReward = surrMin.mean();/// (surrMin.norm(p: 2) / surrMin.size(0)).pow(0.25f);
        var entropyReward = ec * entropy;
        var kldLoss = kld * kc;
        var loss = (kldLoss - policyReward - entropyReward);
        return loss;
    }

    static Tensor CalculateCISPOLoss(Tensor logits, Tensor oldProbs, Tensor advantage, float ec, bool useIndex0, ref float kldAccumulate, Tensor moveIndex = null)
    {
        Tensor logProbsOld = log(oldProbs.clamp_min(0f) + 1e-9);
        Tensor logProbs = functional.log_softmax(logits, dim: 1);

        Tensor logPiNew = useIndex0 ?
            logProbs.select(1, 0) :
            logProbs.gather(1, moveIndex);

        Tensor logPiOld = useIndex0 ?
            logProbsOld.select(1, 0) :
            logProbsOld.gather(1, moveIndex);

        Tensor ratio = exp(logPiNew - logPiOld);
        Tensor clippedRatio = clamp(ratio, 1f - epsilonLow, 1f + epsilonHigh);
        Tensor weight = clippedRatio.detach();

        Tensor probs = exp(logProbs);
        Tensor entropy = -(probs * logProbs).sum(1).mean();

        Tensor kld = (oldProbs * (logProbsOld - logProbs)).sum(dim: 1).mean();
        kldAccumulate += kld.item<float>();

        Tensor policyLoss = -(weight * advantage * logPiNew).mean();
        Tensor entropyReward = ec * entropy;
        Tensor loss = policyLoss - entropyReward;
        return loss;
    }
}
