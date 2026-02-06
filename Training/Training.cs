#define STACKLESS
namespace Ramen.Training;

using System.Linq;
using Ramen.AI;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

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
            weight_decay: 0.01f,
            beta1: 0.9f,
            beta2: 0.998f
        );
        Optimizer.zero_grad();
    }

    public static void TrainPolicyModelSupervised(IPolicyModel model, TrainingParams tp, CancellationToken cancel, bool validate = false)
    {
        Console.WriteLine($"Training evaluation model for {tp.epochs} epochs, batch size {tp.batchSize}");

        PolicyTrainingSample stacked;
        lock (TrainingData.PolicyData)
        {
            stacked = TensorGroupExtentions.Stack(TrainingData.PolicyData, false, true);
        }

        stacked = stacked.IndexSelect(0, randperm(stacked.SamplingProb.size(0)));

        SetModelAndOptimizer(model, tp);

        int samples = TrainingData.PolicyData.Count;

        int valCount = validate ? Math.Max(1, samples / 10) : 0;
        int trainCount = Math.Max(0, samples - valCount);

        for (int epoch = 0; epoch < tp.epochs; ++epoch)
        {

            float valLossAvg = 0f;
            int valBatchCount = 0;
            using (no_grad())
            {
                for (int i = trainCount; i < samples; i += tp.batchSize)
                {
                    int end = Math.Min(i + tp.batchSize, samples);
                    PolicyTrainingSample inputs = stacked.GetBatch(i, end);
                    Tensor loss = CalculateSupervisedLoss(model, inputs);
                    loss.backward();

                    valLossAvg += loss.item<float>();
                    valBatchCount++;
                }
            }

            valLossAvg /= Math.Max(1, valBatchCount);
            float trainLossAvg = 0f;
            int trainBatchCount = 0;

            float kldTotal = 0;
            for (int i = 0; i < trainCount; i += tp.batchSize)
            {
                Optimizer.zero_grad();

                int end = Math.Min(i + tp.batchSize, samples);
                PolicyTrainingSample inputs = stacked.GetBatch(i, end);
                Tensor loss = CalculateSupervisedLoss(model, inputs);
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

            Console.WriteLine($"Eval Epoch {epoch} | Train Loss = {trainLossAvg} | KLD = {kldTotal / Math.Max(1, trainBatchCount)}");
        }

        stacked.Dispose();
    }

    public static void TrainPolicyModelGRPO(IPolicyModel model, TrainingParams tp, CancellationToken cancel, bool validate = false)
    {
        Console.WriteLine($"Training GRPO model for {tp.epochs} epochs, batch size {tp.batchSize}");
        _ = validate; // GRPO uses a surrogate objective, so validation loss is not meaningful.

        PolicyTrainingSample stacked;
        lock (TrainingData.PolicyData)
        {
            stacked = TensorGroupExtentions.Stack(TrainingData.PolicyData, false, true);
        }

        stacked = stacked.IndexSelect(0, randperm(stacked.SamplingProb.size(0)));

        SetModelAndOptimizer(model, tp);

        int samples = TrainingData.PolicyData.Count;

        int trainCount = samples;

        for (int epoch = 0; epoch < tp.epochs; ++epoch)
        {
            float trainLossAvg = 0f;
            int trainBatchCount = 0;
            float kldTotal = 0f;

            for (int i = 0; i < trainCount; i += tp.batchSize)
            {
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
            Console.WriteLine($"GRPO Epoch {epoch} | Train Loss = {trainLossAvg} | KLD = {kldTotal / Math.Max(1, trainBatchCount)}");
        }

        stacked.Dispose();
    }

    static Tensor CalculateSupervisedLoss(IPolicyModel model, PolicyTrainingSample sample)
    {
        Tensor logits = model.GetPolicyLogits(sample.StateTensors, sample.UseHandTensors, sample.MoveIndices);
        Tensor logQ = log(sample.SamplingProb + 1e-9);
        Tensor logitsAdjusted = logits - logQ;

        Tensor logProbs = functional.log_softmax(logitsAdjusted, dim: 1);
        Tensor ceLoss = -(sample.Target * logProbs);
        Tensor loss = ceLoss.mean();
        return loss;
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
