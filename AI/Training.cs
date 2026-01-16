#define STACKLESS
namespace Ramen.AI;

using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public record struct TrainingParams(
    int epochs = 5,
    int batchSize = 128,
    float learningRate = 1e-5f,
    float entropyCoeff = 0.01f,
    float kldCoeff = 0.01f);

public static class Training
{
    public const float epsilonLow = 1, epsilonHigh = 1000000;

    public static void TrainEvaluationModel(PolicyModel model, TrainingParams tp, CancellationToken cancel)
    {

        Console.WriteLine($"Training evaluation model for {tp.epochs} epochs, batch size {tp.batchSize}");

        EvaluationTrainingSample stacked;
        lock (TrainingData.EvaluationTrainingData)
        {
            stacked = TensorGroupExtentions.Stack(TrainingData.EvaluationTrainingData, false, true);
        }

        stacked = stacked.IndexSelect(0, randperm(stacked.Advantage.size(0)));

        var optimizer = optim.AdamW(model.parameters(),
            lr: tp.learningRate,
            weight_decay: 0.01f,
            beta1: 0.9f,
            beta2: 0.998f
            );

        var lossFunc = MSELoss();

        int samples = TrainingData.EvaluationTrainingData.Count;

        bool validate = false;
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
                    EvaluationTrainingSample inputs = stacked.GetBatch(i, end);
                    Tensor processedState = model.ProcessState(inputs.State);
                    Tensor logits = model.GetPolicyLogits(inputs.Moves, processedState).squeeze_(2);

                    var logQ = log(inputs.MoveProbDist + 1e-9);
                    var logitsAdjusted = logits - logQ;

                    var currentBatchSize = logits.shape[0];
                    var targets = zeros([currentBatchSize], ScalarType.Int64, device: logits.device);
                    var ceLoss = functional.cross_entropy(logitsAdjusted, targets, reduction: Reduction.None);
                    var loss = (ceLoss * inputs.Advantage).mean();

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
                optimizer.zero_grad();

                int end = Math.Min(i + tp.batchSize, samples);
                EvaluationTrainingSample inputs = stacked.GetBatch(i, end);

                var probDist = inputs.MoveProbDist / inputs.MoveProbDist.sum(dim: 1, true);
                Tensor processedState = model.ProcessState(inputs.State);
                int moveCount = (int)inputs.Moves.HandsAndDiscards.size(1);

                Tensor moveLogits = model.GetPolicyLogits(inputs.Moves, processedState).squeeze(2);
                Tensor moveLoss = CalculatePPOLoss(moveLogits, probDist, inputs.Advantage, tp.entropyCoeff, tp.kldCoeff, true, ref kldTotal);

                Tensor loss = moveLoss;
                
                loss.backward();
                optimizer.step();

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

    static Tensor CalculatePPOLoss(Tensor logits, Tensor oldProbs, Tensor advantage, float ec, float kc, bool useIndex0, ref float kldAccumulate, Tensor moveIndex = null)
    {
        var logProbsOld = log(oldProbs.clamp_min(0f) + 1e-9);
        var logProbs = functional.log_softmax(logits, dim: 1);

        var logPiNew = useIndex0 ?
            logProbs.select(1, 0) :
            logProbs.gather(1, moveIndex);

        var probs = exp(logProbs);
        var entropy = -(probs * logProbs).sum(1).mean();

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

}

