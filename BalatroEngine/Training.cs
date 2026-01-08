#define STACKLESS
namespace Ramen.AI;

using System.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Training
{
    public const float epsilonLow = 0.2f, epsilonHigh = 0.2f;
    public static float entropyCoeff = 0;
    public static float kldCoeff = 0.005f;
    public static float lr = 2e-4f;

    public static void TrainEvaluationModel(GameEvalModel model, int epochs, int batchSize, bool validate = false)
    {

        Console.WriteLine($"Training evaluation model for {epochs} epochs, batch size {batchSize}");

        EvaluationTrainingSample stacked;
        lock (TrainingData.EvaluationTrainingData)
        {
            stacked = TensorGroupExtentions.Stack(TrainingData.EvaluationTrainingData, false, false);
        }

        stacked = stacked.IndexSelect(0, randperm(stacked.Advantage.size(0)));

        var optimizer = optim.AdamW(model.parameters(),
            lr: lr,
            weight_decay: 0.001f,
            beta1: 0.9f,
            beta2: 0.994f
            );

        var lossFunc = MSELoss();

        int samples = TrainingData.EvaluationTrainingData.Count;

        int valCount = validate ? Math.Max(1, samples / 10) : 0; 
        int trainCount = Math.Max(0, samples - valCount);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {

            float valLossAvg = 0f;
            int valBatchCount = 0;
            using (no_grad())
            {
                for (int i = trainCount; i < samples; i += batchSize)
                {
                    int end = Math.Min(i + batchSize, samples);
                    EvaluationTrainingSample inputs = stacked.GetBatch(i, end) as EvaluationTrainingSample;

                    Tensor logits = model.forward(inputs.GameStateTensors).squeeze(2); 

                    var logQ = log(inputs.ProbDist + 1e-9);
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


            for (int i = 0; i < trainCount; i += batchSize)
            {
                optimizer.zero_grad();

                int end = Math.Min(i + batchSize, samples);
                EvaluationTrainingSample inputs = stacked.GetBatch(i, end);

                var probDist = inputs.ProbDist / inputs.ProbDist.sum(dim: 1, true);
                var logProbsOld = log(probDist.clamp_min(0f) + 1e-9);

                Tensor logits = model.forward(inputs.GameStateTensors).squeeze(2);
                var logProbs = functional.log_softmax(logits, dim: 1);
                var logPiNew = logProbs.select(1, 0);

                var probs = exp(logProbs);
                var entropy = -(probs * logProbs).sum(1).mean();

                var logPiOld = logProbsOld.select(1, 0);

                var ratio = exp(logPiNew - logPiOld) - 1;

                var kld = (probDist * (logProbsOld - logProbs)).sum(dim: 1);

                var advantage = inputs.Advantage;
                var surr1 = ratio * advantage;
                var surr2 = clamp(ratio, -epsilonLow, epsilonHigh) * advantage;
                var surrMin = min(surr1, surr2);

                var policyReward = surrMin.mean();/// (surrMin.norm(p: 2) / surrMin.size(0)).pow(0.25f);
                var entropyReward = entropyCoeff * entropy;
                var kldLoss = kld.mean() * kldCoeff;
                var loss = (kldLoss - policyReward - entropyReward);

                loss.backward();
                optimizer.step();

                trainLossAvg += loss.item<float>();
                trainBatchCount++;
            }


            trainLossAvg /= Math.Max(1, trainBatchCount);



            Console.WriteLine($"Eval Epoch {epoch} | Train Loss = {trainLossAvg} | Val Loss = {valLossAvg}");
        }

        stacked.Dispose();
    }
}

