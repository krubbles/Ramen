namespace BalatroAI;

using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Training
{

    public static void TrainEvaluationModel(AIModels models, int epochs, int batchSize, bool validate = false)
    {
        Console.WriteLine($"Training evaluation model for {epochs} epochs, batch size {batchSize}");

        EvaluationTrainingSample stacked;
        lock (TrainingData.EvaluationTrainingData)
        {
            stacked = EvaluationTrainingSample.Stack(TrainingData.EvaluationTrainingData, false);
        }

        var optimizer = optim.AdamW(models.Evaluation.parameters(), lr: TrainingConfig.LearningRate, weight_decay: 0.01f);
        var lossFunc = GaussianNLLLoss();

        int samples = (int)stacked.Target.size(dim: 0);
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
                    GameStateTensors inputs = stacked.GameStateTensors.GetBatch(i, end);
                    Tensor targets = stacked.Target[i..end];

                    Tensor predictions = models.Evaluation.forward(inputs);
                    Tensor predictedMeans = predictions[TensorIndex.Colon, 0];
                    Tensor predictedDeviations = predictions[TensorIndex.Colon, 1];
                    var loss = lossFunc.forward(predictedMeans, targets, predictedDeviations);
                    valLossAvg += loss.item<float>();
                    valBatchCount++;
                }
            }

            valLossAvg /= Math.Max(1, valBatchCount);
            float trainLossAvg = 0f;
            int trainBatchCount = 0;
            for (int i = trainCount; i < samples; i += batchSize)
            {
                int end = Math.Min(i + batchSize, samples);
                GameStateTensors inputs = stacked.GameStateTensors.GetBatch(i, end);
                Tensor targets = stacked.Target[i..end];

                optimizer.zero_grad();

                Tensor predictions = models.Evaluation.forward(inputs);
                Tensor predictedMeans = predictions[TensorIndex.Colon, 0];
                Tensor predictedDeviations = predictions[TensorIndex.Colon, 1];
                
                var loss = lossFunc.forward(predictedMeans, targets, predictedDeviations);
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

    public static void TrainPolicyModel(AIModels models, int epochs, int batchSize, bool validate = false)
    {
        Console.WriteLine($"Training policy model for {epochs} epochs, batch size {batchSize}");

        PolicyTrainingSample stackedSamples;
        lock (TrainingData.PolicyTrainingData)
        {
            stackedSamples = PolicyTrainingSample.Stack(TrainingData.PolicyTrainingData, false);
        }

        var optimizer = optim.Adam(models.Policy.parameters(), lr: TrainingConfig.LearningRate);
        var lossFunc = CrossEntropyLoss();


        float lossAvg = 0;
        int batchCount = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {

            int fullSampleCount = (int)stackedSamples.Output.size(dim: 0);
            int valCount = validate ? Math.Max(1, fullSampleCount / 10) : 0;
            int trainSampleCount = Math.Max(0, fullSampleCount - valCount);


            // validation
            float valLossAvg = 0f;
            int valBatchCount = 0;
            using (no_grad())
            {
                for (int i = trainSampleCount; i < fullSampleCount; i += batchSize)
                {
                    var batchFullHands = stackedSamples.GameStateTensors.Hand[i..Math.Min(i + batchSize, fullSampleCount)];
                    var batchOtherInputs = stackedSamples.GameStateTensors.OtherState[i..Math.Min(i + batchSize, fullSampleCount)];
                    var batchInUseMasks = stackedSamples.InUseMask[i..Math.Min(i + batchSize, fullSampleCount)];
                    var batchOutputs = stackedSamples.Output[i..Math.Min(i + batchSize, fullSampleCount)];

                    var predictions = models.GetCardUseRewards(batchFullHands, batchOtherInputs, batchInUseMasks);
                    var loss = lossFunc.forward(predictions, batchOutputs);
                    valLossAvg += loss.item<float>();
                    valBatchCount++;
                }
            }

            valLossAvg /= Math.Max(1, valBatchCount);


            lossAvg = 0;
            batchCount = 0;
            for (int i = 0; i < trainSampleCount; i += batchSize)
            {
                var batchFullHands = stackedSamples.GameStateTensors.Hand[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOtherInputs = stackedSamples.GameStateTensors.OtherState[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchInUseMasks = stackedSamples.InUseMask[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOutputs = stackedSamples.Output[i..Math.Min(i + batchSize, trainSampleCount)];
                optimizer.zero_grad();

                var predictions = models.GetCardUseRewards(batchFullHands, batchOtherInputs, batchInUseMasks);
                var loss = lossFunc.forward(predictions, batchOutputs);
                loss.backward();
                optimizer.step();
                lossAvg += loss.item<float>();
                batchCount++;

            }

            lossAvg /= Math.Max(1, batchCount);
            Console.WriteLine($"Epoch {epoch} | Train Loss = {lossAvg} | Val Loss = {valLossAvg}");

            if (true)
            {
                int averageRewardSampleCount = 100;
                Console.WriteLine($"Reward {Testing.GetAverageReward(models, averageRewardSampleCount)} over {averageRewardSampleCount} samples");
            }
        }
        { // final reward
            int averageRewardSampleCount = 1000;
            Console.WriteLine($"Reward {Testing.GetAverageReward(models, averageRewardSampleCount)} over {averageRewardSampleCount} samples");
        }
    }
}

