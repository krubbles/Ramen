namespace BalatroAI;

using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Training
{

    public static void TrainEvaluationModel(AIModels models, int epochs, int batchSize)
    {
        Console.WriteLine($"Training evaluation model for {epochs} epochs, batch size {batchSize}");

        EvaluationTrainingSample stacked;
        lock (TrainingData.EvaluationTrainingData)
        {
            stacked = EvaluationTrainingSample.Stack(TrainingData.EvaluationTrainingData, false);
        }

        var optimizer = optim.Adam(models.Evaluation.parameters(), lr: TrainingConfig.LearningRate);
        var lossFunc = MSELoss();

        int samples = (int)stacked.Target.size(dim: 0);
        int valCount = Math.Max(1, samples / 10); // last 10% for validation (at least 1)
        int trainCount = Math.Max(0, samples - valCount);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float lossAvg = 0f;
            int batchCount = 0;
            for (int i = 0; i < trainCount; i += batchSize)
            {
                int end = Math.Min(i + batchSize, trainCount);
                var batchFullHands = stacked.GameStateTensors.FullHand[i..end];
                var batchOther = stacked.GameStateTensors.OtherState[i..end];
                var batchInUseMasks = stacked.InUseMask[i..end];
                var batchTargets = stacked.Target[i..end];
                optimizer.zero_grad();

                Tensor fullHandEmbedded = models.EmbedCards(batchFullHands);
                Tensor predictedReward = models.Evaluation.forward(fullHandEmbedded, batchOther, batchInUseMasks);
                var loss = lossFunc.forward(predictedReward, batchTargets);
                loss.backward();
                optimizer.step();

                lossAvg += loss.item<float>();
                batchCount++;
            }

            lossAvg /= Math.Max(1, batchCount);

            // validation
            float valLossAvg = 0f;
            int valBatchCount = 0;
            using (no_grad())
            {
                for (int i = trainCount; i < samples; i += batchSize)
                {
                    int end = Math.Min(i + batchSize, samples);
                    var batchFullHands = stacked.GameStateTensors.FullHand[i..end];
                    var batchOther = stacked.GameStateTensors.OtherState[i..end];
                    var batchInUseMasks = stacked.InUseMask[i..end];
                    var batchTargets = stacked.Target[i..end];

                    Tensor fullHandEmbedded = models.EmbedCards(batchFullHands);
                    Tensor predictedReward = models.Evaluation.forward(fullHandEmbedded, batchOther, batchInUseMasks);
                    var loss = lossFunc.forward(predictedReward, batchTargets);
                    valLossAvg += loss.item<float>();
                    valBatchCount++;
                }
            }

            valLossAvg /= Math.Max(1, valBatchCount);

            Console.WriteLine($"Eval Epoch {epoch} | Train Loss = {lossAvg} | Val Loss = {valLossAvg}");

            if (epoch % 5 == 4)
            {
                Testing.ShowExpectedReward(models, 1);
            }
        }

        stacked.Dispose();
    }

    public static void TrainPolicyModel(AIModels models, int epochs, int batchSize)
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
            int valCount = Math.Max(1, fullSampleCount / 10);
            int trainSampleCount = Math.Max(0, fullSampleCount - valCount);

            lossAvg = 0;
            batchCount = 0;
            for (int i = 0; i < trainSampleCount; i += batchSize)
            {
                var batchFullHands = stackedSamples.GameStateTensors.FullHand[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOtherInputs = stackedSamples.GameStateTensors.OtherState[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchInUseMasks = stackedSamples.InUseMask[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOutputs = stackedSamples.Output[i..Math.Min(i + batchSize, trainSampleCount)];
                optimizer.zero_grad();
                Tensor fullHandEmbedded = models.EmbedCards(batchFullHands);
                var predictions = models.GetCardUseRewards(fullHandEmbedded, batchOtherInputs, batchInUseMasks);
                var loss = lossFunc.forward(predictions, batchOutputs);
                loss.backward();
                optimizer.step();
                lossAvg += loss.item<float>();
                batchCount++;

            }

            lossAvg /= Math.Max(1, batchCount);

            // validation
            float valLossAvg = 0f;
            int valBatchCount = 0;
            using (no_grad())
            {
                for (int i = trainSampleCount; i < fullSampleCount; i += batchSize)
                {
                    var batchFullHands = stackedSamples.GameStateTensors.FullHand[i..Math.Min(i + batchSize, fullSampleCount)];
                    var batchOtherInputs = stackedSamples.GameStateTensors.OtherState[i..Math.Min(i + batchSize, fullSampleCount)];
                    var batchInUseMasks = stackedSamples.InUseMask[i..Math.Min(i + batchSize, fullSampleCount)];
                    var batchOutputs = stackedSamples.Output[i..Math.Min(i + batchSize, fullSampleCount)];

                    Tensor fullHandEmbedded = models.EmbedCards(batchFullHands);
                    var predictions = models.GetCardUseRewards(fullHandEmbedded, batchOtherInputs, batchInUseMasks);
                    var loss = lossFunc.forward(predictions, batchOutputs);
                    valLossAvg += loss.item<float>();
                    valBatchCount++;
                }
            }

            valLossAvg /= Math.Max(1, valBatchCount);

            Console.WriteLine($"Epoch {epoch} | Train Loss = {lossAvg} | Val Loss = {valLossAvg}");

            if (epoch % 5 == 4)
            {
                Console.WriteLine("Average Reward: " + Testing.GetAverageReward(models, 1000));
            }
        }
    }
}

