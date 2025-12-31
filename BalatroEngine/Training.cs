namespace BalatroAI;

using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Training
{

    public static void TrainEvaluationModel(GameEvalModel model, int epochs, int batchSize, bool validate = false)
    {
        Console.WriteLine($"Training evaluation model for {epochs} epochs, batch size {batchSize}");

        EvaluationTrainingSample stacked;
        lock (TrainingData.EvaluationTrainingData)
        {
            stacked = EvaluationTrainingSample.Stack(TrainingData.EvaluationTrainingData, false);
        }

        var optimizer = optim.AdamW(model.parameters(), lr: TrainingConfig.LearningRate, weight_decay: 0.001f);
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

                    Tensor predictions = model.forward(inputs);
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

                Tensor predictions = model.forward(inputs);
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
}

