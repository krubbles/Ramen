#define STACKLESS
namespace BalatroAI;

using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Training
{
    public static void TrainEvaluationModelStackless(GameEvalModel model, int epochs, int batchSize, bool validate = false)
    {
        Console.WriteLine($"Training evaluation model for {epochs} epochs, batch size {batchSize}");

        var optimizer = optim.AdamW(model.parameters(), lr: TrainingConfig.LearningRate, weight_decay: 0.001f);
        var lossFunc = CrossEntropyLoss();

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
                    Tensor predictions = model.forward(TrainingData.EvaluationTrainingData[i].GameStateTensors);
                    var loss = lossFunc.forward(predictions, TrainingData.EvaluationTrainingData[i].Target);
                    valLossAvg += loss.item<float>();
                    valBatchCount++;
                }
            }

            valLossAvg /= Math.Max(1, valBatchCount);
            float trainLossAvg = 0f;
            int trainBatchCount = 0;

            optimizer.zero_grad();

            for (int i = 0; i < trainCount; i += batchSize)
            {
                Tensor predictions = model.forward(TrainingData.EvaluationTrainingData[i].GameStateTensors).squeeze(1);
                var loss = lossFunc.forward(predictions, TrainingData.EvaluationTrainingData[i].Target.squeeze(1));
                loss.backward();

                trainLossAvg += loss.item<float>();
                trainBatchCount++;
            }

            optimizer.step();

            trainLossAvg /= Math.Max(1, trainBatchCount);



            Console.WriteLine($"Eval Epoch {epoch} | Train Loss = {trainLossAvg} | Val Loss = {valLossAvg}");
        }
    }

    public static void TrainEvaluationModel(GameEvalModel model, int epochs, int batchSize, bool validate = false)
    {
        Console.WriteLine($"Training evaluation model for {epochs} epochs, batch size {batchSize}");

        EvaluationTrainingSample stacked;
        lock (TrainingData.EvaluationTrainingData)
        {
            stacked = EvaluationTrainingSample.Stack(TrainingData.EvaluationTrainingData, false);
        }

        stacked.Shuffle();

        var optimizer = optim.AdamW(model.parameters(), lr: TrainingConfig.LearningRate, weight_decay: 0.001f);
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
                    GameStateTensors inputs = stacked.GameStateTensors.GetBatch(i, end);
                    Tensor targets = stacked.Target[i..end];

                    Tensor predictions = model.forward(inputs);
                    var loss = lossFunc.forward(predictions, targets);
                    valLossAvg += loss.item<float>();
                    valBatchCount++;
                }
            }

            valLossAvg /= Math.Max(1, valBatchCount);
            float trainLossAvg = 0f;
            int trainBatchCount = 0;

            optimizer.zero_grad();

            for (int i = 0; i < trainCount; i += batchSize)
            {
                int end = Math.Min(i + batchSize, samples);
                GameStateTensors inputs = stacked.GameStateTensors.GetBatch(i, end);
                Tensor targets = stacked.Target[i..end];


                Tensor predictions = model.forward(inputs);

                var loss = lossFunc.forward(predictions, targets);
                loss.backward();

                trainLossAvg += loss.item<float>();
                trainBatchCount++;
            }

            optimizer.step();

            trainLossAvg /= Math.Max(1, trainBatchCount);



            Console.WriteLine($"Eval Epoch {epoch} | Train Loss = {trainLossAvg} | Val Loss = {valLossAvg}");
        }

        stacked.Dispose();
    }
}

