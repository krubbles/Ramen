namespace BalatroAI;

using System;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using BalatroAI;
using BalatroAI.ConsoleApp;
using System.Buffers;

class Program
{
    static void Main()
    {
        Console.WriteLine($"Num threads: {get_num_threads()}");
        //set_num_threads(1);
        //set_num_interop_threads(1);

        random.manual_seed(0);
        var device = CPU;

        GameEvalModel model = new();

        long totalTrainableParams = 0;
        foreach (var param in model.parameters())
        {
            if (param.requires_grad)
                totalTrainableParams += param.numel();
        }
        Console.WriteLine($"Total number of trainable evaluation parameters: {totalTrainableParams}");

        Console.WriteLine("Av Score: " + Testing.GetAverageScore(model, 100));

        List<float> avgScores = new();
        for (int i = 0; i < 1000; i++)
        {
            if (i % 10 == 9)
            {
                avgScores.Add(Testing.GetAverageScore(model, 100));
                Console.WriteLine("scores:");
                foreach (float score in avgScores)
                    Console.WriteLine(score);
            }
            TrainingData.EvaluationTrainingData.Clear();
            TrainingData.GenerateEvaluationTrainingData(model, i < 100 ? 1000 : 2000);
            Training.TrainEvaluationModel(model, 5,  128, true);
        }
    }
}