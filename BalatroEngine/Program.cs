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

        Console.WriteLine("Av Score: " + Testing.GetAverageScore(model));

        int dataSize = 100;
        for (int i = 0; i < 100; i++)
        {
            TrainingData.GenerateEvaluationTrainingData(model, dataSize);
            Training.TrainEvaluationModel(model, 4, 256, false);
            Console.WriteLine("Av Score: " + Testing.GetAverageScore(model));
            dataSize += 100;
        }
    }
}