namespace Ramen.AI;

using System;
using static TorchSharp.torch;

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
            int multiplier = 1;

            if (i % 10 == 9)
            {
                Testing.GetAverageScore(model, 5, true);
                avgScores.Add(Testing.GetAverageScore(model, multiplier * 1000));
                Console.WriteLine("scores:");
                foreach (float score in avgScores)
                    Console.WriteLine(score);
            }
            TrainingData.EvaluationTrainingData.Clear();
            TrainingData.GenerateEvaluationTrainingData(model, multiplier * 1000, 1f);
            Training.entropyCoeff = 0.02f;
            Training.kldCoeff = 0.05f;
            if (i > 300)
                Training.entropyCoeff *= 0.1f;
            Training.TrainEvaluationModel(model, 5,  64, false);
        }
    }
}