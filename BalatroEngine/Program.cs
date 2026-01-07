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

        List<float> avgScores = new();
        List<float>[] moveNLProbs = new List<float>[8];
        List<float>[] moveAverageAttributions = new List<float>[8];
        List<float> meanTemp1Scores = new();
        
        for (int i = 0; i < 8; ++i)
        {
            moveNLProbs[i] = new();
            moveAverageAttributions[i] = new();
        }

        for (int i = 0; i < 1000; i++)
        {
            int multiplier = 1;

            TrainingData.EvaluationTrainingData.Clear();
            TrainingDataStats stats = TrainingData.GenerateEvaluationTrainingData(model, multiplier * 3000, 1f);
            Training.TrainEvaluationModel(model, 5, 64, false);
            meanTemp1Scores.Add(stats.MeanReward);

            for (int j = 0; j < moveNLProbs.Length; ++j)
            {
                moveNLProbs[j].Add(stats.AverageNLProb(j));
                moveAverageAttributions[j].Add(stats.AverageAttribution(j));
            }

            if (true)
            {
                Console.WriteLine("Data:");
                System.Text.StringBuilder sb = new();
                for (int row = 0; row < meanTemp1Scores.Count; ++row)
                {
                    sb.Clear();
                    sb.Append(meanTemp1Scores[row]);
                    sb.Append(", ");
                    for (int col = 0; col < 7; ++col)
                    {
                        sb.Append(moveNLProbs[col][row].ToString());
                        sb.Append(", ");
                    }
                    for (int col = 0; col < 7; ++col)
                    {
                        sb.Append(moveAverageAttributions[col][row].ToString());
                        if (col != 6)
                            sb.Append(", ");
                    }
                    Console.WriteLine(sb.ToString());
                }
            }

        }
    }
}