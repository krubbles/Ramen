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
        List<TrainingDataStats> stats = new();

        for (int i = 0; i < 1000; i++)
        {
            int multiplier = 1;

            TrainingData.EvaluationTrainingData.Clear();
            TrainingDataStats stat = TrainingData.GenerateEvaluationTrainingData(model, multiplier * 3000, 1f);
            stats.Add(stat);
            Training.TrainEvaluationModel(model, 5, 128, false);

            avgScores.Add(Testing.GetAverageScore(model, 150));

            if (true)
            {
                Console.WriteLine("Data:");
                System.Text.StringBuilder sb = new();
                bool simplifiedData = true;
                if (simplifiedData)
                    sb.AppendLine("Mean score Temp 0, Mean Score Temp 1, Move 1 Avg NLProb, Move 5 Avg NLProb");
                for (int row = 0; row < stats.Count; ++row)
                {
                    TrainingDataStats s = stats[row];
                    sb.Clear();
                    sb.Append(avgScores[row].ToString("F3"));
                    sb.Append(", ");
                    sb.Append(s.MeanReward.ToString("F3"));
                    sb.Append(", ");
                    if (simplifiedData)
                    {
                        sb.Append(s.AverageNLProb(0).ToString("F3"));
                        sb.Append(", ");
                        sb.Append(s.AverageNLProb(4).ToString("F3"));
                        foreach (int count in s.CountByTier)
                        {
                            sb.Append(", ");
                            sb.Append(count);
                        }
                    }
                    else
                    {
                    }
                    Console.WriteLine(sb.ToString());
                }
            }

        }
    }
}