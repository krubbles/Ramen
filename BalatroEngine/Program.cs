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
            Training.TrainEvaluationModel(model, 5, 128, false);
            meanTemp1Scores.Add(stats.MeanReward);

            for (int j = 0; j < moveNLProbs.Length; ++j)
            {
                moveNLProbs[j].Add(stats.AverageNLProb(j));
                moveAverageAttributions[j].Add(stats.AverageAttribution(j));
            }

            avgScores.Add(Testing.GetAverageScore(model, 150));

            if (true)
            {
                Console.WriteLine("Data:");
                System.Text.StringBuilder sb = new();
                bool simplifiedData = true;
                if (simplifiedData)
                    sb.AppendLine("Mean score Temp 0, Mean Score Temp 1, Move 1 Avg NLProb, Move 5 Avg NLProb");
                for (int row = 0; row < meanTemp1Scores.Count; ++row)
                {
                    sb.Clear();
                    sb.Append(avgScores[row].ToString("F3"));
                    sb.Append(", ");
                    sb.Append(meanTemp1Scores[row].ToString("F3"));
                    sb.Append(", ");
                    if (simplifiedData)
                    {
                        sb.Append(moveNLProbs[0][row].ToString("F3"));
                        sb.Append(", ");
                        sb.Append(moveNLProbs[4][row].ToString("F3"));
                    }
                    else
                    {
                        for (int col = 0; col < 7; ++col)
                        {
                            sb.Append(moveNLProbs[col][row].ToString("F3"));
                            sb.Append(", ");
                        }
                        for (int col = 0; col < 7; ++col)
                        {
                            sb.Append(moveAverageAttributions[col][row].ToString("F3"));
                            if (col != 6)
                                sb.Append(", ");
                        }
                    }
                    Console.WriteLine(sb.ToString());
                }
            }

        }
    }
}