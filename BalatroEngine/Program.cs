namespace Ramen.AI;

using System;
using static TorchSharp.torch;
using Ramen.Game;
class Program
{
    static void Main()
    {
        for (int i = 0; i < 0; ++i)
        {
            GameData gd = new();
            gd.Hands = 2;
            gd.Discards = 0;
            GameState gs = new(gd);
            (Move bestMove, float bestProb) = Testing.GetBestDiscard(gs, 300);
            Console.WriteLine(gs.ToString());
            Console.WriteLine(bestMove.ToString());
            Console.WriteLine(bestProb.ToString());
        }
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
        float entropy = 0f;

        for (int i = 0; i < 1000; i++)
        {
            int multiplier = 1;
            TrainingData.EvaluationTrainingData.Clear();
            TrainingDataStats stat = TrainingData.GenerateEvaluationTrainingData(model, 5000, 1f);
            stats.Add(stat);
            Training.TrainEvaluationModel(model, 5, 128, entropy, false);
            entropy *= MathF.Pow(0.5f, 1f / 10);
            avgScores.Add(Testing.GetAverageScore(model, 1000 * multiplier));

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
                        sb.Append(s.AverageNLProb(1).ToString("F3"));
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
            if (i % 20 == 19)
            {
                Console.WriteLine();
                GameData gd = new();
                gd.Hands = 2;
                gd.Discards = 0;
                GameState gs = new(gd);
                gs.AdvanceToNextPlayerChoice();
                Console.WriteLine(gs);
                (Move bestMove, float bestProb) = Testing.GetBestDiscard(gs, 300);
                RamenAgent agent = new(gs, model);
                agent.MakeMoveStochastic(0.001f);
                Console.WriteLine("Agent Move:" + gs.MoveState.MoveHistory[^1]);
                Console.WriteLine();
            }
        }
    }
}