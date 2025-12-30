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

    private static readonly object EvaluationDataLock = new();

    public const int HandSize = 8;


    public static Card[] ExtractHand(Tensor fullHand, Tensor hand)
    {
        Tensor capped = min(fullHand, hand);
        float[] data = capped.data<float>().ToArray();
        capped.Dispose();
        List<Card> cards = new();
        List<float> extras = new();
        for (int i = 0; i < data.Length; ++i)
        {
            int count = (int)Math.Round(data[i]);
            if (count == 0)
                continue;
            float extra = data[i] - count;
            Card card = new(rank: (i / 4) + 2, suit: (Suit)(i % 4 + 1));
            for (int j = 0; j < count; ++j)
            {
                cards.Add(card);
                extras.Add(extra);
                extra++;
            }
        }
        while (cards.Count > 5)
        {
            int toRemoveIndex = 0;
            float lowestExtra = float.MaxValue;
            for (int i = 0; i < cards.Count; ++i)
            {
                if (extras[i] < lowestExtra)
                {
                    lowestExtra = extras[i];
                    toRemoveIndex = i;
                }
            }
            cards.RemoveAt(toRemoveIndex);
            extras.RemoveAt(toRemoveIndex);
        }
        if (cards.Count > 0)
        {
            return cards.ToArray();
        }
        else
        {
            float strongestActivation = float.MinValue;
            int strongestActivationIndex = 0;
            for (int i = 0; i < data.Length; ++i)
            {
                float activation = data[i];
                if (activation > strongestActivation)
                {
                    strongestActivation = activation;
                    strongestActivationIndex = i;
                }
            }
            return [new(rank: (strongestActivationIndex / 4) + 2, suit: (Suit)(strongestActivationIndex % 4 + 1))];
        }
    }


    static void Main()
    {
        random.manual_seed(0);
        var device = CPU;
        
        AIModels models = new();

        long totalTrainableParams = 0;
        foreach (var param in models.Policy.parameters())
        {
            if (param.requires_grad)
                totalTrainableParams += param.numel();
        }
        Console.WriteLine($"Total number of trainable policy parameters: {totalTrainableParams}");

        totalTrainableParams = 0;
        foreach (var param in models.Evaluation.parameters())
        {
            if (param.requires_grad)
                totalTrainableParams += param.numel();
        }
        Console.WriteLine($"Total number of trainable evaluation parameters: {totalTrainableParams}");


        TrainingData.GenerateEvaluationTrainingData(models, 30000);
        Training.TrainEvaluationModel(models, epochs: 4, batchSize: 32, validate: true);
        TrainingData.GeneratePolicyTrainingData(models, samples: 4000);
        Training.TrainPolicyModel(models, epochs: 4, batchSize: 64, validate: true);


        TrainingData.EvaluationTrainingData.Clear();
        TrainingData.PolicyTrainingData.Clear();

        ShowExampleMoveRewards();

        TrainingData.GenerateEvaluationTrainingData(models, 40000);
        Training.TrainEvaluationModel(models, epochs: 4, batchSize: 32, validate: true);
        TrainingData.GeneratePolicyTrainingData(models, samples: 5000);
        Training.TrainPolicyModel(models, epochs: 4, batchSize: 64, validate: true);

        TrainingData.EvaluationTrainingData.Clear();
        TrainingData.PolicyTrainingData.Clear();


        for (int i = 0; i < 10; ++i)
        {
            int evalSampleCount = 200000;
            int policySampleCount = 25000;
            TrainingData.GenerateEvaluationTrainingData(models, samples: evalSampleCount);
            Training.TrainEvaluationModel(models, epochs: 6, batchSize: 64, validate: true);

            TrainingData.GeneratePolicyTrainingData(models, samples: policySampleCount);
            Training.TrainPolicyModel(models, epochs: 6, batchSize: 64, validate: true);

            ShowExampleMoveRewards();

            TrainingData.EvaluationTrainingData.Clear();
            TrainingData.PolicyTrainingData.Clear();
        }

        void ShowExampleMoveRewards()
        {
            Console.WriteLine();
            Console.WriteLine("--- Example Move Rewards ---");
            for (int i = 0; i < 5; ++i)
            {
                FastRandom r = FastRandom.SeededByClock();
                GameState gs = new GameState(new());
                gs.Reseed(r.Next());
                gs.StartRound();
                Console.WriteLine(gs.HandToString());
                AIGameState aigs = new(gs, models);
                float[] rewards = (models.GetCardUseRewards(aigs.GameStateTensors.Hand, aigs.GameStateTensors.OtherState, aigs.InUseMaskTensor)).div(1f).softmax(dim: 1).data<float>().ToArray();
                Console.WriteLine(LoggingUtility.FormatArray(rewards));
            }
            Console.WriteLine("---------------------------");
            Console.WriteLine();
        }
    }


}