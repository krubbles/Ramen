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
        foreach (var named in models.parameters())
        {
            if (named.requires_grad)
                totalTrainableParams += named.numel();
        }
        Console.WriteLine($"Total number of trainable parameters: {totalTrainableParams}");

        ShowExampleMoveRewards();

        TrainingData.GeneratePolicyTrainingData(models, samples: 4000, logCount: 10, log: true);
        Training.TrainPolicyModel(models, epochs: 10, batchSize: 64);

        void ShowExampleMoveRewards()
        {
            for (int i = 0; i < 3; ++i)
            {
                FastRandom r = FastRandom.SeededByClock();
                GameState gs = new GameState(new());
                gs.Reseed(r.Next());
                gs.StartRound();
                Console.WriteLine(gs.HandToString());
                AIGameState aigs = new(gs, models);
                float[] rewards = (models.GetCardUseRewards(aigs.HandTensor, aigs.GameStateTensors.OtherState, aigs.InUseMaskTensor)).div(1f).softmax(dim: 1).data<float>().ToArray();
                Console.WriteLine(LoggingUtility.FormatArray(rewards));
            }
        }
    }



    class ResidualMLP : Module<Tensor, Tensor>
    {
        private ModuleList<Linear> upLayers = new();
        private ModuleList<Linear> downLayers = new();

        private ModuleList<LayerNorm> norms = new();
        private ModuleList<GELU> activations = new();

        public ResidualMLP(int size, int depth) : base("ResidualMLP")
        {
            for (int i = 0; i < depth; ++i)
            {
                upLayers.append(Linear(size, size * 4));
                downLayers.append(Linear(size * 4, size));
                activations.append(GELU());
                norms.append(LayerNorm(size));
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            for (int i = 0; i < upLayers.Count; i++)
            {
                Tensor normed = norms[i].forward(x);
                Tensor up = upLayers[i].forward(x);
                Tensor activated = activations[i].forward(up);
                Tensor down = downLayers[i].forward(activated);
                x = x / upLayers.Count + down;
            }
            return x;
        }
    }
}