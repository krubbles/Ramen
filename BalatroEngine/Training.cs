namespace BalatroAI;

using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Training
{

    public static void TrainEvaluationModel(AIModels models, int epochs, int batchSize)
    {
        // stack data
        EvaluationTrainingSample stacked;
        lock (TrainingData.EvaluationData)
        {
            stacked = EvaluationTrainingSample.Stack(TrainingData.EvaluationData, false);
        }

        var optimizer = optim.Adam(models.Evaluation.parameters(), lr: TrainingConfig.LearningRate);
        var lossFunc = MSELoss();

        int samples = (int)stacked.Target.size(dim: 0);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float lossAvg = 0f;
            int batchCount = 0;
            for (int i = 0; i < samples; i += batchSize)
            {
                int end = Math.Min(i + batchSize, samples);
                var batchFullHands = stacked.GameStateTensors.FullHand[i..end];
                var batchOther = stacked.GameStateTensors.OtherState[i..end];
                var batchTargets = stacked.Target[i..end];

                optimizer.zero_grad();

                Tensor preds = models.Evaluation.forward(batchFullHands, batchOther);

                var loss = lossFunc.forward(preds, batchTargets);
                loss.backward();
                optimizer.step();

                lossAvg += loss.item<float>();
                batchCount++;
            }

            lossAvg /= Math.Max(1, batchCount);
            Console.WriteLine($"Eval Epoch {epoch} | Loss = {lossAvg}");
        }

        stacked.Dispose();
    }

    public static void TrainPolicyModel(AIModels models, int epochs, int batchSize)
    {
        PolicyTrainingSample stackedSamples;
        lock (TrainingData.PolicyTrainingData)
        {
            stackedSamples = PolicyTrainingSample.Stack(TrainingData.PolicyTrainingData, false);
        }

        var optimizer = optim.Adam(models.Policy.parameters(), lr: TrainingConfig.LearningRate);
        var lossFunc = CrossEntropyLoss();


        float lossAvg = 0;
        int batchCount = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {

            int fullSampleCount = (int)stackedSamples.Output.size(dim: 0);
            int trainSampleCount = fullSampleCount;


            lossAvg = 0;
            batchCount = 0;
            for (int i = 0; i < trainSampleCount; i += batchSize)
            {
                var batchFullHands = stackedSamples.GameStateTensors.FullHand[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOtherInputs = stackedSamples.GameStateTensors.OtherState[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchInUseMasks = stackedSamples.InUseMask[i..Math.Min(i + batchSize, trainSampleCount)];
                var batchOutputs = stackedSamples.Output[i..Math.Min(i + batchSize, trainSampleCount)];
                optimizer.zero_grad();
                Tensor fullHandEmbedded = models.EmbedCards(batchFullHands);
                var predictions = models.GetCardUseRewards(fullHandEmbedded, batchOtherInputs, batchInUseMasks);
                var loss = lossFunc.forward(predictions, batchOutputs);
                loss.backward();
                optimizer.step();
                lossAvg += loss.item<float>();
                batchCount++;

            }

            lossAvg /= batchCount;

            Console.WriteLine($"Epoch {epoch} | Loss = {lossAvg}");

            if (epoch % 10 == 9)
            {
                Console.WriteLine("Average Reward: " + Testing.GetAverageReward(models, 500));
            }
        }
    }
}

public struct EvaluationTrainingSample : IDisposable
{
    public GameStateTensors GameStateTensors;
    public Tensor Target; // (1) scalar

    public static EvaluationTrainingSample Stack(IReadOnlyList<EvaluationTrainingSample> samples, bool disposeInputs)
    {
        GameStateTensors[] gameStates = new GameStateTensors[samples.Count];
        Tensor[] targets = new Tensor[samples.Count];

        for (int i = 0; i < samples.Count; ++i)
        {
            gameStates[i] = samples[i].GameStateTensors;
            targets[i] = samples[i].Target;
        }

        EvaluationTrainingSample result = new()
        {
            GameStateTensors = GameStateTensors.Stack(gameStates, disposeInputs),
            Target = concat(targets, dim: 0)
        };

        if (disposeInputs)
        {
            for (int i = 0; i < samples.Count; ++i)
            {
                samples[i].Dispose();
            }
        }

        return result;
    }

    public void Dispose()
    {
        GameStateTensors.Dispose();
        Target.Dispose();
    }
}
