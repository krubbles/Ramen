namespace Ramen.UnitTests;

using System.Threading;
using NUnit.Framework;
using Ramen.AI;
using Ramen.Training;
using static TorchSharp.torch;

public class GRPOTrainingTests
{
    [Test]
    public void TrainPolicyModelGRPO_DoesNotThrow()
    {
        using var scope = NewDisposeScope();

        IPolicyModel model = new PolicyModel();
        TrainingData.Clear();

        TrainingDataStats stats = GRPOTrainingData.GenerateTrainingData(model, games: 1, sampleCount: 4, groupSize: 1);
        Assert.That(stats.GamesCount, Is.EqualTo(1));
        Assert.That(TrainingData.PolicyData.Count, Is.GreaterThan(0));

        TrainingParams trainingParams = new(epochs: 1, batchSize: 2, learningRate: 1e-5f, entropyCoeff: 0f, kldCoeff: 0f);
        CancellationTokenSource cancel = new();

        Assert.That(() => Training.TrainPolicyModelGRPO(model, trainingParams, cancel.Token), Throws.Nothing);
    }
}
