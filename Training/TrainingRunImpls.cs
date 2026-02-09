namespace Ramen.Training;

using System.Threading;
using Ramen.AI;

public class BasicGRPOTrainingRun : ITrainingRun
{
    public int RolloutSize { get; set; } = 500;

    public int Epochs { get; set; } = 3;

    public float LearningRate { get; set; } = 3e-4f;

    public float Entropy { get; set; } = 0f;

    /// <summary>
    /// Clears prior policy samples, generates a GRPO rollout, then trains the model on that rollout.
    /// </summary>
    public void Step(IPolicyModel model)
    {
        // Reset any previously accumulated policy training samples.
        TrainingData.Clear();

        // Generate a rollout where group size matches rollout size.
        GRPOTrainingData.GenerateTrainingData(model, games: RolloutSize, sampleCount: 10, groupSize: RolloutSize);

        // Train on the freshly generated rollout.
        TrainingParams trainingParams = new(epochs: Epochs, learningRate: LearningRate, entropyCoeff: Entropy);
        Training.TrainPolicyModelGRPO(model, trainingParams, CancellationToken.None);
    }
}
