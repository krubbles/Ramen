namespace Ramen.Training;

using System.Threading;
using Ramen.AI;

public class BasicGRPOTrainingRun : ITrainingRun
{
    public int RolloutSize { get; set; } = 3000;

    public int Epochs { get; set; } = 3;

    public float LearningRate { get; set; } = 3e-4f;

    public float Entropy { get; set; } = 0f;

    /// <summary>
    /// Clears prior policy samples, generates a GRPO rollout, then trains the model on that rollout.
    /// </summary>
    public void Step(IPolicyModel model)
    {
        // Generate a rollout where group size matches rollout size.
        IReadOnlyList<PolicyTrainingSample> trainingData = GRPOTrainingData.GenerateTrainingData(
            model, 
            trainingSampleCount: RolloutSize, 
            sampledSoftmaxCount: 10, 
            groupSize: 32);

        // Train on the freshly generated rollout.
        TrainingParams trainingParams = new(
            epochs: Epochs, 
            learningRate: LearningRate, 
            entropyCoeff: Entropy, 
            batchSize: 256);

        Training.TrainPolicyModelGRPO(model, trainingData, trainingParams, CancellationToken.None);
    }
}
