namespace Ramen.AI;

public interface IRolloutAnalyzer
{
    string Name { get; }
    float Value { get; }

    void ObserveCompletedTrajectory(IReadOnlyList<PolicyTrainingSample> samples, float finalReward);
}

public sealed class AverageRewardRolloutAnalyzer : IRolloutAnalyzer
{
    float _rewardSum;
    int _trajectoryCount;

    public string Name => "average_reward";
    public float Value => _trajectoryCount == 0 ? 0f : _rewardSum / _trajectoryCount;

    public void ObserveCompletedTrajectory(IReadOnlyList<PolicyTrainingSample> samples, float finalReward)
    {
        _rewardSum += finalReward;
        _trajectoryCount++;
    }
}

public sealed class AverageEntropyRolloutAnalyzer : IRolloutAnalyzer
{
    float _entropySum;
    int _sampleCount;

    public string Name => "average_entropy";
    public float Value => _sampleCount == 0 ? 0f : _entropySum / _sampleCount;

    public void ObserveCompletedTrajectory(IReadOnlyList<PolicyTrainingSample> samples, float finalReward)
    {
        for (int sampleIndex = 0; sampleIndex < samples.Count; ++sampleIndex)
        {
            if (!AnnotationDataUtils.TryDecodePolicyAnnotation(samples[sampleIndex].PolicyAnnotation, out float[] policy))
                continue;

            float entropy = 0f;
            for (int moveIndex = 0; moveIndex < policy.Length; ++moveIndex)
            {
                float probability = policy[moveIndex];
                if (probability > 0f)
                    entropy -= probability * MathF.Log(probability);
            }

            _entropySum += entropy;
            _sampleCount++;
        }
    }
}

public readonly record struct RolloutAnalysis(float AverageReward, float AverageEntropy);
