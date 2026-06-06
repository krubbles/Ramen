namespace Ramen.AI;

public static class RolloutAnalyzers
{
    public static float AverageReward(RolloutData rolloutData)
    {
        IReadOnlyList<GameState> trajectoryGameStates = rolloutData.Trajectories;
        if (trajectoryGameStates.Count == 0)
            return 0f;

        float rewardSum = 0f;
        for (int trajectoryIndex = 0; trajectoryIndex < trajectoryGameStates.Count; ++trajectoryIndex)
            rewardSum += PolicyValueNetworkTraining.GetReward(trajectoryGameStates[trajectoryIndex]);

        return rewardSum / trajectoryGameStates.Count;
    }


    public static float AverageEntropy(RolloutData rolloutData)
    {
        float entropySum = 0f;
        int annotationCount = 0;
        IReadOnlyList<GameState> trajectories = rolloutData.Trajectories;

        for (int trajectoryIndex = 0; trajectoryIndex < trajectories.Count; ++trajectoryIndex)
        {
            List<Move> moveHistory = trajectories[trajectoryIndex].MoveState.MoveHistory;
            for (int moveIndex = 0; moveIndex < moveHistory.Count; ++moveIndex)
            {
                if (moveHistory[moveIndex] is not AnnotatingDataMove annotation ||
                    !AnnotationDataUtils.TryDecodePolicyAnnotation(annotation, out float[] policy))
                    continue;

                entropySum -= policy[0] * MathF.Log(MathF.Max(policy[0], 1e-9f));
                annotationCount++;
            }
        }

        return annotationCount == 0 ? 0f : entropySum / annotationCount;
    }

    public static float AverageGradNorm(RolloutData rolloutData)
    {
        IReadOnlyList<float> gradNorms = rolloutData.GradNorms;
        if (gradNorms.Count == 0)
            return 0f;

        float gradNormSum = 0f;
        for (int gradNormIndex = 0; gradNormIndex < gradNorms.Count; ++gradNormIndex)
            gradNormSum += gradNorms[gradNormIndex];

        return gradNormSum / gradNorms.Count;
    }
}
