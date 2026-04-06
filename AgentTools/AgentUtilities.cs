namespace Ramen.AgentTools;

using Ramen.Game;

public static class AgentUtilities
{
    public static float GetTemperatureForTargetMaxProbability(ReadOnlySpan<float> logits, float targetProbability)
    {
        return GetTemperatureForTargetTopProbabilityMass(
            logits: logits,
            topProbabilityCount: 1,
            targetProbabilityMass: targetProbability);
    }


    public static float GetTemperatureForTargetTopProbabilityMass(
        ReadOnlySpan<float> logits,
        int topProbabilityCount,
        float targetProbabilityMass)
    {
        if (logits.Length == 0)
            throw new ArgumentException("Logits must not be empty.", nameof(logits));

        if (topProbabilityCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(topProbabilityCount), "Top probability count must be positive.");

        if (!float.IsFinite(targetProbabilityMass) || targetProbabilityMass <= 0f || targetProbabilityMass > 1f)
            throw new ArgumentOutOfRangeException(nameof(targetProbabilityMass), "Target probability mass must be in the range (0, 1].");

        int effectiveTopProbabilityCount = Math.Min(topProbabilityCount, logits.Length);

        float minimumAchievableProbabilityMass = (float)effectiveTopProbabilityCount / logits.Length;

        const float minimumTemperature = 1e-4f;
        const float probabilityTolerance = 1e-5f;

        float maximumAchievableProbabilityMass = GetTopProbabilityMass(
            logits: logits,
            temperature: minimumTemperature,
            topProbabilityCount: effectiveTopProbabilityCount);
        float clampedTargetProbabilityMass = MathF.Min(
            MathF.Max(targetProbabilityMass, minimumAchievableProbabilityMass),
            maximumAchievableProbabilityMass);

        if (maximumAchievableProbabilityMass - minimumAchievableProbabilityMass <= probabilityTolerance)
            return float.PositiveInfinity;

        if (clampedTargetProbabilityMass <= minimumAchievableProbabilityMass + probabilityTolerance)
            return float.PositiveInfinity;

        if (clampedTargetProbabilityMass >= maximumAchievableProbabilityMass - probabilityTolerance)
            return minimumTemperature;

        float lowTemperature = minimumTemperature;
        float highTemperature = 1f;
        float highProbabilityMass = GetTopProbabilityMass(
            logits: logits,
            temperature: highTemperature,
            topProbabilityCount: effectiveTopProbabilityCount);
        while (highProbabilityMass > clampedTargetProbabilityMass)
        {
            highTemperature *= 2f;
            highProbabilityMass = GetTopProbabilityMass(
                logits: logits,
                temperature: highTemperature,
                topProbabilityCount: effectiveTopProbabilityCount);

            if (!float.IsFinite(highTemperature) || highTemperature > 1e12f)
                return float.PositiveInfinity;
        }

        for (int iteration = 0; iteration < 64; ++iteration)
        {
            float midTemperature = (lowTemperature + highTemperature) * 0.5f;
            float midProbabilityMass = GetTopProbabilityMass(
                logits: logits,
                temperature: midTemperature,
                topProbabilityCount: effectiveTopProbabilityCount);
            if (midProbabilityMass > clampedTargetProbabilityMass)
                lowTemperature = midTemperature;
            else
                highTemperature = midTemperature;
        }

        return highTemperature;
    }


    public static float[] SafeSoftmax(ReadOnlySpan<float> logits, float temp)
    {
        if (logits.Length == 0)
            return [];

        float[] probabilities = new float[logits.Length];

        float invTemp = 1f / MathF.Max(temp, 1e-4f);
        float maxLogit = float.NegativeInfinity;

        for (int index = 0; index < logits.Length; ++index)
        {
            float scaledLogit = logits[index] * invTemp;
            if (!float.IsFinite(scaledLogit))
                scaledLogit = -1e9f;

            if (scaledLogit > maxLogit)
                maxLogit = scaledLogit;

            probabilities[index] = scaledLogit;
        }

        float sum = 0f;
        for (int index = 0; index < logits.Length; ++index)
        {
            float shiftedLogit = probabilities[index] - maxLogit;
            float expValue = shiftedLogit < -80f ? 0f : MathF.Exp(shiftedLogit);
            if (!float.IsFinite(expValue))
                expValue = 0f;

            probabilities[index] = expValue;
            sum += expValue;
        }

        if (sum <= 0f || !float.IsFinite(sum))
        {
            float uniformProb = 1f / logits.Length;
            for (int index = 0; index < logits.Length; ++index)
                probabilities[index] = uniformProb;

            return probabilities;
        }

        for (int index = 0; index < logits.Length; ++index)
            probabilities[index] /= sum;

        return probabilities;
    }


    public static int SampleIndex(ReadOnlySpan<float> probabilities, FastRandom random)
    {
        float sample = random.NextPortion();
        float runningTotal = 0f;
        for (int index = 0; index < probabilities.Length; ++index)
        {
            runningTotal += probabilities[index];
            if (sample <= runningTotal)
                return index;
        }

        return probabilities.Length - 1;
    }


    static float GetTopProbabilityMass(ReadOnlySpan<float> logits, float temperature, int topProbabilityCount)
    {
        float[] probabilities = SafeSoftmax(logits, temperature);

        int effectiveTopProbabilityCount = Math.Min(topProbabilityCount, probabilities.Length);
        if (effectiveTopProbabilityCount <= 0)
            return 0f;

        float[] topProbabilities = new float[effectiveTopProbabilityCount];
        int topProbabilityLength = 0;
        for (int index = 0; index < probabilities.Length; ++index)
        {
            float probability = probabilities[index];

            if (topProbabilityLength < effectiveTopProbabilityCount)
            {
                topProbabilities[topProbabilityLength] = probability;
                topProbabilityLength++;
                for (int insertIndex = topProbabilityLength - 1; insertIndex > 0; --insertIndex)
                {
                    if (topProbabilities[insertIndex] <= topProbabilities[insertIndex - 1])
                        break;

                    (topProbabilities[insertIndex], topProbabilities[insertIndex - 1]) =
                        (topProbabilities[insertIndex - 1], topProbabilities[insertIndex]);
                }

                continue;
            }

            if (probability <= topProbabilities[^1])
                continue;

            topProbabilities[^1] = probability;
            for (int insertIndex = topProbabilities.Length - 1; insertIndex > 0; --insertIndex)
            {
                if (topProbabilities[insertIndex] <= topProbabilities[insertIndex - 1])
                    break;

                (topProbabilities[insertIndex], topProbabilities[insertIndex - 1]) =
                    (topProbabilities[insertIndex - 1], topProbabilities[insertIndex]);
            }
        }

        float topProbabilityMass = 0f;
        for (int index = 0; index < topProbabilityLength; ++index)
            topProbabilityMass += topProbabilities[index];

        return topProbabilityMass;
    }
}
