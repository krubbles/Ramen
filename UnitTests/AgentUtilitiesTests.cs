namespace Ramen.UnitTests;

using Ramen.AgentTools;

public class AgentUtilitiesTests
{
    [Test]
    public void GetTemperatureForTargetMaxProbabilityMatchesRequestedProbability()
    {
        float[] logits = [2f, 0.5f, -1f];
        float targetProbability = 0.7f;

        float temperature = AgentUtilities.GetTemperatureForTargetMaxProbability(logits, targetProbability);
        float[] probabilities = AgentUtilities.SafeSoftmax(logits, temperature);

        Assert.That(temperature, Is.GreaterThan(0f));
        Assert.That(GetTopProbabilityMass(probabilities, topProbabilityCount: 1), Is.EqualTo(targetProbability).Within(1e-4f));
    }


    [Test]
    public void GetTemperatureForTargetMaxProbabilityReturnsInfinityForUniformTarget()
    {
        float[] logits = [3f, 1f, -2f, -4f];
        float targetProbability = 1f / logits.Length;

        float temperature = AgentUtilities.GetTemperatureForTargetMaxProbability(logits, targetProbability);

        Assert.That(float.IsPositiveInfinity(temperature), Is.True);
    }


    [Test]
    public void GetTemperatureForTargetMaxProbabilityClampsToMaximumAchievableProbability()
    {
        float[] logits = [5f, 5f, 1f];

        float temperature = AgentUtilities.GetTemperatureForTargetMaxProbability(logits, targetProbability: 0.9f);
        float[] probabilities = AgentUtilities.SafeSoftmax(logits, temperature);

        Assert.That(temperature, Is.EqualTo(1e-4f).Within(1e-8f));
        Assert.That(GetTopProbabilityMass(probabilities, topProbabilityCount: 1), Is.EqualTo(0.5f).Within(1e-4f));
    }


    [Test]
    public void GetTemperatureForTargetMaxProbabilityClampsToUniformWhenTargetIsTooSmall()
    {
        float[] logits = [2f, 0.5f, -1f];

        float temperature = AgentUtilities.GetTemperatureForTargetMaxProbability(logits, targetProbability: 0.2f);
        float[] probabilities = AgentUtilities.SafeSoftmax(logits, temperature);

        Assert.That(float.IsPositiveInfinity(temperature), Is.True);
        Assert.That(GetTopProbabilityMass(probabilities, topProbabilityCount: 1), Is.EqualTo(1f / logits.Length).Within(1e-4f));
    }


    [Test]
    public void GetTemperatureForTargetTopProbabilityMassMatchesRequestedProbability()
    {
        float[] logits = [4f, 3f, 2f, 1f, 0f, -1f];
        float targetProbabilityMass = 0.9f;

        float temperature = AgentUtilities.GetTemperatureForTargetTopProbabilityMass(
            logits: logits,
            topProbabilityCount: 4,
            targetProbabilityMass: targetProbabilityMass);
        float[] probabilities = AgentUtilities.SafeSoftmax(logits, temperature);

        Assert.That(temperature, Is.GreaterThan(0f));
        Assert.That(GetTopProbabilityMass(probabilities, topProbabilityCount: 4), Is.EqualTo(targetProbabilityMass).Within(1e-4f));
    }


    [Test]
    public void GetTemperatureForTargetTopProbabilityMassClampsWhenTopCountExceedsMoveCount()
    {
        float[] logits = [2f, 1f, 0f];

        float temperature = AgentUtilities.GetTemperatureForTargetTopProbabilityMass(
            logits: logits,
            topProbabilityCount: 4,
            targetProbabilityMass: 0.9f);
        float[] probabilities = AgentUtilities.SafeSoftmax(logits, temperature);

        Assert.That(float.IsPositiveInfinity(temperature), Is.True);
        Assert.That(GetTopProbabilityMass(probabilities, topProbabilityCount: 4), Is.EqualTo(1f).Within(1e-4f));
    }


    static float GetTopProbabilityMass(ReadOnlySpan<float> probabilities, int topProbabilityCount)
    {
        int effectiveTopProbabilityCount = Math.Min(topProbabilityCount, probabilities.Length);
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
