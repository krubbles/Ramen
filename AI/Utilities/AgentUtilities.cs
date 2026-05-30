namespace Ramen.AI;

public static class AgentUtilities
{
    public static UseHandMove MoveForPolicyIndex(GameState state, int index)
    {
        int[][] useHandOptions = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        return new(state.HandState.RemainingDiscards >= 1 && index % 2 == 1, useHandOptions[index / 2]);
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
}
