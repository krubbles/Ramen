namespace Ramen.AI;

/// <summary>
/// Measures how well a small move tower's top-K ranking covers a large tower's probability
/// mass, which is the question a two-stage cascade turns on: if the large tower's choice is
/// reliably inside the small tower's top-K, the small tower can filter candidates for it.
/// <para>
/// The headline number is the mean missed mass — the large tower's probability on moves the
/// small tower left out of its top-K. That is exactly the cascade's failure rate. A mean of
/// the per-state negative log would instead be dominated by well-covered states and would
/// barely register the badly-covered tail that actually matters.
/// </para>
/// </summary>
public static class LeafCoverageStats
{
    public static readonly int[] KValues = [1, 2, 4, 8, 16, 32, 64, 128];

    static readonly double[] _missedMassSums = new double[KValues.Length];
    static readonly double[] _argmaxHitSums = new double[KValues.Length];
    static long _stateCount;

    public static bool Enabled;

    public static void Clear()
    {
        Array.Clear(_missedMassSums);
        Array.Clear(_argmaxHitSums);
        _stateCount = 0;
    }

    /// <summary>
    /// Accumulates one batch. Both arguments are full-move logits for the same states, already
    /// masked so illegal moves carry the mask logit.
    /// </summary>
    public static void Accumulate(Tensor smallLogits, Tensor largeLogits)
    {
        if (!Enabled)
            return;

        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        long stateCount = largeLogits.size(0);
        long moveCount = largeLogits.size(1);
        Tensor largeProbs = softmax(largeLogits, dim: 1);
        Tensor largeArgmax = largeProbs.argmax(dim: 1, keepdim: true);

        // One host round-trip for every K, rather than one per K per statistic.
        Tensor missedByK = zeros(KValues.Length, dtype: ScalarType.Float32, device: largeLogits.device);
        Tensor hitByK = zeros(KValues.Length, dtype: ScalarType.Float32, device: largeLogits.device);
        for (int kIndex = 0; kIndex < KValues.Length; ++kIndex)
        {
            long k = Math.Min(KValues[kIndex], moveCount);
            (Tensor _, Tensor topIndices) = smallLogits.topk((int)k, dim: 1, largest: true, sorted: false);

            // Illegal moves carry zero probability under the mask, so a K larger than the legal
            // move count simply covers everything and contributes no missed mass.
            Tensor coveredMass = largeProbs.gather(dim: 1, index: topIndices).sum(dim: 1);
            missedByK[kIndex] = (1f - coveredMass).clamp(0f, 1f).sum();

            Tensor argmaxHit = topIndices.eq(largeArgmax).any(dim: 1).to_type(ScalarType.Float32);
            hitByK[kIndex] = argmaxHit.sum();
        }

        float[] missed = [.. stack([missedByK, hitByK]).reshape([-1]).cpu().data<float>()];
        for (int kIndex = 0; kIndex < KValues.Length; ++kIndex)
        {
            _missedMassSums[kIndex] += missed[kIndex];
            _argmaxHitSums[kIndex] += missed[KValues.Length + kIndex];
        }
        _stateCount += stateCount;
    }

    /// <summary>
    /// Mean missed mass and mean argmax recall per K, then resets. Null when nothing was seen.
    /// </summary>
    public static LeafCoverageSummary? Drain()
    {
        if (_stateCount == 0)
            return null;

        float[] missedMass = new float[KValues.Length];
        float[] argmaxRecall = new float[KValues.Length];
        for (int kIndex = 0; kIndex < KValues.Length; ++kIndex)
        {
            missedMass[kIndex] = (float)(_missedMassSums[kIndex] / _stateCount);
            argmaxRecall[kIndex] = (float)(_argmaxHitSums[kIndex] / _stateCount);
        }

        LeafCoverageSummary summary = new(missedMass, argmaxRecall, _stateCount);
        Clear();
        return summary;
    }
}

public readonly record struct LeafCoverageSummary(
    float[] MissedMassByK,
    float[] ArgmaxRecallByK,
    long StateCount);
