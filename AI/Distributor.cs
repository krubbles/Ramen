namespace Ramen.AI;

using System.Runtime.CompilerServices;
using Ramen.Game;

/// <summary>
/// Currently unused.
/// </summary>
public static class MeanDistributionAnalyzer
{
    public static int SampleFromDistribution(FastRandom random, float[] distribution)
    {
        // Get a random value between 0.0 and 1.0
        float r = random.NextPortion();
        float cumulativeSum = 0.0f;

        for (int i = 0; i < distribution.Length; i++)
        {
            cumulativeSum += distribution[i];

            // If the random number falls within this index's range
            if (r <= cumulativeSum)
            {
                return i;
            }
        }

        // Fallback: in case of minor floating point inaccuracies at the end of the array
        return distribution.Length - 1;
    }
    /// <summary>
    /// Returns the probability distribution using Clark's Approximation.
    /// Complexity: O(Width)
    /// </summary>
    public static float[] GetProbabilityDistribution(float[] means, float[] vars)
    {
        int _width = means.Length;
        float[] result = new float[_width];

        var prefixMaxM = new float[_width];
        var prefixMaxV = new float[_width];
        var suffixMaxM = new float[_width];
        var suffixMaxV = new float[_width];

        prefixMaxM[0] = means[0];
        prefixMaxV[0] = vars[0];
        for (int i = 1; i < _width; i++)
        {
            (prefixMaxM[i], prefixMaxV[i]) = ApproxMax(prefixMaxM[i - 1], prefixMaxV[i - 1], means[i], vars[i]);
        }

        suffixMaxM[_width - 1] = means[_width - 1];
        suffixMaxV[_width - 1] = vars[_width - 1];
        for (int i = _width - 2; i >= 0; i--)
        {
            (suffixMaxM[i], suffixMaxV[i]) = ApproxMax(suffixMaxM[i + 1], suffixMaxV[i + 1], means[i], vars[i]);
        }

        // 3. Compare each channel to the max of all other channels
        double totalProb = 0;
        for (int i = 0; i < _width; i++)
        {
            float otherM, otherV;
            if (i == 0) { otherM = suffixMaxM[1]; otherV = suffixMaxV[1]; }
            else if (i == _width - 1) { otherM = prefixMaxM[_width - 2]; otherV = prefixMaxV[_width - 2]; }
            else
            {
                (otherM, otherV) = ApproxMax(prefixMaxM[i - 1], prefixMaxV[i - 1], suffixMaxM[i + 1], suffixMaxV[i + 1]);
            }

            float diffM = means[i] - otherM;
            float combinedSd = (float)Math.Sqrt(vars[i] + otherV);
            float p = (float)Phi(diffM / combinedSd);

            result[i] = p;
            totalProb += p;
        }

        // 4. Final normalization pass
        float invSum = (float)(1.0 / totalProb);
        for (int i = 0; i < _width; i++) result[i] *= invSum;

        return result;
    }

    /// <summary>
    /// Clark's Approximation for the mean and variance of Z = max(X, Y)
    /// </summary>
    private static (float m, float v) ApproxMax(float m1, float v1, float m2, float v2)
    {
        float sDiff = (float)Math.Sqrt(v1 + v2);
        float alpha = (m1 - m2) / sDiff;

        double p = Phi(alpha);
        double pdf = GaussianPdf(alpha);

        double meanMax = m1 * p + m2 * (1.0 - p) + sDiff * pdf;
        double varMax = (v1 + (double)m1 * m1) * p +
                        (v2 + (double)m2 * m2) * (1.0 - p) +
                        ((double)m1 + m2) * sDiff * pdf -
                        meanMax * meanMax;

        return ((float)meanMax, (float)Math.Max(varMax, 1e-12));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double GaussianPdf(double x) => 0.3989422804014327 * Math.Exp(-0.5 * x * x);

    /// <summary>
    /// Abramowitz and Stegun approximation for Normal CDF. Accuracy ~7.5e-8.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Phi(double x)
    {
        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        int sign = (x < 0) ? -1 : 1;
        double absX = Math.Abs(x) / 1.4142135623730951; // x / sqrt(2)

        double t = 1.0 / (1.0 + p * absX);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-absX * absX);
        return 0.5 * (1.0 + sign * y);
    }
}
