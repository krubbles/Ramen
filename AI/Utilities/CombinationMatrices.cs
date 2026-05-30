namespace Ramen.AI;

// NOTE: CURRENTLY NOT IN USE.
// MAY END UP USING IN THE FUTURE HOWEVER

/// <summary>
/// Utilities for generating a choose b matrices. Use for generating use hand move data.
/// Each matrix has shape (nChoosek, setSize) and contains 0/1 entries where 1 indicates the item
/// at that column is included in the subset for that row.
/// </summary>
public static class CombinationMatrices
{
    static readonly Dictionary<(int SetSize, int SubsetSize), Tensor> _cache = new();

    static readonly Dictionary<(int SetSize, int MinSubsetSize, int MaxSubsetSize), Tensor> _rangeCache = new();

    /// <summary>
    /// Returns a tensor of shape (C(setSize, subsetSize), setSize) where each row is a 0/1 vector
    /// indicating which items are chosen for that combination.
    /// </summary>
    public static Tensor GetCombinationMatrices(int setSize, int subsetSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(setSize, nameof(setSize));
        ArgumentOutOfRangeException.ThrowIfNegative(subsetSize, nameof(subsetSize));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(subsetSize, setSize, nameof(subsetSize));

        (int SetSize, int SubsetSize) key = (setSize, subsetSize);

        lock (_cache)
        {
            if (_cache.TryGetValue(key, out Tensor cached))
            {
                return cached;
            }
        }

        int[][] combos = Combinatorics.GetCombinations(setSize, subsetSize);
        int rows = combos.Length;
        int cols = setSize;

        float[] data = new float[rows * cols];

        for (int r = 0; r < rows; r++)
        {
            int[] combo = combos[r];
            for (int i = 0; i < combo.Length; i++)
            {
                int idx = combo[i];
                data[r * cols + idx] = 1.0f;
            }
        }

        Tensor matrix = tensor(data, dtype: float32).view([rows, cols]);

        lock (_cache)
        {
            _cache.TryAdd(key, matrix);
            return matrix;
        }
    }

    /// <summary>
    /// Returns a tensor containing all combinations with subset sizes in the inclusive range [minSubsetSize, maxSubsetSize].
    /// Rows are concatenated in increasing subsetSize order.
    /// </summary>
    public static Tensor GetCombinationMatrices(int setSize, int minSubsetSize, int maxSubsetSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(setSize, nameof(setSize));
        ArgumentOutOfRangeException.ThrowIfNegative(minSubsetSize, nameof(minSubsetSize));
        ArgumentOutOfRangeException.ThrowIfNegative(maxSubsetSize, nameof(maxSubsetSize));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(minSubsetSize, maxSubsetSize, nameof(minSubsetSize));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(maxSubsetSize, setSize, nameof(maxSubsetSize));

        (int SetSize, int MinSubsetSize, int MaxSubsetSize) key = (setSize, minSubsetSize, maxSubsetSize);

        lock (_rangeCache)
        {
            if (_rangeCache.TryGetValue(key, out Tensor cached))
            {
                return cached;
            }
        }

        List<Tensor> parts = [];
        for (int subsetSize = minSubsetSize; subsetSize <= maxSubsetSize; subsetSize++)
        {
            parts.Add(GetCombinationMatrices(setSize, subsetSize));
        }

        Tensor result = parts.Count == 0
            ? empty([0, setSize], dtype: float32)
            : (parts.Count == 1 ? parts[0] : cat([.. parts], 0));

        lock (_rangeCache)
        {
            _rangeCache.TryAdd(key, result);
            return result;
        }
    }
}
