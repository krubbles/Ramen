namespace Ramen.Game;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

/// <summary>
/// Utility class for enumerating over A choose B combinations, intended for finding play/discard hand moves. 
/// </summary>
public static class Combinatorics
{
    static readonly Dictionary<(int A, int B), int[][]> _cache = [];

    /// <summary>
    /// Get all combinations of choosing <paramref name="subsetSize"/> items from a set with <paramref name="setSize"/> items.
    /// </summary>
    /// <returns>A 2D array of integers size (<paramref name="setSize"/> choose <paramref name="subsetSize"/>, <paramref name="subsetSize"/>). 
    /// Each row is a set of indices representing a unique subset of <paramref name="subsetSize"/>.</returns>    
    public static int[][] GetCombinations(int setSize, int subsetSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(setSize, nameof(setSize));
        ArgumentOutOfRangeException.ThrowIfNegative(subsetSize, nameof(subsetSize));

        (int A, int B) key = (setSize, subsetSize);

        lock (_cache)
        {
            if (_cache.TryGetValue(key, out int[][] cached))
            {
                return cached;
            }
        }

        int[][] combinations = GenerateCombinations(setSize, subsetSize);

        lock (_cache)
        {
            _cache.TryAdd(key, combinations);
            return combinations;
        }
    }

    static readonly Dictionary<(int SetSize, int MinSubsetSize, int MaxSubsetSize), int[][]> _rangeCache = [];

    public static int[][] GetCombinations(int setSize, int minSubsetSize, int maxSubsetSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(setSize, nameof(setSize));
        ArgumentOutOfRangeException.ThrowIfNegative(minSubsetSize, nameof(minSubsetSize));
        ArgumentOutOfRangeException.ThrowIfNegative(maxSubsetSize, nameof(maxSubsetSize));

        (int SetSize, int MinSubsetSize, int MaxSubsetSize) key = (setSize, minSubsetSize, maxSubsetSize);

        lock (_rangeCache)
        {
            if (_rangeCache.TryGetValue(key, out int[][] cached))
            {
                return cached;
            }
        }

        List<int[]> output = [];
        for (int subsetSize = minSubsetSize; subsetSize <= maxSubsetSize; subsetSize++)
        {
            int[][] combos = GetCombinations(setSize, subsetSize);
            output.AddRange(combos);
        }

        int[][] result = output.ToArray();

        lock (_rangeCache)
        {
            _rangeCache.TryAdd(key, result);
            return result;
        }
    }

    static int[][] GenerateCombinations(int setSize, int subsetSize)
    {
        int count = CalculateCombinationCount(setSize, subsetSize);
        int[][] output = new int[count][];
        int[] combo = new int[subsetSize];
        for (int i = 0; i < subsetSize; i++)
        {
            combo[i] = i;
        }

        int index = 0;

        while (true)
        {
            output[index] = new int[subsetSize];
            for (int i = 0; i < subsetSize; i++)
            {
                output[index][i] = combo[i];
            }

            index++;

            int k = subsetSize - 1;

            while (k >= 0 && combo[k] == setSize - subsetSize + k)
            {
                k--;
            }

            if (k < 0)
            {
                break;
            }

            combo[k]++;

            for (int i = k + 1; i < subsetSize; i++)
            {
                combo[i] = combo[i - 1] + 1;
            }
        }

        return output;
    }

    public static int CalculateCombinationCount(int setSize, int subsetSize)
    {
        if (subsetSize > setSize || subsetSize <= 0)
            return 0;

        int count = 1;

        for (int i = 1; i <= subsetSize; i++)
        {
            count = count * (setSize - (subsetSize - i)) / i;        
        }

        return count;
    }

    public static int CalculateCombinationCount(int setSize, int maxSubsetSize, int minSubsetSize)
    {
        int count = 0;

        for (int subsetSize = minSubsetSize; subsetSize <= maxSubsetSize; subsetSize++)
        {
            count += CalculateCombinationCount(setSize, subsetSize);
        }

        return count;
    }
}