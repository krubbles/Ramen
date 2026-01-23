namespace Ramen.UnitTests;

using System.Linq;
using NUnit.Framework;
using Ramen.Game;
using Ramen.AI;
using static TorchSharp.torch;

public class CombinationMatricesTests
{
    [Test]
    public void GetCombinationMatrices_ShapeAndContent_MatchesCombinatorics()
    {
        int setSize = 5;
        int subsetSize = 3;

        int[][] combos = Combinatorics.GetCombinations(setSize, subsetSize);
        Tensor matrix = CombinationMatrices.GetCombinationMatrices(setSize, subsetSize);

        Assert.That(matrix.shape[0], Is.EqualTo(combos.Length), "Row count should equal number of combinations");
        Assert.That(matrix.shape[1], Is.EqualTo(setSize), "Column count should equal setSize");

        float[] data = matrix.data<float>().ToArray();
        int rows = combos.Length;
        int cols = setSize;

        for (int r = 0; r < rows; r++)
        {
            // Check row sum equals subset size
            int sum = 0;
            for (int c = 0; c < cols; c++)
            {
                float val = data[r * cols + c];
                Assert.That(val == 0.0f || val == 1.0f, Is.True, "Matrix entries must be 0 or 1");
                sum += (int)val;
            }

            Assert.That(sum, Is.EqualTo(subsetSize), $"Row {r} should have {subsetSize} ones");

            // Check ones indices match combination indices
            int[] ones = Enumerable.Range(0, cols).Where(c => data[r * cols + c] == 1.0f).ToArray();
            Assert.That(ones, Is.EqualTo(combos[r]), $"Row {r} one indices do not match combination");
        }
    }

    [Test]
    public void GetCombinationMatrices_RangeConcatenation_Works()
    {
        int setSize = 4;
        int min = 1;
        int max = 2;

        Tensor range = CombinationMatrices.GetCombinationMatrices(setSize, min, max);

        // Build expected by concatenating individual subset matrices
        Tensor part1 = CombinationMatrices.GetCombinationMatrices(setSize, 1);
        Tensor part2 = CombinationMatrices.GetCombinationMatrices(setSize, 2);
        Tensor expected = cat([part1, part2], 0);

        using (Assert.EnterMultipleScope())
        {
            Assert.That(range.shape[0], Is.EqualTo(expected.shape[0]));
            Assert.That(range.shape[1], Is.EqualTo(expected.shape[1]));
        }

        float[] actualData = range.data<float>().ToArray();
        float[] expectedData = expected.data<float>().ToArray();

        Assert.That(actualData, Is.EqualTo(expectedData));
    }

    [Test]
    public void GetCombinationMatrices_Caching_ReturnsSameObject()
    {
        int setSize = 6;
        int subsetSize = 2;

        Tensor a = CombinationMatrices.GetCombinationMatrices(setSize, subsetSize);
        Tensor b = CombinationMatrices.GetCombinationMatrices(setSize, subsetSize);

        // Implementation caches the tensor instance; expect reference equality
        Assert.That(object.ReferenceEquals(a, b), Is.True);
    }

    [Test]
    public void GetCombinationMatrices_ZeroAndFullSize_Works()
    {
        int setSize = 4;

        Tensor zero = CombinationMatrices.GetCombinationMatrices(setSize, 0);
        using (Assert.EnterMultipleScope())
        {
            Assert.That(zero.shape[0], Is.EqualTo(1));
            Assert.That(zero.shape[1], Is.EqualTo(setSize));
        }
        float[] zdata = zero.data<float>().ToArray();
        Assert.That(zdata.All(v => v == 0.0f));

        Tensor full = CombinationMatrices.GetCombinationMatrices(setSize, setSize);
        using (Assert.EnterMultipleScope())
        {
            Assert.That(full.shape[0], Is.EqualTo(1));
            Assert.That(full.shape[1], Is.EqualTo(setSize));
        }
        float[] fdata = full.data<float>().ToArray();
        Assert.That(fdata.All(v => v == 1.0f));
    }
}