namespace Ramen.UnitTests;

using Ramen.AI;
using static TorchSharp.torch;

public class CountEmbedderTests
{
    [Test]
    public void OneHotCountEmbedderProducesExpectedOneHotEncoding()
    {
        using var scope = NewDisposeScope();

        Tensor counts = tensor(new long[] { 0, 1, 7, 24 });
        Tensor output = OneHotCountEmbedder.Embed(counts, embeddingWidth: 25).to(CPU);

        Assert.That(output.shape[0], Is.EqualTo(4));
        Assert.That(output.shape[1], Is.EqualTo(25));

        float[] actual = output.data<float>().ToArray();
        AssertOneHot(actual, rowIndex: 0, expectedHotIndex: 0);
        AssertOneHot(actual, rowIndex: 1, expectedHotIndex: 1);
        AssertOneHot(actual, rowIndex: 2, expectedHotIndex: 7);
        AssertOneHot(actual, rowIndex: 3, expectedHotIndex: 24);
    }


    static void AssertOneHot(float[] actual, int rowIndex, int expectedHotIndex)
    {
        int rowOffset = rowIndex * 25;
        for (int bucketIndex = 0; bucketIndex < 25; ++bucketIndex)
        {
            float expectedValue = bucketIndex == expectedHotIndex ? 1f : 0f;
            Assert.That(
                actual[rowOffset + bucketIndex],
                Is.EqualTo(expectedValue).Within(1e-5f),
                $"Mismatch for row {rowIndex}, bucket {bucketIndex}.");
        }
    }
}
