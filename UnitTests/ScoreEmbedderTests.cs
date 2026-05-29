namespace Ramen.UnitTests;

using System.Reflection;
using Ramen.AgentTools;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class ScoreEmbedderTests
{
    [Test]
    public void BilinearOneHotScoreEmbedderInterpolatesAcrossTenPointBuckets()
    {
        using var scope = NewDisposeScope();

        BilinearOneHotScoreEmbedder embedder = new();
        Tensor scores = tensor(new float[,] { { 0f }, { 5f }, { 10f }, { 15f }, { 290f }, { 295f }, { 300f } });
        Tensor output = embedder.forward(scores).to(CPU);

        Assert.That(output.shape[0], Is.EqualTo(7));
        Assert.That(output.shape[1], Is.EqualTo(1));
        Assert.That(output.shape[2], Is.EqualTo(BilinearOneHotScoreEmbedder.BucketCount));

        float[] actual = output.data<float>().ToArray();
        AssertBucketWeights(actual, rowIndex: 0, expectedLowerIndex: 0, expectedLowerWeight: 1f, expectedUpperIndex: 0, expectedUpperWeight: 0f);
        AssertBucketWeights(actual, rowIndex: 1, expectedLowerIndex: 0, expectedLowerWeight: 0.5f, expectedUpperIndex: 1, expectedUpperWeight: 0.5f);
        AssertBucketWeights(actual, rowIndex: 2, expectedLowerIndex: 1, expectedLowerWeight: 1f, expectedUpperIndex: 1, expectedUpperWeight: 0f);
        AssertBucketWeights(actual, rowIndex: 3, expectedLowerIndex: 1, expectedLowerWeight: 0.5f, expectedUpperIndex: 2, expectedUpperWeight: 0.5f);
        AssertBucketWeights(actual, rowIndex: 4, expectedLowerIndex: 29, expectedLowerWeight: 1f, expectedUpperIndex: 29, expectedUpperWeight: 0f);
        AssertBucketWeights(actual, rowIndex: 5, expectedLowerIndex: 29, expectedLowerWeight: 1f, expectedUpperIndex: 29, expectedUpperWeight: 0f);
        AssertBucketWeights(actual, rowIndex: 6, expectedLowerIndex: 29, expectedLowerWeight: 1f, expectedUpperIndex: 29, expectedUpperWeight: 0f);
    }


    [Test]
    public void BilinearRangeScoreEmbedderInterpolatesAcrossConfiguredRange()
    {
        using var scope = NewDisposeScope();

        BilinearRangeScoreEmbedder embedder = new(minValue: 0f, maxValue: 1f, bucketCount: 4);
        Tensor scores = tensor(new float[,] { { -1f }, { 0f }, { 0.25f }, { 0.5f }, { 0.75f }, { 1f }, { 2f } });
        Tensor output = embedder.forward(scores).to(CPU);

        Assert.That(output.shape[0], Is.EqualTo(7));
        Assert.That(output.shape[1], Is.EqualTo(1));
        Assert.That(output.shape[2], Is.EqualTo(4));

        float[] actual = output.data<float>().ToArray();
        AssertBucketWeights(actual, rowIndex: 0, bucketCount: 4, expectedLowerIndex: 0, expectedLowerWeight: 1f, expectedUpperIndex: 0, expectedUpperWeight: 0f);
        AssertBucketWeights(actual, rowIndex: 1, bucketCount: 4, expectedLowerIndex: 0, expectedLowerWeight: 1f, expectedUpperIndex: 0, expectedUpperWeight: 0f);
        AssertBucketWeights(actual, rowIndex: 2, bucketCount: 4, expectedLowerIndex: 0, expectedLowerWeight: 0.25f, expectedUpperIndex: 1, expectedUpperWeight: 0.75f);
        AssertBucketWeights(actual, rowIndex: 3, bucketCount: 4, expectedLowerIndex: 1, expectedLowerWeight: 0.5f, expectedUpperIndex: 2, expectedUpperWeight: 0.5f);
        AssertBucketWeights(actual, rowIndex: 4, bucketCount: 4, expectedLowerIndex: 2, expectedLowerWeight: 0.75f, expectedUpperIndex: 3, expectedUpperWeight: 0.25f);
        AssertBucketWeights(actual, rowIndex: 5, bucketCount: 4, expectedLowerIndex: 3, expectedLowerWeight: 1f, expectedUpperIndex: 3, expectedUpperWeight: 0f);
        AssertBucketWeights(actual, rowIndex: 6, bucketCount: 4, expectedLowerIndex: 3, expectedLowerWeight: 1f, expectedUpperIndex: 3, expectedUpperWeight: 0f);
    }


    [Test]
    public void ThresholdScoreEmbeddingInterpolatesAndUsesOverflowEmbedding()
    {
        using var scope = NewDisposeScope();

        ThresholdScoreEmbedding embedder = new(threshold: 100f, bucketCount: 3, embeddingWidth: 2);
        SetEmbeddingWeights(embedder);

        Tensor scores = tensor(new float[,] { { 0f }, { 25f }, { 50f }, { 75f }, { 100f }, { 125f } });
        Tensor output = embedder.forward(scores).to(CPU);

        float[] actual = output.data<float>().ToArray();
        float[] expected =
        [
            0f, 0f,
            0.5f, 1f,
            1f, 2f,
            1.5f, 3f,
            10f, 20f,
            10f, 20f
        ];

        Assert.That(output.shape[0], Is.EqualTo(6));
        Assert.That(output.shape[1], Is.EqualTo(1));
        Assert.That(output.shape[2], Is.EqualTo(2));

        for (int valueIndex = 0; valueIndex < expected.Length; ++valueIndex)
            Assert.That(actual[valueIndex], Is.EqualTo(expected[valueIndex]).Within(1e-5f), $"Mismatch at flattened index {valueIndex}.");
    }


    static void SetEmbeddingWeights(ThresholdScoreEmbedding embedder)
    {
        BindingFlags bindingFlags = BindingFlags.Instance | BindingFlags.NonPublic;
        FieldInfo bucketField = typeof(ThresholdScoreEmbedding).GetField("_bucketEmbeddings", bindingFlags);
        FieldInfo overflowField = typeof(ThresholdScoreEmbedding).GetField("_overflowEmbedding", bindingFlags);

        Assert.That(bucketField, Is.Not.Null);
        Assert.That(overflowField, Is.Not.Null);

        Embedding bucketEmbeddings = bucketField.GetValue(embedder) as Embedding;
        Parameter overflowEmbedding = overflowField.GetValue(embedder) as Parameter;

        Assert.That(bucketEmbeddings, Is.Not.Null);
        Assert.That(overflowEmbedding, Is.Not.Null);

        using var noGrad = no_grad();
        bucketEmbeddings.weight.copy_(tensor(new float[,]
        {
            { 0f, 0f },
            { 1f, 2f },
            { 2f, 4f }
        }));
        overflowEmbedding.copy_(tensor([10f, 20f]));
    }


    static void AssertBucketWeights(float[] actual, int rowIndex, int expectedLowerIndex, float expectedLowerWeight, int expectedUpperIndex, float expectedUpperWeight)
    {
        AssertBucketWeights(
            actual: actual,
            rowIndex: rowIndex,
            bucketCount: BilinearOneHotScoreEmbedder.BucketCount,
            expectedLowerIndex: expectedLowerIndex,
            expectedLowerWeight: expectedLowerWeight,
            expectedUpperIndex: expectedUpperIndex,
            expectedUpperWeight: expectedUpperWeight);
    }


    static void AssertBucketWeights(float[] actual, int rowIndex, int bucketCount, int expectedLowerIndex, float expectedLowerWeight, int expectedUpperIndex, float expectedUpperWeight)
    {
        int rowOffset = rowIndex * bucketCount;
        for (int bucketIndex = 0; bucketIndex < bucketCount; ++bucketIndex)
        {
            float expectedWeight = 0f;
            if (bucketIndex == expectedLowerIndex)
                expectedWeight += expectedLowerWeight;
            if (bucketIndex == expectedUpperIndex)
                expectedWeight += expectedUpperWeight;

            Assert.That(
                actual[rowOffset + bucketIndex],
                Is.EqualTo(expectedWeight).Within(1e-5f),
                $"Mismatch for row {rowIndex}, bucket {bucketIndex}.");
        }
    }
}
