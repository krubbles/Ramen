namespace Ramen.UnitTests;

using System.Reflection;
using Ramen.AgentTools;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class ScoreEmbedderTests
{
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
}
