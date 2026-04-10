namespace Ramen.AgentTools;

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public sealed class ThresholdScoreEmbedding : Module<Tensor, Tensor>
{
    readonly Embedding _bucketEmbeddings;
    readonly Parameter _overflowEmbedding;
    readonly float _threshold;
    readonly int _bucketCount;

    public ThresholdScoreEmbedding(float threshold, int bucketCount, int embeddingWidth, Device device = null) : base(nameof(ThresholdScoreEmbedding))
    {
        Device targetDevice = device ?? CPU;
        _threshold = threshold;
        _bucketCount = bucketCount;
        _bucketEmbeddings = Embedding(bucketCount, embeddingWidth, device: targetDevice);
        _overflowEmbedding = Parameter(randn([embeddingWidth], device: targetDevice));

        RegisterComponents();
    }


    public override Tensor forward(Tensor score)
    {
        using var scope = NewDisposeScope();

        Tensor relativeScore = (score.to_type(ScalarType.Float32) / _threshold).clamp(0f, 1f);
        Tensor overflowMask = relativeScore.greater_equal(1f);

        Tensor bucketPosition = relativeScore * (_bucketCount - 1);
        Tensor lowerIndex = bucketPosition.floor().to_type(ScalarType.Int64);
        Tensor upperIndex = lowerIndex.add(1).clamp_max(_bucketCount - 1);
        Tensor upperWeight = (bucketPosition - lowerIndex.to_type(ScalarType.Float32)).unsqueeze(-1);
        Tensor lowerWeight = 1f - upperWeight;

        Tensor lowerEmbedding = _bucketEmbeddings.forward(lowerIndex);
        Tensor upperEmbedding = _bucketEmbeddings.forward(upperIndex);
        Tensor interpolatedEmbedding = lowerEmbedding * lowerWeight + upperEmbedding * upperWeight;

        long[] overflowShape = new long[score.shape.Length + 1];
        for (int dimIndex = 0; dimIndex < score.shape.Length; ++dimIndex)
            overflowShape[dimIndex] = score.shape[dimIndex];
        overflowShape[^1] = _overflowEmbedding.shape[0];

        Tensor overflowEmbedding = _overflowEmbedding.expand(overflowShape);
        Tensor result = where(overflowMask.unsqueeze(-1), overflowEmbedding, interpolatedEmbedding);
        result.MoveToOuterDisposeScope();
        return result;
    }
}
