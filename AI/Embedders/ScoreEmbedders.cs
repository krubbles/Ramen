namespace Ramen.AI;

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public sealed class BilinearOneHotScoreEmbedder : Module<Tensor, Tensor>
{
    public const int BucketCount = 30;
    public const float BucketWidth = 10f;

    public BilinearOneHotScoreEmbedder() : base(nameof(BilinearOneHotScoreEmbedder))
    {
        RegisterComponents();
    }


    public override Tensor forward(Tensor score)
    {
        using var scope = NewDisposeScope();

        Tensor bucketPosition = (score.to_type(ScalarType.Float32) / BucketWidth).clamp(0f, BucketCount - 1);
        Tensor lowerIndex = bucketPosition.floor().to_type(ScalarType.Int64);
        Tensor upperIndex = lowerIndex.add(1).clamp_max(BucketCount - 1);
        Tensor upperWeight = bucketPosition - lowerIndex.to_type(ScalarType.Float32);
        Tensor lowerWeight = 1f - upperWeight;

        Tensor lowerOneHot = functional.one_hot(lowerIndex, BucketCount).to_type(ScalarType.Float32);
        Tensor upperOneHot = functional.one_hot(upperIndex, BucketCount).to_type(ScalarType.Float32);
        Tensor result = lowerOneHot * lowerWeight.unsqueeze(-1) + upperOneHot * upperWeight.unsqueeze(-1);

        result.MoveToOuterDisposeScope();
        return result;
    }
}

public sealed class BilinearRangeScoreEmbedder : Module<Tensor, Tensor>
{
    readonly float _minValue;
    readonly float _maxValue;
    readonly int _bucketCount;

    public int BucketCount => _bucketCount;

    public BilinearRangeScoreEmbedder(float minValue, float maxValue, int bucketCount) : base(nameof(BilinearRangeScoreEmbedder))
    {
        if (bucketCount < 2)
            throw new ArgumentOutOfRangeException(nameof(bucketCount), "Bucket count must be at least 2.");
        if (maxValue <= minValue)
            throw new ArgumentOutOfRangeException(nameof(maxValue), "Max value must be greater than min value.");

        _minValue = minValue;
        _maxValue = maxValue;
        _bucketCount = bucketCount;

        RegisterComponents();
    }


    public override Tensor forward(Tensor score)
    {
        using var scope = NewDisposeScope();

        Tensor scoreFloat = score.to_type(ScalarType.Float32);
        Tensor normalized = ((scoreFloat - _minValue) / (_maxValue - _minValue)).clamp(0f, 1f);
        Tensor bucketPosition = normalized * (_bucketCount - 1);
        Tensor lowerIndex = bucketPosition.floor().to_type(ScalarType.Int64);
        Tensor upperIndex = lowerIndex.add(1).clamp_max(_bucketCount - 1);
        Tensor upperWeight = bucketPosition - lowerIndex.to_type(ScalarType.Float32);
        Tensor lowerWeight = 1f - upperWeight;

        Tensor lowerOneHot = functional.one_hot(lowerIndex, _bucketCount).to_type(ScalarType.Float32);
        Tensor upperOneHot = functional.one_hot(upperIndex, _bucketCount).to_type(ScalarType.Float32);
        Tensor result = lowerOneHot * lowerWeight.unsqueeze(-1) + upperOneHot * upperWeight.unsqueeze(-1);

        result.MoveToOuterDisposeScope();
        return result;
    }
}

public sealed class BilinearBucketScoreEmbedder : Module<Tensor, Tensor>
{
    readonly float _minValue;
    readonly float _maxValue;
    readonly int _bucketCount;
    readonly Tensor _bucketIndices;

    public int BucketCount => _bucketCount;
    public int EmbeddingWidth => _bucketCount + 1;

    public BilinearBucketScoreEmbedder(float minValue, float maxValue, int bucketCount) : base(nameof(BilinearBucketScoreEmbedder))
    {
        if (bucketCount < 1)
            throw new ArgumentOutOfRangeException(nameof(bucketCount), "Bucket count must be at least 1.");
        if (maxValue <= minValue)
            throw new ArgumentOutOfRangeException(nameof(maxValue), "Max value must be greater than min value.");

        _minValue = minValue;
        _maxValue = maxValue;
        _bucketCount = bucketCount;
        _bucketIndices = arange(_bucketCount + 1, dtype: ScalarType.Float32, device: CPU);

        RegisterComponents();
    }


    public override Tensor forward(Tensor score)
    {
        using var scope = NewDisposeScope();

        Tensor scaledScore = ((score.to_type(ScalarType.Float32) - _minValue) * _bucketCount / (_maxValue - _minValue))
            .clamp(0f, _bucketCount);
        Tensor bucketIndices = _bucketIndices.to(scaledScore.device);
        Tensor distanceFromBuckets = scaledScore.unsqueeze(-1) - bucketIndices;
        Tensor result = (1f - distanceFromBuckets.abs()).clamp_min(0f);

        result.MoveToOuterDisposeScope();
        return result;
    }
}

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
