namespace Ramen.AgentTools;

using Ramen.Game;

public enum AnnotationDataType : ushort
{
    Policy = 1,
    ExpectedReward = 3,
    MoveRewards = 4,
}

public static class AnnotationDataUtils
{
    public static AnnotatingDataMove CreatePolicyAnnotation(ReadOnlySpan<float> policy)
    {
        ushort[] encodedProbs = new ushort[policy.Length];
        for (int i = 0; i < policy.Length; i++)
            encodedProbs[i] = AnnotatingDataMove.EncodeProb(policy[i]);

        return AnnotatingDataMove.FromArray((ushort)AnnotationDataType.Policy, encodedProbs);
    }

    public static float[] DecodePolicyAnnotation(AnnotatingDataMove annotation)
    {
        if (annotation.DataTypeID != (ushort)AnnotationDataType.Policy)
            throw new InvalidOperationException($"Annotation type {annotation.DataTypeID} does not contain {nameof(AnnotationDataType.Policy)} data.");

        ushort[] encodedProbs = annotation.ToArray<ushort>();
        float[] policy = new float[encodedProbs.Length];
        for (int i = 0; i < encodedProbs.Length; i++)
            policy[i] = AnnotatingDataMove.DecodeProb(encodedProbs[i]);

        return policy;
    }

    public static bool TryDecodePolicyAnnotation(AnnotatingDataMove annotation, out float[] policy)
    {
        if (annotation == null || annotation.DataTypeID != (ushort)AnnotationDataType.Policy)
        {
            policy = [];
            return false;
        }

        policy = DecodePolicyAnnotation(annotation);
        return true;
    }

    public static AnnotatingDataMove CreateExpectedRewardAnnotation(float expectedReward)
    {
        byte[] data = BitConverter.GetBytes(expectedReward);
        return new((ushort)AnnotationDataType.ExpectedReward, data);
    }

    public static bool TryDecodeExpectedRewardAnnotation(AnnotatingDataMove annotation, out float expectedReward)
    {
        if (annotation == null || annotation.DataTypeID != (ushort)AnnotationDataType.ExpectedReward)
        {
            expectedReward = 0f;
            return false;
        }

        expectedReward = BitConverter.ToSingle(annotation.Data, 0);
        return true;
    }

    public static AnnotatingDataMove CreateMoveRewardsAnnotation(ReadOnlySpan<float> moveRewards)
    {
        if (moveRewards.Length == 0)
            return new((ushort)AnnotationDataType.MoveRewards, []);

        float minReward = moveRewards[0];
        float maxReward = moveRewards[0];
        for (int i = 1; i < moveRewards.Length; ++i)
        {
            minReward = MathF.Min(minReward, moveRewards[i]);
            maxReward = MathF.Max(maxReward, moveRewards[i]);
        }

        return CreateMoveRewardsAnnotation(moveRewards, minReward, maxReward);
    }

    public static AnnotatingDataMove CreateMoveRewardsAnnotation(ReadOnlySpan<float> moveRewards, float minReward, float maxReward)
    {
        byte[] data = new byte[sizeof(float) * 2 + sizeof(ushort) * moveRewards.Length];
        BitConverter.GetBytes(minReward).CopyTo(data, 0);
        BitConverter.GetBytes(maxReward).CopyTo(data, sizeof(float));

        for (int i = 0; i < moveRewards.Length; ++i)
        {
            ushort encodedReward = EncodeLinearInterp(minReward, maxReward, moveRewards[i]);
            BitConverter.GetBytes(encodedReward).CopyTo(data, sizeof(float) * 2 + i * sizeof(ushort));
        }

        return new((ushort)AnnotationDataType.MoveRewards, data);
    }

    public static float[] DecodeMoveRewardsAnnotation(AnnotatingDataMove annotation)
    {
        if (annotation.DataTypeID != (ushort)AnnotationDataType.MoveRewards)
            throw new InvalidOperationException($"Annotation type {annotation.DataTypeID} does not contain {nameof(AnnotationDataType.MoveRewards)} data.");

        if (annotation.Data.Length == 0)
            return [];

        float minReward = BitConverter.ToSingle(annotation.Data, 0);
        float maxReward = BitConverter.ToSingle(annotation.Data, sizeof(float));
        int rewardCount = (annotation.Data.Length - sizeof(float) * 2) / sizeof(ushort);
        float[] decodedRewards = new float[rewardCount];
        for (int i = 0; i < rewardCount; ++i)
        {
            ushort encodedReward = BitConverter.ToUInt16(annotation.Data, sizeof(float) * 2 + i * sizeof(ushort));
            decodedRewards[i] = DecodeLinearInterp(minReward, maxReward, encodedReward);
        }

        return decodedRewards;
    }

    public static bool TryDecodeMoveRewardsAnnotation(AnnotatingDataMove annotation, out float[] moveRewards)
    {
        if (annotation == null || annotation.DataTypeID != (ushort)AnnotationDataType.MoveRewards)
        {
            moveRewards = [];
            return false;
        }

        moveRewards = DecodeMoveRewardsAnnotation(annotation);
        return true;
    }

    static ushort EncodeLinearInterp(float minValue, float maxValue, float value)
    {
        if (maxValue <= minValue)
            return 0;

        float t = (value - minValue) / (maxValue - minValue);
        t = Math.Clamp(t, 0f, 1f);
        return (ushort)(t * ushort.MaxValue + 0.5f);
    }

    static float DecodeLinearInterp(float minValue, float maxValue, ushort encodedValue)
    {
        if (maxValue <= minValue)
            return minValue;

        float t = encodedValue / (float)ushort.MaxValue;
        return minValue + (maxValue - minValue) * t;
    }
}
