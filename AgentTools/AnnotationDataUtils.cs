namespace Ramen.AgentTools;

using Ramen.Game;

public enum AnnotationDataType : ushort
{
    Policy = 1,
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
}
