namespace Ramen.AI;

using static TorchSharp.torch.nn;

public static class OneHotCountEmbedder
{
    public static Tensor Embed(Tensor counts, int embeddingWidth)
    {
        using var scope = NewDisposeScope();

        Tensor embedded = functional.one_hot(counts.to_type(ScalarType.Int64), embeddingWidth).to_type(ScalarType.Float32);
        embedded.MoveToOuterDisposeScope();
        return embedded;
    }
}
