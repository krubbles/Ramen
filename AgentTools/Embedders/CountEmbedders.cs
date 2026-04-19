namespace Ramen.AgentTools;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class OneHotHandsAndDiscardsEmbedder
{
    public const int EmbeddingWidth = 25;

    public static Tensor Embed(Tensor handsAndDiscards)
    {
        using var scope = NewDisposeScope();

        Tensor embedded = functional.one_hot(handsAndDiscards.to_type(ScalarType.Int64), EmbeddingWidth).to_type(ScalarType.Float32);
        embedded.MoveToOuterDisposeScope();
        return embedded;
    }
}
