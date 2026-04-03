namespace Ramen.AgentTools;

using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class MeanPooledCardSetEmbedding : Module<Tensor, Tensor>
{
    readonly TorchSharp.Modules.Embedding _cardEmbedding;

    public MeanPooledCardSetEmbedding(int embeddingSize, Device device = null) : base(nameof(MeanPooledCardSetEmbedding))
    {
        Device targetDevice = device ?? CPU;
        _cardEmbedding = Embedding(Card.RankCount * Card.SuitCount + 1, embeddingSize, device: targetDevice);

        // Keep the null-card embedding at zero so padded cards contribute nothing by default.
        _cardEmbedding.weight[0].fill_(0f);
        RegisterComponents();
    }

    public override Tensor forward(Tensor cardSet)
    {
        using var scope = NewDisposeScope();

        // Embed each card index, including zero as the null-card slot.
        Tensor embeddedCards = _cardEmbedding.forward(cardSet.to_type(ScalarType.Int64));
        Tensor pooledEmbedding = embeddedCards.mean([embeddedCards.Dimensions - 2]);

        pooledEmbedding.MoveToOuterDisposeScope();
        return pooledEmbedding;
    }
}
