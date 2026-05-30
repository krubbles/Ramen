namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class StandardProcessor : Module<Tensor, Tensor>
{
    public const int ExactCardCountWidth = Card.RankCount * Card.SuitCount + 1;
    public const int RankThresholdWidth = Card.RankCount * 4;
    public const int SuitThresholdWidth = Card.SuitCount * 4;
    public const int OutputWidth = ExactCardCountWidth + RankThresholdWidth + SuitThresholdWidth;

    public StandardProcessor() : base(nameof(StandardProcessor))
    {
    }


    public override Tensor forward(Tensor cardSet)
    {
        using var scope = NewDisposeScope();

        Tensor cardSetIndices = cardSet.to_type(ScalarType.Int64);
        int cardDim = (int)cardSetIndices.Dimensions - 1;

        Tensor exactCardCounts = functional.one_hot(cardSetIndices, ExactCardCountWidth)
            .to_type(ScalarType.Float32)
            .sum(dim: cardDim);

        Tensor validCards = cardSetIndices.gt(0).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor rankIndices = (cardSetIndices - 1).clamp_min(0).remainder(Card.RankCount).to_type(ScalarType.Int64);
        Tensor suitIndices = (cardSetIndices - 1).clamp_min(0).div(Card.RankCount).to_type(ScalarType.Int64);

        Tensor rankCounts = (functional.one_hot(rankIndices, Card.RankCount).to_type(ScalarType.Float32) * validCards)
            .sum(dim: cardDim);
        Tensor suitCounts = (functional.one_hot(suitIndices, Card.SuitCount).to_type(ScalarType.Float32) * validCards)
            .sum(dim: cardDim);

        Tensor[] thresholdedCounts = new Tensor[4 * 2];
        for (int threshold = 1; threshold <= 4; ++threshold)
        {
            thresholdedCounts[(threshold - 1) * 2] = rankCounts.greater_equal(threshold).to_type(ScalarType.Float32);
            thresholdedCounts[(threshold - 1) * 2 + 1] = suitCounts.greater_equal(threshold).to_type(ScalarType.Float32);
        }

        Tensor processed = cat([exactCardCounts, .. thresholdedCounts], dim: -1);
        processed.MoveToOuterDisposeScope();
        return processed;
    }
}

public class MeanPooledCardSetEmbedding : Module<Tensor, Tensor>
{
    readonly Embedding _cardEmbedding;

    public MeanPooledCardSetEmbedding(int embeddingSize, Device device = null) : base(nameof(MeanPooledCardSetEmbedding))
    {
        Device targetDevice = device ?? CPU;
        _cardEmbedding = Embedding(Card.RankCount * Card.SuitCount + 1, embeddingSize, device: targetDevice);

        // Keep the null-card embedding at zero so padded cards contribute nothing by default.
        using var noGrad = no_grad();
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
