namespace Ramen.UnitTests;

using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public class CardSetProcessorsTests
{
    [Test]
    public void StandardProcessorProducesExpectedFeatures()
    {
        using var scope = NewDisposeScope();

        long[,] cardSets =
        {
            {
                new Card(rank: 2, suit: Suit.Diamond).ToIndex(),
                new Card(rank: 2, suit: Suit.Club).ToIndex(),
                new Card(rank: 2, suit: Suit.Heart).ToIndex(),
                new Card(rank: 13, suit: Suit.Spade).ToIndex(),
                Card.Null.ToIndex()
            }
        };

        StandardProcessor processor = new();
        Tensor output = processor.forward(tensor(cardSets, dtype: ScalarType.Int64)).to(CPU);

        Assert.That(output.shape[0], Is.EqualTo(1));
        Assert.That(output.shape[1], Is.EqualTo(StandardProcessor.OutputWidth));

        float[] features = output.data<float>().ToArray();

        Assert.That(features[0], Is.EqualTo(1f));
        Assert.That(features[new Card(rank: 2, suit: Suit.Diamond).ToIndex()], Is.EqualTo(1f));
        Assert.That(features[new Card(rank: 2, suit: Suit.Club).ToIndex()], Is.EqualTo(1f));
        Assert.That(features[new Card(rank: 2, suit: Suit.Heart).ToIndex()], Is.EqualTo(1f));
        Assert.That(features[new Card(rank: 13, suit: Suit.Spade).ToIndex()], Is.EqualTo(1f));

        int thresholdBase = StandardProcessor.ExactCardCountWidth;

        Assert.That(features[thresholdBase + 0], Is.EqualTo(1f));
        Assert.That(features[thresholdBase + 11], Is.EqualTo(1f));
        Assert.That(features[thresholdBase + Card.RankCount + 0], Is.EqualTo(1f));
        Assert.That(features[thresholdBase + Card.RankCount + 1], Is.EqualTo(1f));
        Assert.That(features[thresholdBase + Card.RankCount + 2], Is.EqualTo(1f));
        Assert.That(features[thresholdBase + Card.RankCount + 3], Is.EqualTo(1f));

        int thresholdTwoBase = thresholdBase + Card.RankCount + Card.SuitCount;
        Assert.That(features[thresholdTwoBase + 0], Is.EqualTo(1f));
        Assert.That(features[thresholdTwoBase + Card.RankCount + 0], Is.EqualTo(0f));
        Assert.That(features[thresholdTwoBase + Card.RankCount + 1], Is.EqualTo(0f));
        Assert.That(features[thresholdTwoBase + Card.RankCount + 2], Is.EqualTo(0f));
        Assert.That(features[thresholdTwoBase + Card.RankCount + 3], Is.EqualTo(0f));

        int thresholdThreeBase = thresholdTwoBase + Card.RankCount + Card.SuitCount;
        Assert.That(features[thresholdThreeBase + 0], Is.EqualTo(1f));

        int thresholdFourBase = thresholdThreeBase + Card.RankCount + Card.SuitCount;
        Assert.That(features[thresholdFourBase + 0], Is.EqualTo(0f));
    }
}
