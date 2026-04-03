namespace Ramen.UnitTests;

using System.Reflection;
using Ramen.AI;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public class PolicyModelTests
{
    [SetUp]
    public void SetUp()
    {
        TensorManager.Init();
    }

    [Test]
    public void GetRemainingDeckSuitInputExpandedMatchesManagedComputation()
    {
        using var scope = NewDisposeScope();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();
        Tensor remainingDeckTensor = BuildRemainingDeckTensor(gameState).to(PolicyModel.EvalDevice);
        PolicyModel model = new();

        int moveCount = 7;

        for (Suit suit = Suit.Diamond; suit <= Suit.Spade; ++suit)
        {
            Tensor actualTensor = InvokeGetRemainingDeckSuitInputExpanded(model, remainingDeckTensor, suit, moveCount).to(CPU);
            float[] actual = actualTensor.data<float>().ToArray();

            float[] expected = ComputeExpectedInput(model, gameState, suit, moveCount);

            Assert.That(actualTensor.shape[0], Is.EqualTo(1));
            Assert.That(actualTensor.shape[1], Is.EqualTo(moveCount));
            Assert.That(actualTensor.shape[2], Is.EqualTo(64));
            Assert.That(actual.Length, Is.EqualTo(expected.Length));

            for (int i = 0; i < actual.Length; ++i)
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f), $"Mismatch at flattened index {i} for suit {suit}.");
        }
    }

    static Tensor BuildRemainingDeckTensor(GameState gameState)
    {
        long[,] cards = new long[1, 52];
        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int i = 0; i < deck.Length; ++i)
            cards[0, i] = deck[i].ToIndex();
        return tensor(cards, ScalarType.Int64);
    }

    static float[] ComputeExpectedInput(PolicyModel model, GameState gameState, Suit suit, int moveCount)
    {
        using var scope = NewDisposeScope();

        // Get embedding vectors for each real remaining-deck card.
        TorchSharp.Modules.Embedding remainingDeckEmbedding = GetRemainingDeckEmbedding(model);
        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        float[] cardEmbeds = [];
        if (deck.Length > 0)
        {
            long[,] cardIndices = new long[1, deck.Length];
            for (int i = 0; i < deck.Length; ++i)
                cardIndices[0, i] = deck[i].Rank - 1 + ((int)deck[i].Suit - 1) * PolicyModel.RankCount;

            Tensor indexTensor = tensor(cardIndices, ScalarType.Int64, device: PolicyModel.EvalDevice);
            Tensor cardEmbedsTensor = remainingDeckEmbedding.forward(indexTensor).to(CPU); // 1 x deckCount x 32
            cardEmbeds = cardEmbedsTensor.data<float>().ToArray();
        }

        // Reproduce model math in managed loops.
        int deckCount = deck.Length;
        int embedWidth = 32;
        float[] fullDeckAverage = new float[embedWidth];
        float[] matchingSuitAverage = new float[embedWidth];
        for (int cardIndex = 0; cardIndex < deckCount; ++cardIndex)
        {
            bool isMatchingSuit = deck[cardIndex].Suit == suit;
            int embedOffset = cardIndex * embedWidth;
            for (int embedIndex = 0; embedIndex < embedWidth; ++embedIndex)
            {
                float value = cardEmbeds[embedOffset + embedIndex];
                fullDeckAverage[embedIndex] += value;
                if (isMatchingSuit)
                    matchingSuitAverage[embedIndex] += value;
            }
        }

        float denominator = Math.Max(deckCount, 1);
        for (int embedIndex = 0; embedIndex < embedWidth; ++embedIndex)
        {
            fullDeckAverage[embedIndex] /= denominator;
            matchingSuitAverage[embedIndex] /= denominator;
        }

        // Concatenate [fullDeckAverage, matchingSuitAverage] and expand across move count.
        float[] expectedSingle = new float[embedWidth * 2];
        Array.Copy(fullDeckAverage, 0, expectedSingle, 0, embedWidth);
        Array.Copy(matchingSuitAverage, 0, expectedSingle, embedWidth, embedWidth);

        float[] expectedExpanded = new float[moveCount * expectedSingle.Length];
        for (int moveIndex = 0; moveIndex < moveCount; ++moveIndex)
            Array.Copy(expectedSingle, 0, expectedExpanded, moveIndex * expectedSingle.Length, expectedSingle.Length);
        return expectedExpanded;
    }

    static TorchSharp.Modules.Embedding GetRemainingDeckEmbedding(PolicyModel model)
    {
        BindingFlags bindingFlags = BindingFlags.Instance | BindingFlags.NonPublic;
        FieldInfo fieldInfo = typeof(PolicyModel).GetField("_remainingDeckCardEmbedding", bindingFlags);
        Assert.That(fieldInfo, Is.Not.Null);
        TorchSharp.Modules.Embedding embedding = fieldInfo.GetValue(model) as TorchSharp.Modules.Embedding;
        Assert.That(embedding, Is.Not.Null);
        return embedding;
    }

    static Tensor InvokeGetRemainingDeckSuitInputExpanded(PolicyModel model, Tensor remainingDeckTensor, Suit suit, int moveCount)
    {
        BindingFlags bindingFlags = BindingFlags.Instance | BindingFlags.Public;
        MethodInfo methodInfo = typeof(PolicyModel).GetMethod("GetRemainingDeckSuitInputExpanded", bindingFlags);
        Assert.That(methodInfo, Is.Not.Null);
        object result = methodInfo.Invoke(model, [remainingDeckTensor, suit, moveCount]);
        Assert.That(result, Is.Not.Null);
        return result as Tensor;
    }
}
