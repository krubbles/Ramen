namespace Ramen.UnitTests;

using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public class GameStateEmbedderTests
{
    [Test]
    public void ToTensorsLeavesPlayHandScoresNullByDefault()
    {
        using var scope = NewDisposeScope();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();

        GameStateEmbedder embedder = new(1);
        embedder.AddGameState(gameState);

        GameStateTensors tensors = embedder.ToTensors();

        Assert.That(tensors.PlayHandScores, Is.Null);
    }


    [Test]
    public void ToTensorsEmbedsRawScoreAndThreshold()
    {
        using var scope = NewDisposeScope();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();
        SetCurrentRoundScoreMove setScoreMove = new(150f);
        setScoreMove.Apply(gameState);

        GameStateEmbedder embedder = new(1);
        embedder.AddGameState(gameState);

        GameStateTensors tensors = embedder.ToTensors();
        Tensor score = tensors.Score.to(CPU);
        Tensor scoreThreshold = tensors.ScoreThreshold.to(CPU);
        float[] actual = score.data<float>().ToArray();
        float[] thresholdActual = scoreThreshold.data<float>().ToArray();

        Assert.That(score.shape[0], Is.EqualTo(1));
        Assert.That(score.shape[1], Is.EqualTo(1));
        Assert.That(scoreThreshold.shape[0], Is.EqualTo(1));
        Assert.That(scoreThreshold.shape[1], Is.EqualTo(1));
        Assert.That(actual[0], Is.EqualTo(150f).Within(1e-5f));
        Assert.That(thresholdActual[0], Is.EqualTo((float)gameState.ScoringState.CurrentRoundThresholdScore).Within(1e-5f));
    }


    [Test]
    public void ToTensorsStoresRemainingHandsAndDiscardsAsSeparateTensors()
    {
        using var scope = NewDisposeScope();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();
        SetRemainingHandsAndDiscardsMove setCountsMove = new(remainingHands: 2, remainingDiscards: 1);
        setCountsMove.Apply(gameState);

        GameStateEmbedder embedder = new(1);
        embedder.AddGameState(gameState);

        GameStateTensors tensors = embedder.ToTensors();
        Tensor remainingHands = tensors.RemainingHands.to(CPU);
        Tensor remainingDiscards = tensors.RemainingDiscards.to(CPU);

        Assert.That(remainingHands.shape[0], Is.EqualTo(1));
        Assert.That(remainingDiscards.shape[0], Is.EqualTo(1));
        Assert.That(remainingHands.data<long>().ToArray()[0], Is.EqualTo(2));
        Assert.That(remainingDiscards.data<long>().ToArray()[0], Is.EqualTo(1));
    }


    [Test]
    public void ToTensorsCanIncludePlayHandScoresInStandardOrdering()
    {
        using var scope = NewDisposeScope();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();

        GameStateEmbedder embedder = new(1);
        embedder.AddGameState(gameState);

        GameStateTensors tensors = embedder.ToTensors(includePlayHandScores: true);
        Tensor playHandScores = tensors.PlayHandScores.to(CPU);
        float[] actual = playHandScores.data<float>().ToArray();

        Assert.That(playHandScores.shape[0], Is.EqualTo(1));
        Assert.That(playHandScores.shape[1], Is.EqualTo(GameStateEmbedder.PlayHandScoreCount));

        float[] expected = GetExpectedPlayHandScores(gameState);
        for (int handIndex = 0; handIndex < expected.Length; ++handIndex)
            Assert.That(actual[handIndex], Is.EqualTo(expected[handIndex]).Within(1e-5f), $"Mismatch at hand index {handIndex}.");
    }


    [Test]
    public void ToTensorsStoresOwnedStoreJokersPricesMoneyRoundAndStage()
    {
        using var scope = NewDisposeScope();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();
        gameState.JokerState.AddJoker(Joker.Jimbo);
        gameState.JokerState.AddJoker(Joker.Jimbo);
        gameState.JokerState.AddJoker(Joker.GreedyJoker);

        SetCurrentRoundScoreMove setScoreMove = new(300f);
        setScoreMove.Apply(gameState);
        int[] playedCardIndices = [0];
        UseHandMove useHandMove = new(false, playedCardIndices);
        useHandMove.Apply(gameState);
        new EndRoundMove().Apply(gameState);
        new EnterShopMove().Apply(gameState);

        gameState.ShopState.ShopOfferings.Clear();
        gameState.ShopState.ShopOfferings.Add(new(Joker.GreedyJoker));
        gameState.ShopState.ShopOfferings.Add(new(Joker.GreedyJoker));

        GameStateEmbedder embedder = new(1);
        embedder.AddGameState(gameState);

        GameStateTensors tensors = embedder.ToTensors();
        long[] ownedJokers = tensors.OwnedJokers.to(CPU).data<long>().ToArray();
        long[] storeJokers = tensors.StoreJokers.to(CPU).data<long>().ToArray();
        long[] storePrices = tensors.StorePrices.to(CPU).data<long>().ToArray();
        long[] rerollPrice = tensors.RerollPrice.to(CPU).data<long>().ToArray();
        long[] money = tensors.Money.to(CPU).data<long>().ToArray();
        long[] round = tensors.Round.to(CPU).data<long>().ToArray();
        long[] stage = tensors.Stage.to(CPU).data<long>().ToArray();

        int jimboIndex = Array.IndexOf(Joker.Page1Jokers, Joker.Jimbo) + 1;
        int greedyIndex = Array.IndexOf(Joker.Page1Jokers, Joker.GreedyJoker) + 1;

        Assert.That(ownedJokers.Length, Is.EqualTo(GameStateEmbedder.MaxOwnedJokerCount));
        Assert.That(storeJokers.Length, Is.EqualTo(GameStateEmbedder.MaxStoreJokerCount));
        Assert.That(ownedJokers[0], Is.EqualTo(jimboIndex));
        Assert.That(ownedJokers[1], Is.EqualTo(jimboIndex));
        Assert.That(ownedJokers[2], Is.EqualTo(greedyIndex));
        Assert.That(ownedJokers[3], Is.EqualTo(0));
        Assert.That(ownedJokers[4], Is.EqualTo(0));
        Assert.That(storeJokers[0], Is.EqualTo(greedyIndex));
        Assert.That(storeJokers[1], Is.EqualTo(greedyIndex));
        Assert.That(storePrices[0], Is.EqualTo(Joker.GreedyJoker.BasePrice));
        Assert.That(storePrices[1], Is.EqualTo(Joker.GreedyJoker.BasePrice));
        Assert.That(rerollPrice[0], Is.EqualTo(gameState.ShopState.CurrentRerollCost));
        Assert.That(money[0], Is.EqualTo(6));
        Assert.That(round[0], Is.EqualTo(1));
        Assert.That(stage[0], Is.EqualTo(1));
    }


    [Test]
    public void StorePolicyLogitsMaskNullStoreOffer()
    {
        using var scope = NewDisposeScope();
        TensorManager.Init();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();
        SetCurrentRoundScoreMove setScoreMove = new(300f);
        setScoreMove.Apply(gameState);
        UseHandMove useHandMove = new(false, [0]);
        useHandMove.Apply(gameState);
        new EndRoundMove().Apply(gameState);
        new EnterShopMove().Apply(gameState);
        gameState.ShopState.ShopOfferings.Clear();
        gameState.ShopState.ShopOfferings.Add(new(Joker.Jimbo));
        gameState.ShopState.ShopOfferings.Add(null);

        GameStateEmbedder embedder = new(1);
        embedder.AddGameState(gameState);
        GameStateTensors tensors = embedder.ToTensors(device: PpoPolicyValueModel.EvalDevice);

        PpoPolicyValueModel model = new();
        Tensor logits = model.GetStorePolicyLogits(tensors).to(CPU);
        float[] actual = logits.data<float>().ToArray();

        Assert.That(logits.shape[0], Is.EqualTo(1));
        Assert.That(logits.shape[1], Is.EqualTo(4));
        Assert.That(float.IsFinite(actual[0]), Is.True);
        Assert.That(float.IsFinite(actual[1]), Is.True);
        Assert.That(float.IsFinite(actual[2]), Is.True);
        Assert.That(actual[3], Is.LessThan(-1e8f));
    }


    static float[] GetExpectedPlayHandScores(GameState gameState)
    {
        int[][] playHandOptions = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        float[] playHandScores = new float[GameStateEmbedder.PlayHandScoreCount];
        float roundScoreBefore = (float)gameState.ScoringState.CurrentRoundTotalScore;
        int handCardCount = gameState.HandState.HandCardCount;

        for (int handIndex = 0; handIndex < playHandOptions.Length; ++handIndex)
        {
            int[] cardIndices = playHandOptions[handIndex];
            if (cardIndices[^1] >= handCardCount)
                continue;

            UseHandMove useHandMove = new(false, cardIndices);
            useHandMove.Apply(gameState);
            float roundScoreAfter = (float)gameState.ScoringState.CurrentRoundTotalScore;
            playHandScores[handIndex] = roundScoreAfter - roundScoreBefore;
            useHandMove.Revert(gameState);
        }

        return playHandScores;
    }
}
