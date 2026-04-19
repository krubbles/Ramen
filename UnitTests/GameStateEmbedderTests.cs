namespace Ramen.UnitTests;

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
    public void ToTensorsEmbedsScoreAsSingleRelativeValue()
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
        float[] actual = score.data<float>().ToArray();

        Assert.That(score.shape[0], Is.EqualTo(1));
        Assert.That(score.shape[1], Is.EqualTo(1));
        Assert.That(actual[0], Is.EqualTo(0.5f).Within(1e-5f));
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


    static float[] GetExpectedPlayHandScores(GameState gameState)
    {
        int[][] playHandOptions = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        float[] playHandScores = new float[GameStateEmbedder.PlayHandScoreCount];
        float roundScoreBefore = (float)gameState.ScoringState.CurrentRoundTotalChips;
        int handCardCount = gameState.HandState.HandCardCount;

        for (int handIndex = 0; handIndex < playHandOptions.Length; ++handIndex)
        {
            int[] cardIndices = playHandOptions[handIndex];
            if (cardIndices[^1] >= handCardCount)
                continue;

            UseHandMove useHandMove = new(false, cardIndices);
            useHandMove.Apply(gameState);
            float roundScoreAfter = (float)gameState.ScoringState.CurrentRoundTotalChips;
            playHandScores[handIndex] = (roundScoreAfter - roundScoreBefore) / 300f;
            useHandMove.Revert(gameState);
        }

        return playHandScores;
    }
}
