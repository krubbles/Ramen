namespace Ramen.UnitTests;

using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class ModularPolicyModelTests
{
    [Test]
    public void GetPolicyLogitsSelectedMovesMatchFullMoveOrdering()
    {
        using var scope = NewDisposeScope();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();

        GameStateEmbedder embedder = new(1);
        embedder.AddGameState(gameState);
        GameStateTensors stateTensors = embedder.ToTensors(includePlayHandScores: true);

        ModularPolicyModel.Settings settings = new()
        {
            FullHandCardSet = new()
            {
                Embedder = new StandardProcessor(),
                EmbeddingWidth = StandardProcessor.OutputWidth,
            },
            RemainingDeckCardSet = new()
            {
                Embedder = new StandardProcessor(),
                EmbeddingWidth = StandardProcessor.OutputWidth,
            },
            UsedHandCardSet = new()
            {
                Embedder = new StandardProcessor(),
                EmbeddingWidth = StandardProcessor.OutputWidth,
            },
            RemainingHandCardSet = new()
            {
                Embedder = new StandardProcessor(),
                EmbeddingWidth = StandardProcessor.OutputWidth,
            },
            PreMoveScore = new()
            {
                Embedder = new ThresholdScoreEmbedding(
                    threshold: 1f,
                    bucketCount: 4,
                    embeddingWidth: 8,
                    device: ModularPolicyModel.EvalDevice),
                EmbeddingWidth = 8,
            },
            PostMoveScore = new()
            {
                Embedder = new ThresholdScoreEmbedding(
                    threshold: 1f,
                    bucketCount: 4,
                    embeddingWidth: 8,
                    device: ModularPolicyModel.EvalDevice),
                EmbeddingWidth = 8,
            },
            RemainingCount = new()
            {
                Embedder = Embedding(5, 4, device: ModularPolicyModel.EvalDevice),
                EmbeddingWidth = 4,
            },
            ResidualWidth = 272,
            ResidualBlockCount = 2,
            HiddenToResidualWidthRatio = 1f,
            ActivationFunction = ModularPolicyModel.ActivationFunctionKind.GELU,
            CompressedStateWidth = 64,
        };
        settings.MoveProcessor = Sequential(
            Linear(ModularPolicyModel.GetMoveProcessorInputWidth(settings), 32, device: ModularPolicyModel.EvalDevice),
            GELU(),
            Linear(32, 1, device: ModularPolicyModel.EvalDevice));

        ModularPolicyModel model = new(settings);

        Tensor allLogits = model.GetPolicyLogits(stateTensors).to(CPU);
        Tensor moveIndices = tensor(new long[,] { { 0, 1, 10, 11, 100 } }, dtype: ScalarType.Int64);
        Tensor selectedLogits = model.GetPolicyLogits(stateTensors, moveIndices).to(CPU);
        Tensor expectedSelectedLogits = allLogits.gather(dim: 1, index: moveIndices);

        Assert.That(allLogits.shape[0], Is.EqualTo(1));
        Assert.That(allLogits.shape[1], Is.EqualTo(GameStateEmbedder.PlayHandScoreCount * 2));
        Assert.That(selectedLogits.shape[0], Is.EqualTo(1));
        Assert.That(selectedLogits.shape[1], Is.EqualTo(5));

        float[] actual = selectedLogits.data<float>().ToArray();
        float[] expected = expectedSelectedLogits.data<float>().ToArray();
        for (int moveIndex = 0; moveIndex < expected.Length; ++moveIndex)
            Assert.That(actual[moveIndex], Is.EqualTo(expected[moveIndex]).Within(1e-5f), $"Mismatch at selected move {moveIndex}.");
    }
}
