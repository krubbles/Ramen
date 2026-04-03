namespace Ramen.UnitTests;

using System.Linq;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public class PreferenceTrainingPipelineTests
{
    [SetUp]
    public void SetUp()
    {
        TensorManager.Init();
    }


    [Test]
    public void PreferenceValueModelReturnsOneLogitPerState()
    {
        using var scope = NewDisposeScope();

        GameState[] gameStates =
        [
            new(GameData.Default),
            new(GameData.Default),
        ];

        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
            gameStates[stateIndex].AdvanceToNextPlayerChoice();

        PreferenceValueModel model = new();
        GameStateEmbedder gameStateEmbedder = new(gameStates.Length);
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
            gameStateEmbedder.AddGameState(gameStates[stateIndex]);

        GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
        Tensor logits = model.GetLogits(gameStateTensors).to(CPU);

        Assert.That(logits.shape.Length, Is.EqualTo(1));
        Assert.That(logits.shape[0], Is.EqualTo(gameStates.Length));
    }


    [Test]
    public void PreferenceSamplingAgentReturnsNormalizedPolicy()
    {
        GameState gameState = new(GameData.Default);
        PreferenceValueModel model = new();
        PreferenceSamplingAgent agent = new(model);

        float[][] policy = agent.GetPolicy(temp: 1f, gameState);
        float[] statePolicy = policy[0];

        Assert.That(statePolicy, Is.Not.Null);
        Assert.That(statePolicy.Length, Is.GreaterThan(0));
        Assert.That(statePolicy.Sum(), Is.EqualTo(1f).Within(1e-4f));
        Assert.That(statePolicy.All(probability => probability >= 0f), Is.True);
    }
}
