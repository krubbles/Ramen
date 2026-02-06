namespace Ramen.UnitTests;

using Ramen.AI;
using Ramen.Game;

public class NewAgentTests
{
    [Test]
    public void NewAgentMatchesOldAgentLogits()
    {
        // Arrange: model and game states.
        PolicyModel model = new();

        GameState[] gameStates = new GameState[3];
        RamenAgent[] oldAgents = new RamenAgent[gameStates.Length];
        for (int i = 0; i < gameStates.Length; ++i)
        {
            GameState gameState = new(GameData.Default);
            gameState.AdvanceToNextPlayerChoice();
            gameStates[i] = gameState;
            oldAgents[i] = new RamenAgent(gameState, model);
        }

        // Act: compute old agent probability distributions (per-state).
        float[][] oldProbs = new float[gameStates.Length][];
        for (int i = 0; i < oldAgents.Length; ++i)
            oldProbs[i] = oldAgents[i].GetPolicyProbDistManaged(temp: 1f);

        // Act: compute new agent probability distributions (batched).
        NewAgent newAgent = new(gameStates, model);
        float[][] newProbs = newAgent.GetPolicyProbDistManaged(temp: 1f);

        // Assert: probabilities match within tolerance.
        Assert.That(newProbs, Has.Length.EqualTo(oldProbs.Length));
        for (int i = 0; i < newProbs.Length; ++i)
        {
            Assert.That(newProbs[i], Has.Length.EqualTo(oldProbs[i].Length));
            for (int j = 0; j < newProbs[i].Length; ++j)
                Assert.That(newProbs[i][j], Is.EqualTo(oldProbs[i][j]).Within(1e-4f));
        }
    }
}
