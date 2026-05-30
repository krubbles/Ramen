namespace Ramen.AI;

public interface IAgent
{

    float[][] GetPolicy(float temp, params ReadOnlySpan<GameState> gameStates);

    bool IsGameDone(GameState gameState);


}

public static class IAgentExtensions
{
    public static bool[] IsGameDone(IAgent agent, ReadOnlySpan<GameState> gameStates)
    {
        bool[] results = new bool[gameStates.Length];
        for (int i = 0; i < gameStates.Length; ++i)
            results[i] = agent.IsGameDone(gameStates[i]);
        return results;
    }

    public static void MakeMove(this IAgent agent, float temp, bool annotatePolicy, params ReadOnlySpan<GameState> gameStates)
    {
        int activeGameCount = 0;
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            gameState.AdvanceToNextPlayerChoice();
            if (agent.IsGameDone(gameState))
                continue;

            activeGameCount++;
        }

        if (activeGameCount == 0)
            return;

        GameState[] activeGameStates = new GameState[activeGameCount];
        int activeGameIndex = 0;
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            if (agent.IsGameDone(gameState))
                continue;

            activeGameStates[activeGameIndex] = gameState;
            activeGameIndex++;
        }

        float[][] policies = agent.GetPolicy(temp, activeGameStates);
        for (int policyIndex = 0; policyIndex < activeGameStates.Length; ++policyIndex)
        {
            GameState gameState = activeGameStates[policyIndex];
            Move[] moves = gameState.GetMoveOptions();
            int chosenMoveIndex = AgentUtilities.SampleIndex(policies[policyIndex], gameState.Random);
            if (policies[policyIndex].Length == moves.Length)
                moves[chosenMoveIndex].Apply(gameState);
            else
                AgentUtilities.MoveForPolicyIndex(gameState, chosenMoveIndex).Apply(gameState);

            if (annotatePolicy)
            {
                AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation(policies[policyIndex]);
                annotation.Apply(gameState);
            }
        }
    }
}
