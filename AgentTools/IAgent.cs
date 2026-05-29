namespace Ramen.AgentTools;

using Ramen.Game;

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
        float[][] policies = agent.GetPolicy(temp, gameStates);
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            if (agent.IsGameDone(gameState))
                continue;

            Move[] moves = gameState.GetMoveOptions();
            int chosenMoveIndex = AgentUtilities.SampleIndex(policies[stateIndex], gameState.Random);
            moves[chosenMoveIndex].Apply(gameState);

            if (annotatePolicy)
            {
                AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation(policies[stateIndex]);
                annotation.Apply(gameState);
            }
        }
    }
}
