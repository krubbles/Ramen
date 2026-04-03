namespace Ramen.AgentTools;

using Ramen.Game;

public interface IAgent
{
    void MakeMove(float temp, bool annotatePolicy, params ReadOnlySpan<GameState> gameStates);

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
}
