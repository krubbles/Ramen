namespace Ramen.AgentTools;

using Ramen.Game;

public interface ITrainingRunAnalyzer
{
    void Analyze(IEnumerable<GameState> games, CSVBuilder output);
}
