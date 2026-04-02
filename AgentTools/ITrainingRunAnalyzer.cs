namespace Ramen.Agents;

using Ramen.Game;

public interface ITrainingRunAnalyzer
{
    void Analyze(IEnumerable<GameState> games, CSVBuilder output);
}
