using BalatroAI;
using BalatroAI.ConsoleApp;

FastRandom random = FastRandom.SeededByClock();
GameData gameData = new();
for (int i = 0; i < 10; ++i)
{
    GameState gameState = new(gameData);
    gameState.AdvanceToNextPlayerChoice();
    List<Move> moves = gameState.GetMoveOptions();
    Console.WriteLine(gameState.HandToString());

    double bestScore = 0;
    foreach (Move move in moves)
    {
        move.Apply(gameState);
        bestScore = Math.Max(bestScore, gameState.ScoringState.CurrentRoundTotalChips);
        move.Revert(gameState);
    }
    Console.WriteLine(bestScore);
}