namespace Ramen.UnitTests;

using Ramen.Game;

public static class GameStateUnitTestExtentions
{
    public static void PlayRandomGame(this GameState gameState, FastRandom random)
    {
        while (true)
        {
            gameState.AdvanceToNextPlayerChoice();
            List<Move> moves = gameState.GetMoveOptions();
            if (moves.Count == 0)
                return;
            int moveIndex = random.Next(moves.Count);
            moves[moveIndex].Apply(gameState);
        }
    }
}