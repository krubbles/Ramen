namespace Ramen.UnitTests;

using Ramen.Game;

public static class GameStateUnitTestExtentions
{
    public static void PlayRandomGame(this GameState gameState, FastRandom random)
    {
        while (true)
        {
            gameState.AdvanceToNextPlayerChoice();
            Move[] moves = gameState.GetMoveOptions();
            if (moves.Length == 0)
                return;
            int moveIndex = random.Next(moves.Length);
            moves[moveIndex].Apply(gameState);
        }
    }
}