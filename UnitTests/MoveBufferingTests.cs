namespace Ramen.UnitTests;

using Ramen.Game;

public class MoveBufferingTests
{
    const int GameCount = 1000;

    [Test]
    public void RandomGameTracesRevertToEquivalentHashHistory()
    {
        FastRandom random = new(29);

        for (int gameIndex = 0; gameIndex < GameCount; ++gameIndex)
        {
            GameData gameData = new()
            {
                RandomizeSeed = false,
                Seed = gameIndex,
            };
            GameState gameState = new(gameData);
            int originMoveStep = gameState.MoveState.MoveStep;
            List<int> hashHistory = [gameState.GetHashCode()];

            PlayRandomTrace(gameState, random, hashHistory);
            RevertTraceAndAssertHashHistory(gameState, originMoveStep, hashHistory, gameIndex);
        }
    }

    static void PlayRandomTrace(GameState gameState, FastRandom random, List<int> hashHistory)
    {
        while (true)
        {
            Move[] moves = gameState.GetMoveOptions();
            if (moves.Length == 0)
                return;

            int moveIndex = moves.Length == 1 ? 0 : random.Next(moves.Length);
            moves[moveIndex].Apply(gameState);
            hashHistory.Add(gameState.GetHashCode());
        }
    }

    static void RevertTraceAndAssertHashHistory(GameState gameState, int originMoveStep, List<int> hashHistory, int gameIndex)
    {
        int traceMoveCount = hashHistory.Count - 1;
        Assert.That(
            gameState.MoveState.MoveStep,
            Is.EqualTo(originMoveStep + traceMoveCount),
            $"Game {gameIndex} did not record the expected number of trace moves.");

        for (int traceStep = traceMoveCount; traceStep >= 0; --traceStep)
        {
            Assert.That(
                gameState.GetHashCode(),
                Is.EqualTo(hashHistory[traceStep]),
                $"Game {gameIndex}, trace step {traceStep} hash mismatch while reverting.");

            if (traceStep > 0)
                gameState.MoveState.RevertLastMove();
        }

        Assert.That(
            gameState.MoveState.MoveStep,
            Is.EqualTo(originMoveStep),
            $"Game {gameIndex} did not revert back to the trace origin.");
    }
}
