namespace BalatroAI;

/// <summary>
/// A class that represents a state transition for a <see cref="gameState"/>. All state changes are handled by moves.
/// </summary>
public abstract class Move
{
    // State set during move application and used to properly revert moves
    protected GameState gameState;
    ulong _rngState;
    int _moveStep;

    public void Apply(GameState gameState)
    {
        this.gameState = gameState;
        _rngState = gameState.Random.GetState();
        _moveStep = gameState.MoveState.MoveHistory.Count;
        gameState.MoveState.MoveHistory.Add(this);
        Apply();
    }

    public void Revert(GameState gameState)
    {
        if (this.gameState != gameState)
            throw new InvalidOperationException("Cannot revert move on a different game state than it was applied to.");
        if (gameState.MoveState.MoveHistory[_moveStep] != this)
            throw new InvalidOperationException("Cannot revert move, it cannot be found in the move history.");

        gameState.MoveState.RevertToStep(_moveStep + 1);

        this.gameState.Random.SetState(_rngState);
        Revert();
        this.gameState.Random.SetState(_rngState);

        this.gameState = null;
        _rngState = default;
        gameState.MoveState.MoveHistory.RemoveAt(gameState.MoveState.MoveHistory.Count - 1);
    }

    protected abstract void Apply();

    protected abstract void Revert();
}