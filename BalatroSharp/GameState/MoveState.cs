namespace BalatroAI;

/// <summary>
/// Manages all state transitions for the GameState, including player choices. 
/// </summary>
public sealed class MoveState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    public readonly List<Move> MoveHistory = new();
    
    readonly List<Move> _moveBuffer = new();

    // List of callbacks triggered by the currently applied move. Move runs them after application/reversion.
    // This makes sure that each callback is only called once per move.
    readonly HashSet<Action> _activatedCallbacks = new();

    public MoveState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
    }


    public int MoveStep => MoveHistory.Count;


    internal void CloneFrom(in MoveState other)
    {
        MoveHistory.Clear();
        MoveHistory.AddRange(other.MoveHistory);
    }

    public void MakeMove(Move move)
    {
        move.Apply(GameState);
    }

    public void RevertLastMove()
    {
        if (MoveHistory.Count == 0)
            throw new InvalidOperationException("No moves to revert");
        MoveHistory[^1].Revert(GameState);
    }

    public void RevertToStep(int moveStep)
    {
        int toRevertCount = MoveHistory.Count - moveStep;
        for (int i = 0; i < toRevertCount; ++i)
            RevertLastMove();
    }

    internal void RegisterActivatedCallback(Action callback)
    {
        if (callback != null)
            _activatedCallbacks.Add(callback);
    }

    internal void RunActivatedCallbacks()
    {
        foreach (Action action in _activatedCallbacks)
            action.Invoke();
        _activatedCallbacks.Clear();
    }
}
