namespace Ramen.Game;

using System.Text;
/// <summary>
/// Manages all state transitions for the GameState, including player choices.
/// </summary>
public sealed class MoveState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    public readonly List<Move> MoveHistory = new();

    // List of callbacks triggered by the currently applied move. Move runs them after application/reversion.
    // This makes sure that each callback is only called once per move. Not persistent state.
    readonly HashSet<Action> _activatedCallbacks = new();

    // How many moves are part-way through being applied or reverted. Used to tell a move
    // applied as a side effect of another move from one applied on its own, and to hold
    // callbacks until the outermost operation finishes. Not persistent state.
    int _operationDepth;

    /// <summary>
    /// Whether a move is currently being applied or reverted. A move applied while this
    /// is true is a side effect of that move, and is marked <see cref="Move.IsDerived"/>.
    /// </summary>
    public bool IsApplyingMove => _operationDepth > 0;

    internal void BeginOperation() => _operationDepth++;

    internal void EndOperation() => _operationDepth--;

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

    internal void ScheduleCallback(Action callback)
    {
        if (callback != null)
            _activatedCallbacks.Add(callback);
    }

    internal void RunActivatedCallbacks()
    {
        // Only the outermost move flushes, so a derived move does not fire its parent's
        // callbacks part-way through the parent's application.
        if (IsApplyingMove)
            return;

        foreach (Action action in _activatedCallbacks)
            action.Invoke();
        _activatedCallbacks.Clear();
    }

    public string GameToString()
    {
        StringBuilder sb = new();
        // Only the non-derived moves are replayed; applying them regenerates the
        // derived ones, the same way deserialization does.
        Move[] moves = MoveHistory.FindAll(move => !move.IsDerived).ToArray();
        MoveHistory[0].Revert(GameState);
        foreach (Move move in moves)
        {
            sb.AppendLine(GameState.ToString());
            move.Apply(GameState);
            sb.AppendLine(move.ToString());
        }
        return sb.ToString();
    }

    internal void Serialize(GameStateSerializer serializer)
    {
        // Derived moves are side effects of the moves they follow, so replaying the
        // parent recreates them. Writing them out would apply them a second time.
        int serializedMoveCount = 0;
        foreach (Move move in MoveHistory)
        {
            if (!move.IsDerived)
                serializedMoveCount++;
        }

        serializer.Stream.WriteStartTag("MS");
        serializer.Stream.WriteStruct<int>(serializedMoveCount);
        foreach (Move move in MoveHistory)
        {
            if (!move.IsDerived)
                Move.Serialize(serializer, move);
        }
        serializer.Stream.WriteEndTag("MS");
    }

    internal void Deserialize(GameStateSerializer serializer, int versionNumber)
    {
        if (versionNumber != 2)
            throw new NotSupportedException($"GameState serialization version number {versionNumber} not supported.");
        serializer.Stream.ReadStartTag("MS");
        MoveHistory.Clear();
        int moveCount = serializer.Stream.ReadStruct<int>();
        for (int i = 0; i < moveCount; ++i)
        {
            Move move = Move.Deserialize(serializer);
            move.Apply(GameState);
        }
        serializer.Stream.ReadEndTag("MS");
    }
}
