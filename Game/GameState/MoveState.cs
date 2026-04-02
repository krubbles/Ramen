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
        foreach (Action action in _activatedCallbacks)
            action.Invoke();
        _activatedCallbacks.Clear();
    }

    public string GameToString()
    {
        StringBuilder sb = new();
        Move[] moves = MoveHistory.ToArray();
        moves[0].Revert(GameState);
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
        serializer.Stream.WriteStartTag("MS");
        serializer.Stream.WriteStruct<int>(MoveHistory.Count);
        foreach (Move move in MoveHistory)
        {
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
