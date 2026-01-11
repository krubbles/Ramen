namespace Ramen.Game;

/// <summary>
/// A class that represents a state transition for a <see cref="gameState"/>. All state changes are handled by moves.
/// </summary>
public abstract class Move
{
    // State set during move application and used to properly revert moves
    protected GameState gameState;
    ulong _rngState;
    int _moveStep = -1;

    public void Apply(GameState gameState)
    {
        this.gameState = gameState;
        _rngState = gameState.Random.GetState();
        _moveStep = gameState.MoveState.MoveHistory.Count;
        gameState.MoveState.MoveHistory.Add(this);
        Apply();

        gameState.MoveState.RunActivatedCallbacks();
    }

    public void Revert(GameState gameState)
    {
        if (this.gameState == null)
            throw new InvalidOperationException("Cannot revert a move that hasn't been applied.");
        if (this.gameState != gameState)
            throw new InvalidOperationException("Cannot revert move on a different game state than it was applied to.");
        if (gameState.MoveState.MoveHistory[_moveStep] != this)
            throw new InvalidOperationException("Cannot revert move, it cannot be found in the move history.");

        gameState.MoveState.RevertToStep(_moveStep + 1);

        this.gameState.Random.SetState(_rngState);
        Revert();
        this.gameState.Random.SetState(_rngState);

        this.gameState = null;
        this._moveStep = -1;
        _rngState = default;
        gameState.MoveState.MoveHistory.RemoveAt(gameState.MoveState.MoveHistory.Count - 1);

        gameState.MoveState.RunActivatedCallbacks();
    }

    public bool IsApplied => _moveStep >= 0;

    protected abstract void Apply();

    protected abstract void Revert();

    public abstract MoveType GetMoveType();

    internal static void Serialize(GameStateSerializer gsSerializer, Move move)
    {
        MoveType moveType = move.GetMoveType();
        IMoveSerializer moveSerializer = MoveSerializers[moveType];

        gsSerializer.Stream.WriteStruct<MoveType>(moveType);
        gsSerializer.Stream.WriteStruct<int>(move._moveStep);
        gsSerializer.Stream.WriteStruct<ulong>(move._rngState);

        moveSerializer.Serialize(gsSerializer, move, move.IsApplied);
    }

    internal static Move Deserialize(GameStateSerializer serializer)
    {
        MoveType moveType = serializer.Stream.ReadStruct<MoveType>();
        int moveStep = serializer.Stream.ReadStruct<int>();
        ulong rngState = serializer.Stream.ReadStruct<ulong>();

        bool isApplied = moveStep >= 0;
        IMoveSerializer moveSerializer = MoveSerializers[moveType];

        Move move = moveSerializer.Deserialize(serializer, isApplied);

        move._moveStep = moveStep;
        move._rngState = rngState;
        if (isApplied)
            move.gameState = serializer.GameState;
        return move;
    }

    public static readonly Dictionary<MoveType, IMoveSerializer> MoveSerializers = new()
    {
    };
}

public enum MoveType
{
    None,
    UseHand,
    Redraw,
}

public interface IMoveSerializer
{
    public MoveType MoveType { get; }

    public void Serialize(GameStateSerializer gsSerializer, Move move, bool isApplied);

    public Move Deserialize(GameStateSerializer gsSerializer, bool isApplied);
}