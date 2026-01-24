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
        if (_moveStep != -1)
            throw new InvalidOperationException("Trying to apply move that has already been applied.");

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
        moveSerializer.Serialize(gsSerializer, move);
    }

    internal static Move Deserialize(GameStateSerializer serializer)
    {
        MoveType moveType = serializer.Stream.ReadStruct<MoveType>();

        IMoveSerializer moveSerializer = MoveSerializers[moveType];

        Move move = moveSerializer.Deserialize(serializer);
        return move;
    }

    public static readonly Dictionary<MoveType, IMoveSerializer> MoveSerializers = new()
    {
        { MoveType.UseHand, new UseHandMove.Serializer() },
        { MoveType.AfterHandUse, new AfterHandUsedMove.Serializer() },
        { MoveType.StartRound, new StartRoundMove.Serializer() },
        { MoveType.Reseed, new ReseedMove.Serializer() },
        { MoveType.DrawSpecificHand, new DrawSpecificHandMove.Serializer() },
        { MoveType.SetRemainingHandsAndDiscards, new SetRemainingHandsAndDiscardsMove.Serializer() },
        { MoveType.AnnotatingData, new AnnotatingDataMove.Serializer() },
    };
}

// VERY IMPORTANT. THIS IS SERIALIZED AS AN INT.
// ADDING NEW VALUES THAT ARE NOT AT THE END WILL BREAK SERIALIZATION.
public enum MoveType : byte
{
    None,
    UseHand,
    AfterHandUse,
    StartRound,
    Reseed,
    Shuffle,
    AnnotatingData,
    DrawSpecificHand,
    SetRemainingHandsAndDiscards,
}

public interface IMoveSerializer
{
    public MoveType MoveType { get; }

    public void Serialize(GameStateSerializer gsSerializer, Move move);

    public Move Deserialize(GameStateSerializer gsSerializer);
}