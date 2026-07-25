namespace Ramen.Game;

// Note: implementations of move should go in MoveImplementations.cs

/// <summary>
/// A class that represents a state transition for a <see cref="gameState"/>. All state changes are handled by moves.
/// </summary>
public abstract class Move
{
    // State set during move application and used to properly revert moves
    protected GameState gameState;
    ulong _rngState;
    StageOfGame _stage;
    int _moveStep = -1;
    bool _isDerived;

    /// <summary>
    /// Applies this move to the given <paramref name="gameState"/>. Throws if the move has already been applied.
    /// The move gets added to the <see cref="GameState.MoveState.MoveHistory"/> of the given <paramref name="gameState"/>.
    /// </summary>
    public void Apply(GameState gameState)
    {
        if (IsApplied)
            throw new InvalidOperationException("Trying to apply move that has already been applied.");

        this.gameState = gameState;
        // Read before the depth is raised: a move is derived exactly when some other
        // move was already mid-application as this one was applied.
        _isDerived = gameState.MoveState.IsApplyingMove;
        _rngState = gameState.Random.GetState();
        _stage = gameState.Stage;
        _moveStep = gameState.MoveState.MoveHistory.Count;
        gameState.MoveState.MoveHistory.Add(this);

        gameState.MoveState.BeginOperation();
        try
        {
            Apply();
        }
        finally
        {
            gameState.MoveState.EndOperation();
        }

        gameState.MoveState.RunActivatedCallbacks();
    }

    /// <summary>
    /// Restores the gameState to before the move was made. Throws if this move isn't in <paramref name="gameState"/>'s move history, or if the move hasn't been applied.
    /// Will revert all moves applied after this move before reverting this move.
    /// </summary>
    public void Revert(GameState gameState)
    {
        if (!IsApplied)
            throw new InvalidOperationException("Cannot revert a move that hasn't been applied.");
        if (this.gameState != gameState)
            throw new InvalidOperationException("Cannot revert move on a different game state than it was applied to.");
        if (gameState.MoveState.MoveHistory[_moveStep] != this)
            throw new InvalidOperationException("Cannot revert move, it cannot be found in the move history.");

        gameState.MoveState.BeginOperation();
        try
        {
            gameState.MoveState.RevertToStep(_moveStep + 1);

            this.gameState.Random.SetState(_rngState);
            Revert();
            this.gameState.Random.SetState(_rngState);
            gameState.Stage = _stage;
        }
        finally
        {
            gameState.MoveState.EndOperation();
        }

        this.gameState = null;
        _moveStep = -1;
        _stage = StageOfGame.Null;
        _rngState = default;
        gameState.MoveState.MoveHistory.RemoveAt(gameState.MoveState.MoveHistory.Count - 1);

        gameState.MoveState.RunActivatedCallbacks();
    }

    public bool IsApplied => _moveStep >= 0;

    /// <summary>
    /// Whether this move was applied as a side effect of another move, rather than as a
    /// game transition in its own right. Detected automatically at application time, so
    /// moves do not have to declare it.
    /// <para>
    /// Derived moves take part in rollback normally: they sit after their parent in
    /// <see cref="MoveState.MoveHistory"/>, so reverting the parent reverts them first.
    /// They are NOT serialized, because replaying the parent regenerates them; writing
    /// them out would apply them twice on deserialization.
    /// </para>
    /// <para>
    /// Only meaningful while the move is applied. A reverted move keeps whatever value it
    /// last had until it is applied again.
    /// </para>
    /// </summary>
    public bool IsDerived => _isDerived;

    protected abstract void Apply();

    protected abstract void Revert();

    /// <summary>
    /// Returns an enum value unique to the implementing type of <see cref="Move"/>. Primarily used in serialization.
    /// </summary>
    public abstract MoveType GetMoveType();

    /// <summary>
    /// Serializes the move. State used to revert the move is not serialized.
    /// </summary>
    internal static void Serialize(GameStateSerializer gsSerializer, Move move)
    {
        MoveType moveType = move.GetMoveType();
        IMoveSerializer moveSerializer = MoveSerializers[moveType];

        gsSerializer.Stream.WriteStruct<MoveType>(moveType);
        moveSerializer.Serialize(gsSerializer, move);
    }

    /// <summary>
    /// Deserializes the move. State used to revert the move is not deserialized.
    /// </summary>
    internal static Move Deserialize(GameStateSerializer serializer)
    {
        MoveType moveType = serializer.Stream.ReadStruct<MoveType>();

        IMoveSerializer moveSerializer = MoveSerializers[moveType];

        Move move = moveSerializer.Deserialize(serializer);
        return move;
    }

    /// <summary>
    /// A dictionary mapping each <see cref="MoveType"/> to its corresponding <see cref="IMoveSerializer"/>.
    /// All move implementations must have a corresponding serializer in this dictionary.
    /// </summary>
    public static readonly Dictionary<MoveType, IMoveSerializer> MoveSerializers = new()
    {
        { MoveType.UseHand, new UseHandMove.Serializer() },
        { MoveType.AfterHandUse, new AfterHandUsedMove.Serializer() },
        { MoveType.StartRound, new StartRoundMove.Serializer() },
        { MoveType.Reseed, new ReseedMove.Serializer() },
        { MoveType.Shuffle, new ShuffleMove.Serializer() },
        { MoveType.DrawSpecificHand, new DrawSpecificHandMove.Serializer() },
        { MoveType.SetRemainingHandsAndDiscards, new SetRemainingHandsAndDiscardsMove.Serializer() },
        { MoveType.SetCurrentRoundScore, new SetCurrentRoundScoreMove.Serializer() },
        { MoveType.AnnotatingData, new AnnotatingDataMove.Serializer() },
        { MoveType.BuyShopOffer, new BuyShopOfferMove.Serializer() },
        { MoveType.ExitShop, new ExitShopMove.Serializer() },
        { MoveType.Reroll, new RerollMove.Serializer() },
        { MoveType.EnterShop, new EnterShopMove.Serializer() },
        { MoveType.EndRound, new EndRoundMove.Serializer() },
        { MoveType.SetJokerState, new SetJokerStateMove.Serializer() },
    };
}

// Serialized as an byte. New values must be added to the end to avoid breaking deserialization.
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
    SetCurrentRoundScore,
    BuyShopOffer,
    ExitShop,
    Reroll,
    EnterShop,
    EndRound,
    SetJokerState,
}

/// <summary>
/// An interface for writing classes that serialize and deserialize a specific implementation of <see cref="Move"/>.
/// Each implementation should have a singular corresponding <see cref="IMoveSerializer"/>.
/// The serializer should NOT serialize data used to revert the move.
/// </summary>
public interface IMoveSerializer
{
    public MoveType MoveType { get; }

    public void Serialize(GameStateSerializer gsSerializer, Move move);

    public Move Deserialize(GameStateSerializer gsSerializer);
}
