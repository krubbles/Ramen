namespace Ramen.Game;

// REMEMBER: register all Serializer implementations in Move.cs so the serialization code can find them.

/// <summary>
/// Performs all the setup to begin a round.
/// </summary>
public sealed class StartRoundMove : Move
{

    public override MoveType GetMoveType() => MoveType.StartRound;

    protected override void Apply()
    {
        if (gameState.Stage != StageOfGame.BeginRound)
            throw new InvalidOperationException("Cannot start round, gameState is not in the BeginRound GameStage");

        gameState.Stage = StageOfGame.InRoundAfterHandUsed;

        gameState.HandState.ResetRemainingHandsAndDiscards();
        gameState.ScoringState.ResetCurrentRoundTotalChips();
        gameState.DeckState.ResetDeck();

    }

    protected override void Revert()
    {
        gameState.Stage = StageOfGame.BeginRound;

        gameState.HandState.ResetRemainingHandsAndDiscards();
    }


    public override string ToString()
    {
        return "Start Round";
    }

    public class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.StartRound;

        public void Serialize(GameStateSerializer serializer, Move move)
        {

        }

        public Move Deserialize(GameStateSerializer serializer)
        {
            return new StartRoundMove();
        }
    }
}

public sealed class ReseedMove : Move
{
    public readonly ulong NewRandomState;

    public ReseedMove(ulong newRandomState)
    {
        NewRandomState = newRandomState;
    }

    public override MoveType GetMoveType() => MoveType.Reseed;

    protected override void Apply()
    {
        gameState.Random.SetState(NewRandomState);
    }

    protected override void Revert()
    {
        // Move baseclass already handles reverting the RNG state.
    }

    internal class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.Reseed;

        public void Serialize(GameStateSerializer serializer, Move move)
        {
            ReseedMove reseedMove = (ReseedMove)move;
            serializer.Stream.WriteStruct<ulong>(reseedMove.NewRandomState);
        }

        public Move Deserialize(GameStateSerializer serializer)
        {
            ulong newRandomState = serializer.Stream.ReadStruct<ulong>();
            ReseedMove move = new(newRandomState);
            return move;
        }
    }
}

/// <summary>
/// Move for playing or discarding a hand.
/// </summary>
public sealed class UseHandMove : Move
{
    public readonly bool IsDiscard;
    public readonly byte[] CardIndices;

    Card[] _cards;
    double _roundTotalChipsBeforePlay;

    public UseHandMove(bool isDiscard, params ReadOnlySpan<int> cardIndices)
    {
        IsDiscard = isDiscard;
        CardIndices = new byte[cardIndices.Length];
        for (int i = 0; i < cardIndices.Length; ++i)
            CardIndices[i] = (byte)cardIndices[i];
    }

    public UseHandMove(bool isDiscard, params byte[] cardIndices)
    {
        IsDiscard = isDiscard;
        CardIndices = cardIndices;
    }

    public override MoveType GetMoveType() => MoveType.UseHand;

    public ReadOnlySpan<Card> UsedCards => _cards;

    protected override void Apply()
    {
        gameState.AssertIsStage(StageOfGame.InRoundPlayerChoice);

        _roundTotalChipsBeforePlay = gameState.ScoringState.CurrentRoundTotalChips;
        _cards = new Card[CardIndices.Length];
        for (int i = 0; i < CardIndices.Length; ++i)
            _cards[i] = gameState.HandState.Hand[CardIndices[i]];

        if (IsDiscard)
        {
            gameState.HandState.DiscardHand(CardIndices);
        }
        else
        {
            gameState.HandState.PlayHand(CardIndices);
        }

        gameState.Stage = StageOfGame.InRoundAfterHandUsed;
    }

    protected override void Revert()
    {
        for (int i = 0; i < _cards.Length; ++i)
            gameState.HandState.AddCardToHand(_cards[i]);
        gameState.ScoringState.CurrentRoundTotalChips = _roundTotalChipsBeforePlay;
        if (IsDiscard)
            gameState.HandState.RemainingDiscards++;
        else
            gameState.HandState.RemainingHands++;

        gameState.Stage = StageOfGame.InRoundPlayerChoice;
    }

    public override string ToString()
    {
        return $"{(IsDiscard ? "Discard" : "Play")} Hand: {CardParseUtils.SerializeHand(_cards)}";
    }

    internal sealed class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.UseHand;

        public void Serialize(GameStateSerializer gsSerializer, Move move)
        {
            UseHandMove useHandMove = (UseHandMove)move;

            gsSerializer.Stream.WriteStruct<bool>(useHandMove.IsDiscard);
            gsSerializer.Stream.WriteArrayByteSize<byte>(useHandMove.CardIndices);
        }

        public Move Deserialize(GameStateSerializer gsSerializer)
        {
            bool isDiscard = gsSerializer.Stream.ReadStruct<bool>();
            byte[] cardIndices = gsSerializer.Stream.ReadArrayByteSize<byte>();

            UseHandMove move = new(isDiscard, cardIndices);
            return move;
        }
    }
}

#if false // not currently in use
/// <summary>
/// Move for drawing a fixed quantity of cards.
/// </summary>
public sealed class DrawCardsMove : Move
{
    public readonly int Count;

    Card[] _cards;

    public DrawCardsMove(int count)
    {
        Count = count;
    }

    protected override void Apply()
    {
        int toDraw = Math.Min(gameState.DeckState.RemainingDeckCardCount, Count);
        _cards = gameState.HandState.Draw(toDraw);
    }

    protected override void Revert()
    {
        gameState.HandState.UnDraw(_cards);
    }

    public override string ToString()
    {
        return $"Draw Cards: {CardParseUtils.SerializeHand(_cards)}";
    }
}
#endif

/// <summary>
/// Move for all automatic state changes that happen after a hand is played/discarded. (Mostly redrawing to hand size)
/// </summary>
public sealed class AfterHandUsedMove : Move
{
    Card[] _cards;
    StageOfGame _stage;

    public override MoveType GetMoveType() => MoveType.AfterHandUse;

    protected override void Apply()
    {
        _stage = gameState.Stage;
        int toDraw = Math.Clamp(gameState.HandState.HandSize - gameState.HandState.HandCardCount, 0, gameState.DeckState.RemainingDeckCardCount);
        _cards = gameState.HandState.Draw(toDraw);
        gameState.Stage = StageOfGame.InRoundPlayerChoice;
    }

    protected override void Revert()
    {
        gameState.Stage = _stage;
        gameState.HandState.UnDraw(_cards);
    }

    public override string ToString()
    {
        return $"After Hand Used. Draw Cards: {CardParseUtils.SerializeHand(_cards)}";
    }

    internal class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.AfterHandUse;

        public void Serialize(GameStateSerializer serializer, Move move)
        {
            AfterHandUsedMove afterHandUseMove = (AfterHandUsedMove)move;
        }

        public Move Deserialize(GameStateSerializer serializer)
        {
            AfterHandUsedMove afterHandUseMove = new();
            return afterHandUseMove;
        }
    }
}

/// <summary>
/// Shuffles the state of the <see cref="DeckState.RemainingDeck"/>.
/// </summary>
public sealed class ShuffleMove : Move
{
    public override MoveType GetMoveType() => MoveType.Shuffle;

    protected override void Apply()
    {
        gameState.DeckState.ShuffleDeck();
    }

    protected override void Revert()
    {
        gameState.DeckState.UnshuffleDeck();
    }

    internal class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.Shuffle;

        public void Serialize(GameStateSerializer serializer, Move move) { }

        public Move Deserialize(GameStateSerializer serializer) => new ShuffleMove();
    }
}

/// <summary>
/// A move that stores arbitrary data without affecting game state.
/// </summary>
public sealed class AnnotatingDataMove : Move
{
    public readonly byte[] Data;

    public AnnotatingDataMove(byte[] data)
    {
        Data = data ?? Array.Empty<byte>();
    }

    public override MoveType GetMoveType() => MoveType.AnnotatingData;

    protected override void Apply()
    {
    }

    protected override void Revert()
    {
    }

    public override string ToString()
    {
        return $"Data: {Data.Length} bytes";
    }

    internal class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.AnnotatingData;

        public void Serialize(GameStateSerializer serializer, Move move)
        {
            AnnotatingDataMove trainingDataMove = (AnnotatingDataMove)move;
            serializer.Stream.WriteArrayByteSize<byte>(trainingDataMove.Data);
        }

        public Move Deserialize(GameStateSerializer serializer)
        {
            byte[] data = serializer.Stream.ReadArrayByteSize<byte>();
            return new AnnotatingDataMove(data);
        }
    }
}
