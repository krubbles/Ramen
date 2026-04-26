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
        gameState.StartRound();
    }

    protected override void Revert()
    {
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

/// <summary>
/// Move that reseeds the game's random number generator.
/// </summary>
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
        _roundTotalChipsBeforePlay = gameState.ScoringState.CurrentRoundTotalScore;
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
    }

    protected override void Revert()
    {
        for (int i = 0; i < _cards.Length; ++i)
            gameState.HandState.AddCardToHand(_cards[i]);
        gameState.ScoringState.CurrentRoundTotalScore = _roundTotalChipsBeforePlay;
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

/// <summary>
/// Move for drawing a specific hand from the remaining deck.
/// </summary>
public sealed class DrawSpecificHandMove : Move
{
    public readonly Card[] Cards;

    int[] _removedDeckIndices;
    StageOfGame _stageBeforeApply;

    public DrawSpecificHandMove(params Card[] cards)
    {
        Cards = cards ?? Array.Empty<Card>();
    }

    public override MoveType GetMoveType() => MoveType.DrawSpecificHand;

    protected override void Apply()
    {
        _stageBeforeApply = gameState.Stage;

        _removedDeckIndices = new int[Cards.Length];
        for (int i = 0; i < Cards.Length; ++i)
        {
            _removedDeckIndices[i] = gameState.DeckState.RemoveCardFromRemainingDeck(Cards[i]);
            gameState.HandState.AddCardToHand(Cards[i]);
        }

        gameState.Stage = StageOfGame.InRoundPlayerChoice;
    }

    protected override void Revert()
    {
        gameState.Stage = _stageBeforeApply;
        for (int i = Cards.Length - 1; i >= 0; --i)
        {
            gameState.HandState.RemoveCardFromHand(Cards[i]);
            gameState.DeckState.InsertCardIntoRemainingDeck(Cards[i], _removedDeckIndices[i]);
        }
    }

    public override string ToString()
    {
        return $"Draw Specific Hand: {CardParseUtils.SerializeHand(Cards)}";
    }

    internal sealed class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.DrawSpecificHand;

        public void Serialize(GameStateSerializer gsSerializer, Move move)
        {
            DrawSpecificHandMove drawMove = (DrawSpecificHandMove)move;
            gsSerializer.Stream.WriteArrayByteSize<Card>(drawMove.Cards);
        }

        public Move Deserialize(GameStateSerializer gsSerializer)
        {
            Card[] cards = gsSerializer.Stream.ReadArrayByteSize<Card>();
            return new DrawSpecificHandMove(cards);
        }
    }
}

/// <summary>
/// Move for setting remaining hands and discards.
/// </summary>
public sealed class SetRemainingHandsAndDiscardsMove : Move
{
    public readonly int RemainingHands;
    public readonly int RemainingDiscards;

    int _previousRemainingHands;
    int _previousRemainingDiscards;

    public SetRemainingHandsAndDiscardsMove(int remainingHands, int remainingDiscards)
    {
        RemainingHands = remainingHands;
        RemainingDiscards = remainingDiscards;
    }

    public override MoveType GetMoveType() => MoveType.SetRemainingHandsAndDiscards;

    protected override void Apply()
    {
        _previousRemainingHands = gameState.HandState.RemainingHands;
        _previousRemainingDiscards = gameState.HandState.RemainingDiscards;

        gameState.HandState.RemainingHands = RemainingHands;
        gameState.HandState.RemainingDiscards = RemainingDiscards;
    }

    protected override void Revert()
    {
        gameState.HandState.RemainingHands = _previousRemainingHands;
        gameState.HandState.RemainingDiscards = _previousRemainingDiscards;
    }

    public override string ToString()
    {
        return $"Set Remaining Hands/Discards: {RemainingHands}/{RemainingDiscards}";
    }

    internal sealed class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.SetRemainingHandsAndDiscards;

        public void Serialize(GameStateSerializer gsSerializer, Move move)
        {
            SetRemainingHandsAndDiscardsMove setMove = (SetRemainingHandsAndDiscardsMove)move;
            gsSerializer.Stream.WriteStruct<int>(setMove.RemainingHands);
            gsSerializer.Stream.WriteStruct<int>(setMove.RemainingDiscards);
        }

        public Move Deserialize(GameStateSerializer gsSerializer)
        {
            int remainingHands = gsSerializer.Stream.ReadStruct<int>();
            int remainingDiscards = gsSerializer.Stream.ReadStruct<int>();
            return new SetRemainingHandsAndDiscardsMove(remainingHands, remainingDiscards);
        }
    }
}

/// <summary>
/// Move for setting the current round score. Never a legal move, used for testing.
/// </summary>
public sealed class SetCurrentRoundScoreMove : Move
{
    public readonly float Score;

    float _previousScore;

    public SetCurrentRoundScoreMove(float score)
    {
        Score = score;
    }

    public override MoveType GetMoveType() => MoveType.SetCurrentRoundScore;

    protected override void Apply()
    {
        _previousScore = (float)gameState.ScoringState.CurrentRoundTotalScore;
        gameState.ScoringState.CurrentRoundTotalScore = Score;
    }

    protected override void Revert()
    {
        gameState.ScoringState.CurrentRoundTotalScore = _previousScore;
    }

    public override string ToString()
    {
        return $"Set Current Round Score: {Score}";
    }

    internal sealed class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.SetCurrentRoundScore;

        public void Serialize(GameStateSerializer gsSerializer, Move move)
        {
            SetCurrentRoundScoreMove setMove = (SetCurrentRoundScoreMove)move;
            gsSerializer.Stream.WriteStruct<float>(setMove.Score);
        }

        public Move Deserialize(GameStateSerializer gsSerializer)
        {
            float score = gsSerializer.Stream.ReadStruct<float>();
            return new SetCurrentRoundScoreMove(score);
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

    public override MoveType GetMoveType() => MoveType.AfterHandUse;

    protected override void Apply()
    {
        _cards = gameState.HandState.RedrawAfterHandUse();
    }

    protected override void Revert()
    {
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
    public readonly ushort DataTypeID;
    public readonly byte[] Data;

    public AnnotatingDataMove(ushort dataTypeID, byte[] data)
    {
        DataTypeID = dataTypeID;
        Data = data ?? [];
    }

    public static AnnotatingDataMove FromArray<T>(ushort dataTypeID, ReadOnlySpan<T> array) where T : unmanaged
    {
        if (array.Length == 0)
            return new(dataTypeID, []);

        // Interpret the T[] as bytes and copy into a new byte[] for storage
        ReadOnlySpan<T> tSpan = array;
        ReadOnlySpan<byte> bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(tSpan);
        byte[] data = new byte[bytes.Length];
        bytes.CopyTo(data);

        return new(dataTypeID, data);
    }

    public T[] ToArray<T>() where T : unmanaged
    {
        if (Data == null || Data.Length == 0)
            return [];

        // Cast the byte[] back to a T span and copy into a new T[]
        ReadOnlySpan<byte> bytes = Data;
        ReadOnlySpan<T> tSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, T>(bytes);
        T[] result = new T[tSpan.Length];
        tSpan.CopyTo(result);
        return result;
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
        return $"DataTypeID: {DataTypeID}, Data: {Data.Length} bytes";
    }

    internal class Serializer : IMoveSerializer
    {
        public MoveType MoveType => MoveType.AnnotatingData;

        public void Serialize(GameStateSerializer serializer, Move move)
        {
            AnnotatingDataMove trainingDataMove = (AnnotatingDataMove)move;
            serializer.Stream.WriteStruct<ushort>(trainingDataMove.DataTypeID);
            serializer.Stream.WriteArrayUshortSize<byte>(trainingDataMove.Data);
        }

        public Move Deserialize(GameStateSerializer serializer)
        {
            ushort dataTypeID = serializer.Stream.ReadStruct<ushort>();
            byte[] data = serializer.Stream.ReadArrayUshortSize<byte>();
            return new AnnotatingDataMove(dataTypeID, data);
        }
    }

    /// <summary>
    /// Encodes a probability as a ushort representing the negative natural log probability times 3000.
    /// </summary>
    public static ushort EncodeProb(float prob)
    {
        float nlProb = -MathF.Log(MathF.Max(prob, 1e-9f));
        return (ushort)(nlProb * 3000f + 0.5f);
    }

    /// <summary>
    /// Decodes a probability from a ushort representing the negative natural log probability times 3000.
    /// </summary>
    public static float DecodeProb(ushort encodedProb)
    {
        float nlProb = encodedProb / 3000f;
        return MathF.Exp(-nlProb);
    }
}
