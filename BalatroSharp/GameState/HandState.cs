using System.Numerics;

namespace Ramen.Game;

/// <summary>
/// Holds the state of the player's hand including remaining plays and discards.
/// </summary>
public class HandState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    readonly Card[] _handBuffer = new Card[30], _activeHandBuffer = new Card[30];

    /// <summary>
    /// Number of cards in the player's hand currently.
    /// </summary>
    public int HandCardCount { get; private set; }

    /// <summary>
    /// Number of cards in the currently active hand (played or discarded). Zero if no active hand. Not persistent state.
    /// </summary>
    public int ActiveHandCardCount { get; private set; }

    /// <summary>
    /// The player's hand.
    /// </summary>
    public Span<Card> Hand => _handBuffer.AsSpan(0, HandCardCount);

    /// <summary>
    /// The hand the player is currently playing or discarding, if applicable.
    /// Not persistent state.
    /// </summary>
    public Span<Card> ActiveHand => _activeHandBuffer.AsSpan(0, ActiveHandCardCount);


    int _remainingHands, _remainingDiscards;

    /// <summary>
    /// The number of hands the player can still play this round.
    /// </summary>
    public int RemainingHands
    {
        get => _remainingHands;
        internal set
        {
            if (_remainingHands == value)
                return;
            _remainingHands = value;
            GameState.MoveState.ScheduleCallback(OnRemainingHandsOrDiscardsChanged);
        }
    }

    /// <summary>
    /// The number of discard the player can still use this round.
    /// </summary>
    public int RemainingDiscards
    {
        get => _remainingDiscards;
        internal set
        {
            if (_remainingDiscards == value)
                return;
            _remainingDiscards = value;
            GameState.MoveState.ScheduleCallback(OnRemainingHandsOrDiscardsChanged);
        }
    }

    /// <summary>
    /// Pattern matching results for the currently active (played/discarded) hand. Used by jokers and scoring. Not persistent state.
    /// </summary>
    public HandPatterns ActiveHandPatterns;

    /// <summary>
    /// If the player has less then this number of cards in their hand after playing/discarding, they will draw up to this number.
    /// </summary>
    public int HandSize = 8;

    /// <summary>
    /// The number of hands the player can play each round.
    /// </summary>
    public int HandsPerRound = 4;

    /// <summary>
    /// The number of discards the player can use each round.
    /// </summary>
    public int DiscardsPerRound = 3;

    public Action OnHandChanged;
    public Action OnRemainingHandsOrDiscardsChanged;

    public HandState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
        HandsPerRound = _gameData.Hands;
        DiscardsPerRound = _gameData.Discards;
    }

    public override int GetHashCode()
    {
        return
            463507903 +
            Card.HashCardSet(Hand) * 210384047 +
            RemainingDiscards * 991603139 +
            RemainingHands * 845702419 +
            HandSize * 422750039 +
            HandsPerRound * 226922317 +
            DiscardsPerRound * 535572281;
    }

    internal void CloneFrom(HandState other)
    {
        RemainingHands = other.RemainingHands;
        RemainingDiscards = other.RemainingDiscards;
        other.Hand.CopyTo(_handBuffer);
        HandCardCount = other.HandCardCount;

        ActiveHandPatterns = other.ActiveHandPatterns;
        HandSize = other.HandSize;
        HandsPerRound = other.HandsPerRound;
        DiscardsPerRound = other.DiscardsPerRound;
    }

    internal void AddCardToHand(Card card)
    {
        int index = Array.BinarySearch(_handBuffer, 0, HandCardCount, card);
        if (index < 0)
            index = ~index;
        for (int i = HandCardCount; i > index; --i)
        {
            _handBuffer[i] = _handBuffer[i - 1];
        }
        _handBuffer[index] = card;
        HandCardCount++;
        GameState.MoveState.ScheduleCallback(OnHandChanged);
    }

    internal void RemoveCardFromHand(Card card)
    {
        int index = Array.IndexOf(_handBuffer, card, 0, HandCardCount);
        if (index < 0)
            throw new ArgumentException($"Card {card} not found in hand, cannot be removed");
        HandCardCount--;
        for (int i = index; i < HandCardCount; ++i)
            _handBuffer[i] = _handBuffer[i + 1];
        GameState.MoveState.ScheduleCallback(OnHandChanged);
    }

    internal void ResetRemainingHandsAndDiscards()
    {
        RemainingHands = HandsPerRound;
        RemainingDiscards = DiscardsPerRound;
        GameState.MoveState.ScheduleCallback(OnHandChanged);
    }

    internal void ZeroRemainingHandsAndDiscards()
    {
        RemainingHands = 0;
        RemainingDiscards = 0;
        GameState.MoveState.ScheduleCallback(OnHandChanged);
    }


    internal Card[] Draw(int count)
    {
        Card[] cards = new Card[count];
        GameState.DeckState.Draw(cards);
        for (int i = 0; i < count; ++i)
            AddCardToHand(cards[i]);
        return cards;
    }

    internal void UnDraw(Card[] cards)
    {
        GameState.DeckState.UnDraw(cards);
        for (int i = 0; i < cards.Length; ++i)
            RemoveCardFromHand(cards[i]);
    }

    internal double PlayHand(ReadOnlySpan<int> cardIndices)
    {
        if (RemainingHands < 1)
            throw new Exception("Cannot play hand, out of hands.");
        RemainingHands--;

        Span<Card> playedHand = stackalloc Card[cardIndices.Length];
        for (int i = 0; i < cardIndices.Length; i++)
        {
            playedHand[i] = _handBuffer[cardIndices[i]];
            _handBuffer[cardIndices[i]] = Card.Null;
        }

        SetActiveHand(playedHand);
        GameState.PatternMatchingState.MatchHand(playedHand, out ActiveHandPatterns);

        // remove played cards
        int writeIndex = 0;
        for (int i = 0; i < HandCardCount; ++i)
        {
            Card card = _handBuffer[i];
            if (!card.IsNull)
                _handBuffer[writeIndex++] = card;
        }
        HandCardCount = writeIndex;
        GameState.MoveState.ScheduleCallback(OnHandChanged);

        GameState.JokerState.OnBeforePlayHand();

        double score = GameState.ScoringState.ScoreActiveHand();

        return score;
    }

    internal void DiscardHand(ReadOnlySpan<int> cardIndices)
    {
        if (RemainingDiscards < 1)
            throw new Exception("Cannot discard, out of discards");
        RemainingDiscards--;

        GameState.JokerState.OnDiscardHand();
        for (int i = 0; i < cardIndices.Length; i++)
        {
            GameState.JokerState.OnDiscardCard(_handBuffer[cardIndices[i]]);
            _handBuffer[cardIndices[i]] = Card.Null;
        }

        // remove discarded cards
        int writeIndex = 0;
        for (int i = 0; i < HandCardCount; ++i)
        {
            Card card = _handBuffer[i];
            if (!card.IsNull)
                _handBuffer[writeIndex++] = card;
        }
        HandCardCount = writeIndex;

    }

    internal void SetActiveHand(ReadOnlySpan<Card> hand)
    {
        ActiveHandCardCount = hand.Length;
        hand.CopyTo(_activeHandBuffer);
    }


    internal void AppendLegalUseHandMoves(List<Move> moves)
    {
        if (RemainingDiscards == 0 && RemainingHands == 0)
            return;

        Span<int> indices = stackalloc int[5];
        for (int playMask = 1; playMask < (1 << HandCardCount); ++playMask)
        {
            int handSize = 0;
            bool skip = false;
            for (int i = 0; i < 8; ++i)
            {
                if (((playMask >> i) & 1) != 0)
                {
                    if (handSize >= indices.Length)
                    {
                        skip = true;
                        break;
                    }
                    indices[handSize++] = i;
                }
            }
            if (skip)
                continue;
            int[] indicesArray = indices[0..handSize].ToArray();
            if (RemainingHands > 0)
                moves.Add(new UseHandMove(false, indicesArray));
            if (RemainingDiscards > 0)
                moves.Add(new UseHandMove(true, indicesArray));
        }
    }
}


/// <summary>
/// Move for playing or discarding a hand.
/// </summary>
public sealed class UseHandMove : Move
{
    public readonly bool IsDiscard;
    public readonly int[] CardIndices;

    Card[] _cards;
    double _roundTotalChipsBeforePlay;

    public UseHandMove(bool isDiscard, params int[] cardIndices)
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
            gsSerializer.Stream.WriteArray<int>(useHandMove.CardIndices);
        }

        public Move Deserialize(GameStateSerializer gsSerializer)
        {
            bool isDiscard = gsSerializer.Stream.ReadStruct<bool>();
            int[] cardIndices = gsSerializer.Stream.ReadArray<int>();

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