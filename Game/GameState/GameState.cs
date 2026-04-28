namespace Ramen.Game;

/// <summary>
/// Represents the state of a game of Balatro. Can also represent a played game, since the move history is serialized.
/// </summary>
public sealed class GameState
{
    public readonly GameData GameData;

    /// <summary>
    /// All the state involved in scoring. Includes chips and mult of currently scored hand, current round total chips, and some other data.s
    /// </summary>
    public readonly ScoringState ScoringState;

    /// <summary>
    /// The player's full deck and their remaining deck for the round.
    /// </summary>
    public readonly DeckState DeckState;

    /// <summary>
    /// The player's hand, remaining hands, and remaining discards, and other hand-related state. Also contains <see cref="HandPatterns"/> for the active (currently scoring) hand.
    /// </summary>
    public readonly HandState HandState;

    /// <summary>
    /// The jokers the player owns and any assosiated state.
    /// </summary>
    public readonly JokerState JokerState;

    /// <summary>
    /// State related to the shop, including money.
    /// </summary>
    public readonly ShopState ShopState;

    /// <summary>
    /// State used for matching patterns. Not a lot of state, but some flags for things like 4-card straights are stored here to be modified by jokers.
    /// </summary>
    public readonly PatternMatchingState PatternMatchingState;

    /// <summary>
    /// The history of <see cref="Move"/> objects applied to this GameState. Moves support rollback and serialization.
    /// </summary>
    public readonly MoveState MoveState;

    /// <summary>
    /// The random number generator for this game state.
    /// </summary>
    public readonly FastRandom Random;

    /// <summary>
    /// The current stage of the game. (ex: in store; in round)
    /// </summary>
    public StageOfGame Stage { get; internal set; }

    /// <summary>
    /// The current round of the game. First round is round 1. First store is still round 1.
    /// </summary>
    public int Round { get; internal set; } = 0;

    readonly List<Move> _currentLegalMovesBuffer = new();

    public GameState(GameData gameData)
    {
        GameData = gameData;
        Random = new(0);

        ScoringState = new(this);
        DeckState = new(this);
        HandState = new(this);
        JokerState = new(this);
        ShopState = new(this);
        PatternMatchingState = new(this);
        MoveState = new(this);

        int seed = gameData.RandomizeSeed ? FastRandom.SeededByClock().Next() : gameData.Seed;
        new ReseedMove(FastRandom.SeedToState((ulong)seed)).Apply(this);

        GameData.InitStartingDeck(this);

        Stage = StageOfGame.BeginRound;
    }

    public override int GetHashCode()
    {
        int hash = 562877087 ^
            (int)Stage * 301499677;

        if (StageHasRoundState(Stage))
        {
            hash ^=
                ScoringState.GetHashCode() ^
                HandState.GetHashCode() ^
                DeckState.GetHashCode();
        }

        return hash;
    }

    [Obsolete("Use rollback via MoveState.RevertToStep() instead")]
    public void CloneFrom(GameState other)
    {
        DeckState.CloneFrom(other.DeckState);
        ScoringState.CloneFrom(other.ScoringState);
        HandState.CloneFrom(other.HandState);
        MoveState.CloneFrom(other.MoveState);
    }

    public void Reseed(bool shuffle = true)
    {
        ulong newState = FastRandom.SeededByClock().GetState();
        new ReseedMove(newState).Apply(this);
        if (shuffle)
        {
            new ShuffleMove().Apply(this);
        }
    }

    public bool GameIsDone =>
        HandState.RemainingHands == 0 &&
        ScoringState.CurrentRoundTotalScore < ScoringState.CurrentRoundThresholdScore;

    public bool IsPlayerChoice
    {
        get
        {
            if (GameIsDone)
                return false;
            if (Stage == StageOfGame.InRoundPlayerChoice)
                return true;
            return false;
        }
    }

    internal void BeginRound()
    {
        AssertIsStage(StageOfGame.BeginRound);
        Stage = StageOfGame.InRoundAfterHandUsed;
        Round++;
        HandState.ClearHand();
        HandState.ResetRemainingHandsAndDiscards();
        ScoringState.ResetCurrentRoundTotalChips();
        DeckState.ResetDeck();
    }

    internal void EndRound()
    {
        AssertIsStage(StageOfGame.EndRound);
        Stage = StageOfGame.EnterShop;
        int interest = Math.Min(ShopState.Money / 5, 5);
        int rewardMoney = GameData.GetRewardMoney(Round);
        int handMoney = HandState.RemainingHands;
        ShopState.Money += interest + rewardMoney + handMoney;
        Stage = StageOfGame.EnterShop;
    }

    /// <summary>
    /// Returns a list of legal moves.
    /// If there is 1, then there is an automatic state change that must happen.
    /// If there are multiple, the player/agent has a choice to make.
    /// </summary>
    public Move[] GetMoveOptions()
    {
        // note: make sure this function is synced with IsPlayerChoice().
        // if IsPlayerChoice
        _currentLegalMovesBuffer.Clear();
        switch (Stage)
        {
            case StageOfGame.BeginRound:
                _currentLegalMovesBuffer.Add(new StartRoundMove());
                break;
            case StageOfGame.InRoundPlayerChoice:
                HandState.AppendLegalUseHandMoves(_currentLegalMovesBuffer);
                break;
            case StageOfGame.InRoundAfterHandUsed:
                _currentLegalMovesBuffer.Add(new AfterHandUsedMove());
                break;
            case StageOfGame.EndRound:
                _currentLegalMovesBuffer.Add(new EndRoundMove());
                break;
            case StageOfGame.EnterShop:
                _currentLegalMovesBuffer.Add(new EnterShopMove());
                break;
            case StageOfGame.InShop:
                ShopState.AppendLegalMoves(_currentLegalMovesBuffer);
                break;

        }
        return _currentLegalMovesBuffer.ToArray();
    }

    public void AdvanceToNextPlayerChoice()
    {
        while (!IsPlayerChoice)
        {
            if (GameIsDone)
                return;
            GetMoveOptions()[0].Apply(this);
        }
    }

    public void Serialize(Stream stream)
    {
        int versionNumber = 2;

        GameStateSerializer serializer = new(this, stream);

        serializer.Stream.WriteStartTag("GS");
        serializer.Stream.WriteStruct<int>(versionNumber);
        MoveState.Serialize(serializer);
        serializer.Stream.WriteStruct<int>(GetHashCode());
        serializer.Stream.WriteEndTag("GS");
        serializer.Stream.Flush();
    }

    public void Deserialize(Stream stream)
    {
        GameStateSerializer serializer = new(this, stream);

        serializer.Stream.ReadStartTag("GS");
        int versionNumber = serializer.Stream.ReadStruct<int>();
        MoveState.Deserialize(serializer, versionNumber);
        int hash = serializer.Stream.ReadStruct<int>();
        serializer.Stream.ReadEndTag("GS");

#if DEBUG
        if (hash != GetHashCode())
            throw new Exception("Deserialization hash mismatch.");
#endif
    }

    public override string ToString()
    {
        return $"[Hand: {CardParseUtils.SerializeHand(HandState.Hand)}, RH {HandState.RemainingHands}, RD: {HandState.RemainingDiscards}]";
    }

    internal void AssertIsStage(StageOfGame stage)
    {
        if (Stage != stage)
            throw new InvalidOperationException($"GameState is not in the expected stage. Expected: {stage}, Actual: {Stage}");
    }

    static bool StageHasRoundState(StageOfGame stage)
    {
        return stage == StageOfGame.InRoundPlayerChoice ||
            stage == StageOfGame.InRoundAfterHandUsed;
    }
}

public enum StageOfGame : byte
{
    Null,
    EnterShop,
    InShop,
    BeginRound,
    EndRound,
    InRoundPlayerChoice,
    InRoundAfterHandUsed,
}

public enum RoundType
{
    Null,
    SmallBlind,
    BigBlind,
    BossBlind,
    ShowdownBlind,
}
