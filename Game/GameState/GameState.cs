using System.IO.Compression;

namespace Ramen.Game;

public sealed class GameState
{
    public readonly GameData GameData;

    public readonly ScoringState ScoringState;
    public readonly DeckState DeckState;
    public readonly HandState HandState;
    public readonly JokerState JokerState;
    public readonly PatternMatchingState PatternMatchingState;
    public readonly MoveState MoveState;

    public readonly FastRandom Random;
    public StageOfGame Stage { get; internal set; }

    readonly List<Move> _currentLegalMovesBuffer = new();

    public GameState(GameData gameData)
    {
        GameData = gameData;
        Random = new(0);

        ScoringState = new(this);
        DeckState = new(this);
        HandState = new(this);
        JokerState = new(this);
        PatternMatchingState = new(this);
        MoveState = new(this);

        int seed = gameData.RandomizeSeed ? FastRandom.SeededByClock().Next() : gameData.Seed;
        ReseedMove seeder = new(FastRandom.SeedToState((ulong)seed));
        seeder.Apply(this);

        GameData.InitStartingDeck(this);

        Stage = StageOfGame.BeginRound;
    }

    public override int GetHashCode()
    {
        return 562877087 ^
            (int)Stage * 301499677 ^
            ScoringState.GetHashCode() ^
            HandState.GetHashCode() ^
            DeckState.GetHashCode()
            ;
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

    public bool GameIsDone => HandState.RemainingHands == 0 || ScoringState.CurrentRoundTotalChips >= 300;

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

    /// <summary>
    /// Returns a list of legal moves.
    /// If there is 1, then there is an automatic state change that must happen. 
    /// If there are multiple, the player/agent has a choice to make. 
    /// <
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
        int versionNumber = 1;

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
}

public enum StageOfGame : byte
{
    None,
    EnterStore,
    InStore,
    BeginRound,
    InRoundPlayerChoice,
    InRoundAfterHandUsed,
}
