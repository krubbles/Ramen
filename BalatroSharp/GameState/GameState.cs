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
            DeckState.GetHashCode();
    }

    public void CloneFrom(GameState other)
    {
        DeckState.CloneFrom(other.DeckState);
        ScoringState.CloneFrom(other.ScoringState);
        HandState.CloneFrom(other.HandState);
        MoveState.CloneFrom(other.MoveState);
    }

    public List<Move> GetMoveOptions()
    {
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
        return _currentLegalMovesBuffer;
    }

    public void AdvanceToNextPlayerChoice()
    {
        while (GetMoveOptions().Count == 1)
        {
            _currentLegalMovesBuffer[0].Apply(this);
        }
    }

    public void StartRound()
    {
        DeckState.ResetDeck();
        HandState.ResetRemainingHandsAndDiscards();
        ScoringState.ResetCurrentRoundTotalChips();
    }

    public void Serialize(Stream stream)
    {
        int versionNumber = 1;
        BufferedStream bufferedStream = new(stream);
        GameStateSerializer serializer = new(this, bufferedStream);
        serializer.Stream.WriteStartTag("GS");
        serializer.Stream.WriteStruct<int>(versionNumber);
        MoveState.Serialize(serializer);
        serializer.Stream.WriteEndTag("GS");
        bufferedStream.Flush();
    }

    public void Deserialize(Stream stream)
    {
        BufferedStream bufferedStream = new(stream);
        GameStateSerializer serializer = new(this, bufferedStream);
        serializer.Stream.ReadStartTag("GS");
        int versionNumber = serializer.Stream.ReadStruct<int>();
        MoveState.Deserialize(serializer, versionNumber);
        serializer.Stream.ReadEndTag("GS");
        bufferedStream.Flush();
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


public enum StageOfGame
{
    None,
    EnterStore,
    InStore,
    BeginRound,
    InRoundPlayerChoice,
    InRoundAfterHandUsed,
}
