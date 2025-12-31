namespace BalatroAI;

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
        Random = new(GameData.RandomizeSeed ? FastRandom.SeededByClock().Next() : GameData.Seed);

        ScoringState = new(this);
        DeckState = new(this);
        HandState = new(this);
        JokerState = new(this);
        PatternMatchingState = new(this);
        MoveState = new(this);

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
            case StageOfGame.InRoundRedrawing:
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
    protected override void Apply()
    {
        if (gameState.Stage != StageOfGame.BeginRound)
            throw new InvalidOperationException("Cannot start round, gameState is not in the BeginRound GameStage");

        gameState.Stage = StageOfGame.InRoundRedrawing;

        gameState.HandState.ResetRemainingHandsAndDiscards();
        gameState.ScoringState.ResetCurrentRoundTotalChips();
        gameState.DeckState.ResetDeck();

    }

    protected override void Revert()
    {
        gameState.Stage = StageOfGame.BeginRound;

        gameState.HandState.RemainingHands = 0;
        gameState.HandState.RemainingDiscards = 0;
    }
}


public enum StageOfGame
{
    None,
    EnterStore,
    InStore,
    BeginRound,
    InRoundPlayerChoice,
    InRoundRedrawing,
}
