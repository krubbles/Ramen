namespace BalatroAI;

public sealed partial class GameState
{
    public readonly GameData GameData;

    public readonly ScoringState ScoringState;
    public readonly DeckState DeckState;
    public readonly HandState HandState;
    public readonly JokerState JokerState;
    public readonly PatternMatchingState PatternMatchingState;

    public readonly FastRandom SeedGenerator;

    public GameState(GameData gameData)
    {
        GameData = gameData;
        SeedGenerator = new(GameData.RandomizeSeed ? FastRandom.SeededByClock().Next() : GameData.Seed);

        ScoringState = new(this);
        DeckState = new(this);
        HandState = new(this);
        JokerState = new(this);
        PatternMatchingState = new(this);

        GameData.InitStartingDeck(this);
    }

    public void CloneFrom(GameState other)
    {
        DeckState.CloneFrom(other.DeckState);
        ScoringState.CloneFrom(other.ScoringState);
        HandState.CloneFrom(other.HandState);
    }

    public void Reseed(int seed)
    {
        DeckState.Reseed(seed);
    }
}