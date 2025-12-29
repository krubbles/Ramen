namespace BalatroAI;

public partial class GameState // Round 
{
    public void StartRound()
    {
        DeckState.ResetAndShuffleDeck();
        HandState.ResetRemainingHandsAndDiscards();
        ScoringState.ResetCurrentRoundTotalChips();

        HandState.DrawToHandSize();
    }
}