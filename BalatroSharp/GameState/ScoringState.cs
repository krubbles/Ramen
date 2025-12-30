namespace BalatroAI;

/// <summary>
/// Holds all state used for scoring hands and tracking total chips earned in the current round.
/// </summary>
public class ScoringState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    /// <summary>
    /// Total number of chips earned by all hands played in the current round.
    /// </summary>
    public double CurrentRoundTotalChips;

    /// <summary>
    /// Current hand's chip value
    /// </summary>
    public double CurrentHandChips;

    /// <summary>
    /// Current hand's mult value
    /// </summary>
    public double CurrentHandMult;


    /// <summary>
    /// If a is being scored, this is the hand being played.
    /// </summary>

    public int CurrentScoringCardTriggerCount;
    
    // Settings
    public readonly int[] HandLevels = new int[]
    {
        0, // Null
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    };

    public ScoringState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
    }

    public void CloneFrom(in ScoringState other)
    {
        CurrentRoundTotalChips = other.CurrentRoundTotalChips;
        CurrentHandChips = other.CurrentHandChips;
        CurrentHandMult = other.CurrentHandMult;
        other.HandLevels.CopyTo(HandLevels, 0);
    }

    public void AddChipsToCurrentHand(int chips) => CurrentHandChips += chips;

    public void AddMultToCurrentHand(int mult) => CurrentHandMult += mult;

    public void ScoreCard(Card card)
    {        

        AddChipsToCurrentHand(GameData.BaseChipsForCardRank(card.Rank));
        GameState.JokerState.OnScoreCard(card);
    }

    public double ScoreActiveHand()
    {
        return ScoreHand(GameState.HandState.ActiveHand, GameState.HandState.ActiveHandPatterns);
    }

    public double ScoreHand(ReadOnlySpan<Card> hand, in HandPatternResults patterns)
    {
        (CurrentHandChips, CurrentHandMult) = _gameData.GetHandBaseScore(patterns.HandType, 1);
        int playedCardsMask = patterns.PlayedCardsMask;
        for (int i = 0; i < hand.Length; ++i)
        {
            if (((playedCardsMask >> i) & 1) == 1)
            {
                Card card = hand[i];
                CurrentScoringCardTriggerCount = 1;
                GameState.JokerState.OnBeginScoringCard(card);
                for (int j = 0; j < CurrentScoringCardTriggerCount; ++j) 
                {
                    ScoreCard(card);
                }
            }
        }

        GameState.JokerState.OnPlayHand();

        double score = CurrentHandChips * CurrentHandMult;
        CurrentRoundTotalChips += score;
        (CurrentHandChips, CurrentHandMult) = (0, 0);
        return score;
    }

    public void ResetCurrentRoundTotalChips()
    {
        CurrentRoundTotalChips = 0;
    }
}