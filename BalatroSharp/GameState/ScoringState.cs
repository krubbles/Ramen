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
    /// Current hand's chip value. Not persistent state.
    /// </summary>
    public double CurrentHandChips;

    /// <summary>
    /// Current hand's mult value. Not persistent state.
    /// </summary>
    public double CurrentHandMult;


    /// <summary>
    /// The number of triggers remaining on the scoring current card. 
    /// A joker can make a card retrigger N times by adding N to this value during the
    /// <see cref="Joker.OnBeginScoringCard"/> hook.
    /// <para>Not persistent state.</para>
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

    public override int GetHashCode()
    {
        int handLevelsHash = 0;
        for (int i = 0; i < HandLevels.Length; ++i)
        {
            handLevelsHash += HandLevels[i];
            handLevelsHash *= 449867969;
        }
        return 704315923 +
            handLevelsHash + 
            CurrentRoundTotalChips.GetHashCode() * 687577043;
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

    public double ScoreHand(ReadOnlySpan<Card> hand, in HandPatterns patterns)
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