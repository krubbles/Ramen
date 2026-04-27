namespace Ramen.Game;

/// <summary>
/// Holds all state used for scoring hands and tracking total chips earned in the current round.
/// </summary>
public class ScoringState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    double _currentRoundTotalScore;

    /// <summary>
    /// Total number of chips earned by all hands played in the current round.
    /// </summary>
    public double CurrentRoundTotalScore
    {
        get => _currentRoundTotalScore;
        internal set
        {
            _currentRoundTotalScore = value;
            GameState.MoveState.ScheduleCallback(OnCurrentRoundTotalChipsChanged);
        }
    }

    public double CurrentRoundThresholdScore
    {
        get => _gameData.GetThresholdForRound(GameState.Round);
    }
    
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
    
    readonly int[] _handLevels =
    [
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
    ];

    /// <summary>
    /// Returns the current hand level for a given hand type. 
    /// </summary>
    public int GetHandLevel(HandType handType) => _handLevels[(int)handType];

    public Action OnCurrentRoundTotalChipsChanged;

    public ScoringState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
    }

    public override int GetHashCode()
    {
        int handLevelsHash = 0;
        for (int i = 0; i < _handLevels.Length; ++i)
        {
            handLevelsHash += _handLevels[i];
            handLevelsHash *= 449867969;
        }
        return 704315923 +
            handLevelsHash + 
            CurrentRoundTotalScore.GetHashCode() * 687577043;
    }

    internal void CloneFrom(in ScoringState other)
    {
        CurrentRoundTotalScore = other.CurrentRoundTotalScore;
        CurrentHandChips = other.CurrentHandChips;
        CurrentHandMult = other.CurrentHandMult;
        other._handLevels.CopyTo(_handLevels, 0);
    }

    internal void ResetCurrentRoundTotalChips()
    {
        CurrentRoundTotalScore = 0;
    }

    internal void AddChipsToCurrentHand(int chips) => CurrentHandChips += chips;

    internal void AddMultToCurrentHand(int mult) => CurrentHandMult += mult;

    internal void ScoreCard(Card card)
    {        
        AddChipsToCurrentHand(GameData.BaseChipsForCardRank(card.Rank));
        GameState.JokerState.OnScoreCard(card);
    }

    internal double ScoreActiveHand()
    {
        return ScoreHand(GameState.HandState.ActiveHand, GameState.HandState.ActiveHandPatterns);
    }

    internal double ScoreHand(ReadOnlySpan<Card> hand, in HandPatterns patterns)
    {
        int handLevel = GetHandLevel(patterns.HandType);
        (CurrentHandChips, CurrentHandMult) = _gameData.GetHandBaseScore(patterns.HandType, handLevel);
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
        CurrentRoundTotalScore += score;
        (CurrentHandChips, CurrentHandMult) = (0, 0);
        return score;
    }
}
