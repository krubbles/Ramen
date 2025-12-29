namespace BalatroAI;

public class ScoringState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    public double CurrentRoundTotalChips;

    public double CurrentHandChips, CurrentHandMult;

    public readonly Card[] _currentPlayedHandBuffer = new Card[5];
    public int CurrentPlayedHandSize { get; private set; }
    public Span<Card> CurrentPlayedHand => _currentPlayedHandBuffer.AsSpan(0, CurrentPlayedHandSize);

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
        other.CurrentPlayedHand.CopyTo(_currentPlayedHandBuffer);
        CurrentPlayedHandSize = other.CurrentPlayedHandSize;
        other.HandLevels.CopyTo(HandLevels, 0);
    }

    public void AddChipsToCurrentHand(int chips) => CurrentHandChips += chips;

    public void AddMultToCurrentHand(int mult) => CurrentHandMult += mult;

    public void ScoreCard(Card card)
    {        

        AddChipsToCurrentHand(GameData.BaseChipsForCardRank(card.Rank));
        GameState.JokerState.OnScoreCard(card);
    }

    public double ScoreHand(ReadOnlySpan<Card> hand)
    {
        hand.CopyTo(_currentPlayedHandBuffer);
        CurrentPlayedHandSize = hand.Length;

        (CurrentHandChips, CurrentHandMult) = _gameData.GetHandBaseScore(GameState.HandState.ActiveHandPatternResults.HandType, 1);
        int playedCardsMask = GameState.HandState.ActiveHandPatternResults.PlayedCardsMask;
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

        GameState.JokerState.OnJokerTriggers();

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