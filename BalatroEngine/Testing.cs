namespace BalatroAI;
using static TorchSharp.torch;
using ConsoleApp;

public static class Testing
{
    public static float GetMaxOneShotScore(GameState gameState)
    {
        float maxScore = 0;
        Span<Card> hand = stackalloc Card[5];
        for (int playMask = 1; playMask < 256; ++playMask)
        {
            int handSize = 0;
            bool skip = false;
            for (int i = 0; i < 8; ++i)
            {
                if (((playMask >> i) & 1) != 0)
                {
                    if (handSize >= hand.Length)
                    {
                        skip = true;
                        break;
                    }
                    hand[handSize++] = gameState.HandState.Hand[i];
                }
            }
            if (skip)
                continue;
            gameState.PatternMatchingState.MatchHand(hand[..handSize], out gameState.HandState.ActiveHandPatterns);
            gameState.ScoringState.ResetCurrentRoundTotalChips();
            float score = (float)gameState.ScoringState.ScoreHand(hand[..handSize], gameState.HandState.ActiveHandPatterns);
            if (score > maxScore)
            {
                maxScore = score;
            }
        }
        return maxScore;
    }
}