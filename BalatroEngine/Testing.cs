namespace Ramen.AI;

using Ramen.Game;

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

    public static float GetAverageScore(GameEvalModel model, int samples = 1000, bool log = false)
    {
        float totalReward = 0;
        for (int i = 0; i < samples; ++i)
        {
            GameState gameState = new(new());
            RamenAgent agent = new(gameState, model);
            gameState.AdvanceToNextPlayerChoice();
            while (gameState.ScoringState.CurrentRoundTotalChips < 300 && gameState.HandState.RemainingHands > 0)
            {
                agent.MakeMoveStochastic(0.00001f);
            }
            totalReward += agent.GetCurrentReward();
            if (log)
            {
                Console.WriteLine(gameState.MoveState.GameToString());
                Console.WriteLine("Final Reward: " + agent.GetCurrentReward());
            }
        }
        return totalReward / samples;
    }
}