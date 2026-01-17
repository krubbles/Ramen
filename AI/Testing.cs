namespace Ramen.AI;

using Ramen.Game;
using System.Collections.Generic;

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

    public static float GetOnshotThresholdProb(GameState gameState, int threshold, int samples)
    {
        FastRandom random = FastRandom.SeededByClock();
        float amount = 0;
        int step = gameState.MoveState.MoveStep;
        for (int i = 0; i < samples; ++i)
        {
            gameState.Random.SetState((ulong)random.Next());
            gameState.AdvanceToNextPlayerChoice();
            if (GetMaxOneShotScore(gameState) >= threshold)
                amount++;
            gameState.MoveState.RevertToStep(step);
        }
        return amount / samples;
    }

    public static (Move move, float prob) GetBestDiscard(GameState gameState, int threshold)
    {
        gameState.AdvanceToNextPlayerChoice();
        Move[] moves = gameState.GetMoveOptions().ToArray();
        float[] probs = new float[moves.Length];
        for (int i = 0; i < moves.Length; ++i)
        {
            Move move = moves[i];
            move.Apply(gameState);
            probs[i] = GetOnshotThresholdProb(gameState, threshold - (int)gameState.ScoringState.CurrentRoundTotalChips, 500);
            move.Revert(gameState);
        }
        Array.Sort(probs, moves);
        for (int i = probs.Length - 1; i > probs.Length - 20; --i)
        {
            Console.WriteLine($"{probs[i]}: {moves[i]}");
        }
        return (moves[0], probs[0]);
    }

    public static float GetAverageScore(PolicyModel model, int samples = 1000, float temp = 0.0001f, bool log = false)
    {
        float totalReward = 0;
        for (int i = 0; i < samples; ++i)
        {
            GameState gameState = new(new());
            RamenAgent agent = new(gameState, model);
            gameState.AdvanceToNextPlayerChoice();
            while (gameState.ScoringState.CurrentRoundTotalChips < 300 && gameState.HandState.RemainingHands > 0)
            {
                agent.MakeMove(temp);
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

    public static (float mean, double ciLower, double ciUpper, double stdError) GetScoreStatistics(PolicyModel model, int samples = 1000, float temp = 0.0001f, bool log = false)
    {
        List<float> scores = new();
        float totalReward = 0;

        for (int i = 0; i < samples; ++i)
        {
            GameState gameState = new(new());
            RamenAgent agent = new(gameState, model);
            gameState.AdvanceToNextPlayerChoice();
            while (gameState.ScoringState.CurrentRoundTotalChips < 300 && gameState.HandState.RemainingHands > 0)
            {
                agent.MakeMove(temp);
            }
            float reward = agent.GetCurrentReward();
            scores.Add(reward);
            totalReward += reward;
            if (log)
            {
                Console.WriteLine(gameState.MoveState.GameToString());
                Console.WriteLine("Final Reward: " + agent.GetCurrentReward());
            }
        }

        float mean = totalReward / samples;
        double sumSquaredDiffs = 0;
        foreach (float score in scores)
        {
            sumSquaredDiffs += Math.Pow(score - mean, 2);
        }
        double variance = sumSquaredDiffs / (samples - 1);
        double stdDev = Math.Sqrt(variance);
        double stdError = stdDev / Math.Sqrt(samples);
        double z = 1.96; // 95% confidence interval
        double marginOfError = z * stdError;

        return (mean, mean - marginOfError, mean + marginOfError, stdError);
    }
}