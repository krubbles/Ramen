namespace Ramen.AI;

using Ramen.Game;
using System;
using System.Collections.Generic;

public static class Testing
{
    public struct GameDatabaseStatistics
    {
        public int TotalGames;
        public int PlayedStraightGames;
        public int PlayedFlushGames;
        public int PlayedFullHouseGames;
        public int PlayedTwoPairGames;
        public int DiscardSameSuitGames;
        public int DiscardRankRangeGames;

        public readonly float PlayedStraightPercent => TotalGames == 0 ? 0f : (float)PlayedStraightGames / TotalGames;
        public readonly float PlayedFlushPercent => TotalGames == 0 ? 0f : (float)PlayedFlushGames / TotalGames;
        public readonly float PlayedFullHousePercent => TotalGames == 0 ? 0f : (float)PlayedFullHouseGames / TotalGames;
        public readonly float PlayedTwoPairPercent => TotalGames == 0 ? 0f : (float)PlayedTwoPairGames / TotalGames;
        public readonly float DiscardSameSuitPercent => TotalGames == 0 ? 0f : (float)DiscardSameSuitGames / TotalGames;
        public readonly float DiscardRankRangePercent => TotalGames == 0 ? 0f : (float)DiscardRankRangeGames / TotalGames;

        public static GameDatabaseStatistics operator +(GameDatabaseStatistics a, GameDatabaseStatistics b)
        {
            return new GameDatabaseStatistics
            {
                TotalGames = a.TotalGames + b.TotalGames,
                PlayedStraightGames = a.PlayedStraightGames + b.PlayedStraightGames,
                PlayedFlushGames = a.PlayedFlushGames + b.PlayedFlushGames,
                PlayedFullHouseGames = a.PlayedFullHouseGames + b.PlayedFullHouseGames,
                PlayedTwoPairGames = a.PlayedTwoPairGames + b.PlayedTwoPairGames,
                DiscardSameSuitGames = a.DiscardSameSuitGames + b.DiscardSameSuitGames,
                DiscardRankRangeGames = a.DiscardRankRangeGames + b.DiscardRankRangeGames
            };
        }
    }

    public static GameDatabaseStatistics GetGameDatabaseStatistics(string databaseName)
    {
        GameDatabase database = new(databaseName, load: true);

        GameDatabaseStatistics stats = new();
        foreach (GameState gameState in database)
        {
            GameDatabaseStatistics gameStats = AnalyzeSingleGame(gameState);
            stats = stats + gameStats;  
        }

        return stats;
    }

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

    static GameDatabaseStatistics AnalyzeSingleGame(GameState gameState)
    {
        bool playedStraight = false;
        bool playedFlush = false;
        bool playedFullHouse = false;
        bool playedTwoPair = false;
        bool discardSameSuit = false;
        bool discardRankRange = false;

        Move[] moves = gameState.MoveState.MoveHistory.ToArray();
        gameState.MoveState.RevertToStep(0);
        for (int i = 0; i < moves.Length; ++i)
        {
            Move move = moves[i];
            move.Apply(gameState);

            if (move is UseHandMove useHandMove)
            {
                if (useHandMove.IsDiscard)
                {
                    if (!discardSameSuit && IsSameSuitAfterDiscard(gameState.HandState.Hand))
                        discardSameSuit = true;
                    if (!discardRankRange && IsRankRangeSmallAfterDiscard(gameState.HandState.Hand))
                        discardRankRange = true;
                }
                else
                {
                    HandType handType = gameState.HandState.ActiveHandPatterns.HandType;
                    if (!playedStraight && handType == HandType.Straight)
                        playedStraight = true;
                    if (!playedFlush && handType == HandType.Flush)
                        playedFlush = true;
                    if (!playedFullHouse && handType == HandType.FullHouse)
                        playedFullHouse = true;
                    if (!playedTwoPair && handType == HandType.TwoPair)
                        playedTwoPair = true; 
                }
            }
        }

        GameDatabaseStatistics result = new();
        result.TotalGames = 1;
        result.PlayedStraightGames = playedStraight ? 1 : 0;
        result.PlayedFlushGames = playedFlush ? 1 : 0;
        result.PlayedFullHouseGames = playedFullHouse ? 1 : 0;
        result.PlayedTwoPairGames = playedTwoPair ? 1 : 0;
        result.DiscardSameSuitGames = discardSameSuit ? 1 : 0;
        result.DiscardRankRangeGames = discardRankRange ? 1 : 0;
        return result;
    }

    static bool IsSameSuitAfterDiscard(ReadOnlySpan<Card> hand)
    {
        Suit suit = Suit.None;
        for (int i = 0; i < hand.Length; ++i)
        {
            Suit cardSuit = hand[i].Suit;
            if (cardSuit == Suit.All)
                continue;
            if (suit == Suit.None)
                suit = cardSuit;
            else if (suit != cardSuit)
                return false;
        }
        return true;
    }

    static bool IsRankRangeSmallAfterDiscard(ReadOnlySpan<Card> hand)
    {
        if (hand.Length == 0)
            return false;

        int minRank = int.MaxValue;
        int maxRank = int.MinValue;
        for (int i = 0; i < hand.Length; ++i)
        {
            int rank = hand[i].Rank;
            if (rank < minRank)
                minRank = rank;
            if (rank > maxRank)
                maxRank = rank;
        }

        return maxRank - minRank + 1 <= 4;
    }
}