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
            gameState.PatternMatchingState.MatchHand(hand[..handSize], out gameState.HandState.ActiveHandPatternResults);
            gameState.ScoringState.ResetCurrentRoundTotalChips();
            float score = (float)gameState.ScoringState.ScoreHand(hand[..handSize]);
            if (score > maxScore)
            {
                maxScore = score;
            }
        }
        return maxScore;
    }

    public static float GetAverageReward(AIModels models, int sampleCount)
    {
        FastRandom random = FastRandom.SeededByClock();
        float totalReward = 0f;
        for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
        {
            GameData gameData = new();
            gameData.Seed = random.Next();
            GameState gameState = new(gameData);
            gameState.StartRound();
            AIGameState aigs = new(gameState, models);
            while (aigs.GameState.HandState.RemainingHands >= 4)
            {
                aigs.MakeMoveStochastic(1f);
            }
            totalReward += aigs.GetCurrentReward();
        }
        return totalReward / sampleCount;
    }

    public static void ShowExpectedReward(AIModels models, int sampleCount)
    {
        Console.WriteLine();
        Console.WriteLine("--- Example Expected Rewards ---");
        FastRandom random = FastRandom.SeededByClock();
        using (no_grad())
        {
            for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
            {
                GameData gameData = new();
                GameState gameState = new(gameData);
                gameState.StartRound();
                AIGameState aigs = new(gameState, models);
                while (aigs.GameState.HandState.RemainingHands >= 4)
                {
                    Console.WriteLine($"Hand: {gameState.HandToString()}");
                    Console.WriteLine($"To Use Count: {aigs.ToUseCount}");
                    Console.WriteLine($"Played Indices: {LoggingUtility.FormatArray(aigs.ToUseIndices[0..aigs.ToUseCount])}");
                    float expectedReward = models.GetExpectedReward(aigs.GameStateTensors.Hand, aigs.GameStateTensors.OtherState, aigs.InUseMaskTensor).item<float>();
                    Console.WriteLine($"Predicted Final Reward: {expectedReward}");

                    aigs.MakeMoveStochastic(TrainingConfig.GoodPlayTemp);
                }
            }
        }
        Console.WriteLine("-------------------------------");
        Console.WriteLine();
    }
}