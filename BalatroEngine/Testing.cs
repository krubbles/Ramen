namespace BalatroAI;
using static TorchSharp.torch;

public static class Testing
{
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
                aigs.MakeMoveStochastic(TrainingConfig.GoodPlayTemperature);
            }
            totalReward += aigs.GetCurrentReward();
        }
        return totalReward / sampleCount;
    }
}