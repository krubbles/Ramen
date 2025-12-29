namespace BalatroAI;
using static TorchSharp.torch;
using ConsoleApp;

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

    public static void ShowExpectedReward(AIModels models, int sampleCount)
    {
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
                    aigs.MakeMoveStochastic(TrainingConfig.GoodPlayTemperature);
                }
                Console.WriteLine($"Hand: {gameState.HandToString()}");
                Console.WriteLine($"Played Indices: {LoggingUtility.FormatArray(aigs.ToUseIndices[0..aigs.ToUseCount])}");

                float expectedReward = models.GetExpectedReward(aigs.HandTensor, aigs.GameStateTensors.OtherState, aigs.InUseMaskTensor).item<float>();
                Console.WriteLine($"Predicted Final Reward: {expectedReward}");

            }
        }
    }
}