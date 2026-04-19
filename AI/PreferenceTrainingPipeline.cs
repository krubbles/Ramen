namespace Ramen.AI;

using System.IO;
using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public sealed class PreferenceGameRecord
{
    public readonly byte[] SerializedGame;
    public readonly float FinalReward;
    public readonly int MoveCount;

    public int PositionCount => MoveCount + 1;

    public PreferenceGameRecord(byte[] serializedGame, float finalReward, int moveCount)
    {
        SerializedGame = serializedGame;
        FinalReward = finalReward;
        MoveCount = moveCount;
    }
}

public readonly record struct PreferenceTrainingMetrics(float MeanLoss, int TrainedPairs);

public sealed class PreferenceTrainingPipeline : IDisposable
{
    readonly bool _ownsModel;
    readonly AdamW _optimizer;
    readonly float _gameplaySamplingTemp;

    public readonly PreferenceValueModel Model;
    public readonly PreferenceSamplingAgent Agent;
    public readonly FastRandom Random;

    public PreferenceTrainingPipeline(
        float learningRate,
        float gameplaySamplingTemp = 1f,
        PreferenceValueModel model = null,
        bool ownsModel = true)
    {
        Model = model ?? new();
        Agent = new(Model, ownsModel: false);
        Random = FastRandom.SeededByClock();
        _ownsModel = ownsModel;
        _gameplaySamplingTemp = gameplaySamplingTemp;

        _optimizer = optim.AdamW(
            parameters: Model.parameters(),
            lr: learningRate,
            weight_decay: 0f,
            beta1: 0.9f,
            beta2: 0.99f);
    }


    public void Dispose()
    {
        _optimizer.Dispose();
        Agent.Dispose();
        if (_ownsModel)
            Model.Dispose();
    }


    public PreferenceGameRecord[] PlayTrainingGames(int gameCount)
    {
        GameState[] games = PlayGames(gameCount, annotatePolicy: false);
        PreferenceGameRecord[] records = new PreferenceGameRecord[games.Length];
        for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
        {
            GameState gameState = games[gameIndex];
            using MemoryStream stream = new();
            gameState.Serialize(stream);
            records[gameIndex] = new(
                serializedGame: stream.ToArray(),
                finalReward: GetReward(gameState),
                moveCount: gameState.MoveState.MoveHistory.Count);
        }

        return records;
    }


    public GameState[] PlayAnalysisGames(int gameCount)
    {
        return PlayGames(gameCount, annotatePolicy: true);
    }


    public PreferenceTrainingMetrics TrainOnRandomPairs(IReadOnlyList<PreferenceGameRecord> gameRecords, int pairCount, int batchSize)
    {
        if (gameRecords.Count == 0 || pairCount <= 0)
            return new(MeanLoss: 0f, TrainedPairs: 0);

        int[] cumulativePositionCounts = BuildCumulativePositionCounts(gameRecords);
        int totalPositionCount = cumulativePositionCounts[^1];
        float totalLoss = 0f;
        int batchCount = 0;

        for (int batchStart = 0; batchStart < pairCount; batchStart += batchSize)
        {
            int currentBatchSize = Math.Min(batchSize, pairCount - batchStart);
            (GameState[] leftStates, GameState[] rightStates, float[] targets) = SamplePairBatch(gameRecords, cumulativePositionCounts, totalPositionCount, currentBatchSize);

            using var scope = NewDisposeScope();

            GameStateEmbedder leftGameStateEmbedder = new(leftStates.Length);
            GameStateEmbedder rightGameStateEmbedder = new(rightStates.Length);
            for (int stateIndex = 0; stateIndex < leftStates.Length; ++stateIndex)
            {
                leftGameStateEmbedder.AddGameState(leftStates[stateIndex]);
                rightGameStateEmbedder.AddGameState(rightStates[stateIndex]);
            }

            GameStateTensors leftTensors = leftGameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
            GameStateTensors rightTensors = rightGameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
            Tensor targetsTensor = tensor(targets, dtype: ScalarType.Float32, device: PreferenceValueModel.EvalDevice);

            _optimizer.zero_grad();

            Tensor leftLogits = Model.GetLogits(leftTensors);
            Tensor rightLogits = Model.GetLogits(rightTensors);
            Tensor loss = BradleyTerryLoss(leftLogits, rightLogits, targetsTensor);
            loss.backward();
            _optimizer.step();

            totalLoss += loss.item<float>();
            batchCount++;
        }

        return new(
            MeanLoss: totalLoss / Math.Max(1, batchCount),
            TrainedPairs: pairCount);
    }


    public void Save(string filePath)
    {
        Model.Save(filePath);
    }


    GameState[] PlayGames(int gameCount, bool annotatePolicy)
    {
        GameState[] games = new GameState[Math.Max(0, gameCount)];
        for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
            games[gameIndex] = new(GameData.Default);

        while (true)
        {
            bool allGamesDone = true;
            for (int gameIndex = 0; gameIndex < games.Length; ++gameIndex)
            {
                if (!Agent.IsGameDone(games[gameIndex]))
                {
                    allGamesDone = false;
                    break;
                }
            }

            if (allGamesDone)
                break;

            Agent.MakeMove(temp: _gameplaySamplingTemp, annotatePolicy: annotatePolicy, games);
        }

        return games;
    }


    static int[] BuildCumulativePositionCounts(IReadOnlyList<PreferenceGameRecord> gameRecords)
    {
        int[] cumulativeCounts = new int[gameRecords.Count];
        int runningTotal = 0;
        for (int gameIndex = 0; gameIndex < gameRecords.Count; ++gameIndex)
        {
            runningTotal += gameRecords[gameIndex].PositionCount;
            cumulativeCounts[gameIndex] = runningTotal;
        }

        return cumulativeCounts;
    }


    (GameState[] leftStates, GameState[] rightStates, float[] targets) SamplePairBatch(IReadOnlyList<PreferenceGameRecord> gameRecords, int[] cumulativePositionCounts, int totalPositionCount, int batchSize)
    {
        GameState[] leftStates = new GameState[batchSize];
        GameState[] rightStates = new GameState[batchSize];
        float[] targets = new float[batchSize];

        for (int pairIndex = 0; pairIndex < batchSize; ++pairIndex)
        {
            (PreferenceGameRecord leftRecord, int leftMoveStep) = SamplePosition(gameRecords, cumulativePositionCounts, totalPositionCount);
            (PreferenceGameRecord rightRecord, int rightMoveStep) = SamplePosition(gameRecords, cumulativePositionCounts, totalPositionCount);

            leftStates[pairIndex] = MaterializePosition(leftRecord, leftMoveStep);
            rightStates[pairIndex] = MaterializePosition(rightRecord, rightMoveStep);
            targets[pairIndex] = GetTarget(leftRecord.FinalReward, rightRecord.FinalReward);
        }

        return (leftStates, rightStates, targets);
    }


    (PreferenceGameRecord record, int moveStep) SamplePosition(IReadOnlyList<PreferenceGameRecord> gameRecords, int[] cumulativePositionCounts, int totalPositionCount)
    {
        int flatPositionIndex = Random.Next(totalPositionCount);
        int gameIndex = Array.BinarySearch(cumulativePositionCounts, flatPositionIndex + 1);
        if (gameIndex < 0)
            gameIndex = ~gameIndex;

        int previousCumulativeCount = gameIndex == 0 ? 0 : cumulativePositionCounts[gameIndex - 1];
        int moveStep = flatPositionIndex - previousCumulativeCount;
        return (gameRecords[gameIndex], moveStep);
    }


    static GameState MaterializePosition(PreferenceGameRecord gameRecord, int moveStep)
    {
        GameState gameState = new(GameData.Default);
        using MemoryStream stream = new(gameRecord.SerializedGame, writable: false);
        gameState.Deserialize(stream);
        gameState.MoveState.RevertToStep(moveStep);
        return gameState;
    }


    static Tensor BradleyTerryLoss(Tensor leftLogits, Tensor rightLogits, Tensor targets)
    {
        Tensor pairLogits = leftLogits - rightLogits;
        Tensor logProbLeft = functional.logsigmoid(pairLogits);
        Tensor logProbRight = functional.logsigmoid(-pairLogits);
        return -(targets * logProbLeft + (1f - targets) * logProbRight).mean();
    }


    static float GetTarget(float leftReward, float rightReward)
    {
        if (leftReward > rightReward)
            return 1f;
        if (leftReward < rightReward)
            return 0f;
        return 0.5f;
    }


    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalChips >= 300)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalChips / 1000f;
    }
}
