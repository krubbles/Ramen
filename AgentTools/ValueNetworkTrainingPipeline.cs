namespace Ramen.AgentTools;

using System.IO;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public readonly record struct ValueNetworkTrainingMetrics(float MeanLoss, int TrainedStates);

public sealed class ValueNetworkTrainingPipeline : IDisposable
{
    readonly bool _ownsModel;
    readonly Module _trainableModule;
    readonly AdamW _optimizer;

    public readonly IValueNetwork Model;

    public ValueNetworkTrainingPipeline(
        float learningRate,
        IValueNetwork model = null,
        bool ownsModel = true)
    {
        Model = model ?? new ValueNetwork();
        Module trainableModule = Model as Module;
        if (trainableModule == null)
            throw new InvalidOperationException($"{nameof(ValueNetworkTrainingPipeline)} requires an {nameof(IValueNetwork)} that also inherits {nameof(Module)}.");

        _trainableModule = trainableModule;
        _ownsModel = ownsModel;

        _optimizer = optim.AdamW(
            parameters: _trainableModule.parameters(),
            lr: learningRate,
            weight_decay: 0f,
            beta1: 0.9f,
            beta2: 0.99f);
    }


    public void Dispose()
    {
        _optimizer.Dispose();
        if (_ownsModel)
            _trainableModule.Dispose();
    }


    public ValueNetworkTrainingMetrics TrainOnAllStates(GameDatabase gameDatabase, int epochCount, int batchSize)
    {
        ValueGameRecord[] gameRecords = BuildGameRecords(gameDatabase);
        return TrainOnAllStates(gameRecords: gameRecords, epochCount: epochCount, batchSize: batchSize);
    }


    public ValueNetworkTrainingMetrics TrainOnAllStatesFromFile(string gameDatabasePath, int epochCount, int batchSize)
    {
        ValueGameRecord[] gameRecords = BuildGameRecordsFromFile(gameDatabasePath);
        return TrainOnAllStates(gameRecords: gameRecords, epochCount: epochCount, batchSize: batchSize);
    }


    ValueNetworkTrainingMetrics TrainOnAllStates(ValueGameRecord[] gameRecords, int epochCount, int batchSize)
    {
        if (gameRecords.Length == 0 || epochCount <= 0)
            return new(MeanLoss: 0f, TrainedStates: 0);

        int effectiveBatchSize = Math.Max(batchSize, 1);
        int totalTrainedStates = 0;
        float totalLoss = 0f;
        int batchCount = 0;

        // Train over every stored state for the requested number of epochs.
        for (int epochIndex = 0; epochIndex < epochCount; ++epochIndex)
        {
            foreach (ValueGameRecord gameRecord in gameRecords)
            {
                int gamePositionCount = gameRecord.PositionCount;
                for (int batchStart = 0; batchStart < gamePositionCount; batchStart += effectiveBatchSize)
                {
                    int currentBatchSize = Math.Min(effectiveBatchSize, gamePositionCount - batchStart);
                    (GameState[] sampledStates, float[] targets) = CreateStateBatch(
                        gameRecord: gameRecord,
                        batchStart: batchStart,
                        batchSize: currentBatchSize);

                    using var scope = NewDisposeScope();

                    GameStateEmbedder gameStateEmbedder = new(currentBatchSize);
                    for (int stateIndex = 0; stateIndex < sampledStates.Length; ++stateIndex)
                        gameStateEmbedder.AddGameState(sampledStates[stateIndex]);

                    GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(ValueNetwork.EvalDevice);
                    Tensor targetsTensor = tensor(targets, dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice);

                    _optimizer.zero_grad();

                    Tensor predictedAdvantages = Model.GetAdvantages(gameStateTensors);
                    Tensor loss = functional.mse_loss(predictedAdvantages, targetsTensor);
                    loss.backward();
                    _optimizer.step();

                    totalLoss += loss.item<float>();
                    totalTrainedStates += currentBatchSize;
                    batchCount++;
                }
            }
        }

        return new(
            MeanLoss: totalLoss / Math.Max(1, batchCount),
            TrainedStates: totalTrainedStates);
    }


    static ValueGameRecord[] BuildGameRecords(GameDatabase gameDatabase)
    {
        // Serialize each completed game so random positions can be re-materialized later.
        List<ValueGameRecord> gameRecords = [];
        foreach (GameState gameState in gameDatabase)
        {
            using MemoryStream stream = new();
            gameState.Serialize(stream);
            gameRecords.Add(new(
                SerializedGame: stream.ToArray(),
                FinalReward: GetReward(gameState),
                MoveCount: gameState.MoveState.MoveHistory.Count));
        }

        return [.. gameRecords];
    }


    static ValueGameRecord[] BuildGameRecordsFromFile(string gameDatabasePath)
    {
        List<ValueGameRecord> gameRecords = [];
        using FileStream fileStream = new(gameDatabasePath, FileMode.Open, FileAccess.Read, FileShare.Read);

        while (fileStream.Position < fileStream.Length)
        {
            GameState gameState = new(GameData.Default);
            gameState.Deserialize(fileStream);
            using MemoryStream stream = new();
            gameState.Serialize(stream);
            gameRecords.Add(new(
                SerializedGame: stream.ToArray(),
                FinalReward: GetReward(gameState),
                MoveCount: gameState.MoveState.MoveHistory.Count));
        }

        return [.. gameRecords];
    }

    static (GameState[] sampledStates, float[] targets) CreateStateBatch(
        ValueGameRecord gameRecord,
        int batchStart,
        int batchSize)
    {
        GameState[] sampledStates = new GameState[batchSize];
        float[] targets = new float[batchSize];

        // Materialize a contiguous slice of positions from one game.
        for (int stateIndex = 0; stateIndex < batchSize; ++stateIndex)
        {
            int moveStep = batchStart + stateIndex;
            sampledStates[stateIndex] = MaterializePosition(gameRecord, moveStep);
            targets[stateIndex] = gameRecord.FinalReward;
        }

        return (sampledStates, targets);
    }


    static GameState MaterializePosition(ValueGameRecord gameRecord, int moveStep)
    {
        GameState gameState = new(GameData.Default);
        using MemoryStream stream = new(gameRecord.SerializedGame, writable: false);
        gameState.Deserialize(stream);
        gameState.MoveState.RevertToStep(moveStep);
        return gameState;
    }


    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalScore >= 300)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalScore / 1000f;
    }


    readonly record struct ValueGameRecord(byte[] SerializedGame, float FinalReward, int MoveCount)
    {
        public int PositionCount => MoveCount + 1;
    }
}
