namespace Ramen.AI;

public sealed class PolicyNetworkAgent : IAgent, IDisposable
{
    readonly bool _ownsNetwork;

    public readonly IPolicyNetwork Network;

    public PolicyNetworkAgent(IPolicyNetwork network, bool ownsNetwork = false)
    {
        Network = network;
        _ownsNetwork = ownsNetwork;
    }

    public bool IsGameDone(GameState gameState) => gameState.GameIsDone;

    public void Dispose()
    {
        if (_ownsNetwork && Network is IDisposable disposableNetwork)
            disposableNetwork.Dispose();
    }

    public float[][] GetPolicy(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        if (gameStates.Length == 0)
            return [];

        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var profileScope = ProfileScope.New(nameof(GetPolicy));

        // to avoid many different sized reserved MPS buffers, we process in full batches of 64,
        // then remainder batches of 8, then a final batch of size [1, 8]
        const int FullBatchSize = 64;
        const int RemainderBatchSize = 8;
        float[][] results = new float[gameStates.Length][];
        int batchStart = 0;
        for (; batchStart + FullBatchSize <= gameStates.Length; batchStart += FullBatchSize)
        {
            ProcessPolicyBatch(temp, gameStates.Slice(batchStart, FullBatchSize), results, batchStart);
        }
        for (; batchStart < gameStates.Length; batchStart += RemainderBatchSize)
        {
            int batchSize = Math.Min(RemainderBatchSize, gameStates.Length - batchStart);
            ProcessPolicyBatch(temp, gameStates.Slice(batchStart, batchSize), results, batchStart);
        }

        return results;
    }

    void ProcessPolicyBatch(float temp, ReadOnlySpan<GameState> gameStates, float[][] results, int resultStartIndex)
    {
        (GameStateTensors _, Tensor probs, Tensor _) = GetPolicyProbDist(temp, gameStates);
        probs = probs.to(CPU);

        int batchSize = (int)probs.size(0);
        float[] flat = probs.data<float>().ToArray();

        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            if (IsGameDone(gameStates[batchIndex]))
                continue;

            int moveCount = (int)probs.size(dim: 1);
            float[] row = new float[moveCount];
            Array.Copy(flat, batchIndex * moveCount, row, 0, moveCount);
            results[resultStartIndex + batchIndex] = row;
        }
    }

    public (GameStateTensors gameStateTensors, Tensor probs, Tensor value) GetPolicyProbDist(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var profileScope = ProfileScope.New(nameof(GetPolicyProbDist));

        GameStateTensors gameStateTensors = CreateGameStateTensors(gameStates);

        Profiling.Enter("GetPolicyLogits");
        (Tensor logits, Tensor value) = Network.GetPolicyLogitsAndValue(gameStateTensors);
        Profiling.Exit("GetPolicyLogits");

        int moveCount = (int)logits.size(1);
        Tensor illegalMoveMask = BuildIllegalMoveMask(gameStates, moveCount, logits.device);
        logits += illegalMoveMask;

        Tensor probs = (logits / MathF.Max(temp, 0.0001f)).softmax(1);
        probs.ToOuterScope();
        value.ToOuterScope();
        gameStateTensors.ToOuterScope();
        return (gameStateTensors, probs, value);
    }

    static GameStateTensors CreateGameStateTensors(ReadOnlySpan<GameState> gameStates)
    {
        using var profileScope = ProfileScope.New(nameof(CreateGameStateTensors));

        GameStateEmbedder gameStateEmbedder = new(gameStates.Length);
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
            gameStateEmbedder.AddGameState(gameStates[stateIndex]);

        return gameStateEmbedder.ToTensors(includePlayHandScores: true);
    }

    static Tensor BuildIllegalMoveMask(ReadOnlySpan<GameState> gameStates, int moveCount, Device device)
    {
        using var scope = NewDisposeScope();
        using var profileScope = ProfileScope.New(nameof(BuildIllegalMoveMask));

        float[,] mask = new float[gameStates.Length, 2];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            if (gameStates[stateIndex].HandState.RemainingHands == 0)
                mask[stateIndex, 0] = -1e6f;
            if (gameStates[stateIndex].HandState.RemainingDiscards == 0)
                mask[stateIndex, 1] = -1e6f;
        }

        Tensor actionMask = tensor(mask, device: CPU).to(device);
        Tensor result = actionMask.repeat([1, moveCount / 2]);
        result.MoveToOuterDisposeScope();
        return result;
    }
}
