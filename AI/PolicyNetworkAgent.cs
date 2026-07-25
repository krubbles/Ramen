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
        (GameStateTensors _, Tensor logProbs, Tensor _) = GetPolicyProbDist(temp, gameStates);
        Tensor probs = exp(logProbs);
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

    public (GameStateTensors gameStateTensors, Tensor logProbs, Tensor value) GetPolicyProbDist(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        using var scope = NewDisposeScope();
        using var profileScope = ProfileScope.New(nameof(GetPolicyProbDist));

        GameStateTensors gameStateTensors;
        using (ProfileScope.New("CreateGameStateTensors"))
        {
            gameStateTensors = CreateGameStateTensors(gameStates);
        }

        using var inferenceMode = inference_mode();
        Tensor logits;
        Tensor value;
        using (ProfileScope.New("GetPolicyLogits"))
        {
            (logits, value) = Network.GetPolicyValue(gameStateTensors);
        }

        const float IllegalMoveLogitEpsilon = 1e-2f;
        Tensor legalMoveMask = logits.ge(PolicyLogitMask.IllegalMoveLogit + IllegalMoveLogitEpsilon);
        Tensor scaledLogits = logits / MathF.Max(temp, 0.0001f);
        Tensor maskedLogits = where(
            legalMoveMask,
            scaledLogits,
            ones_like(scaledLogits) * PolicyLogitMask.IllegalMoveLogit);
        Tensor logProbs;
        using (ProfileScope.New("PolicyLogSoftmax"))
        {
            logProbs = TorchSharp.torch.nn.functional.log_softmax(maskedLogits, dim: 1);
        }
        logProbs.ToOuterScope();
        value.ToOuterScope();
        gameStateTensors.ToOuterScope();
        return (gameStateTensors, logProbs, value);
    }

    static GameStateTensors CreateGameStateTensors(ReadOnlySpan<GameState> gameStates)
    {
        GameState[] gameStateArray = gameStates.ToArray();
        GameStateEmbedder gameStateEmbedder = new(gameStateArray.Length);
        gameStateEmbedder.AddGameStates(gameStateArray);

        return gameStateEmbedder.ToTensors(includePlayHandScores: true);
    }
}
