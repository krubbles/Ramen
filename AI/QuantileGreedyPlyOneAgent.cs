namespace Ramen.AI;

using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public sealed class QuantileGreedyPlyOneAgent : IAgent, IDisposable
{
    const int SuccessorEvaluationBatchSize = 256;

    readonly bool _ownsModel;

    public readonly QuantilePaddedSwiGLUValueNetwork Model;
    public readonly FastRandom Random;

    public int MaxActiveBatchSize { get; set; } = 16;

    public float Epsilon { get; set; }

    public QuantileGreedyPlyOneAgent(QuantilePaddedSwiGLUValueNetwork model, float epsilon, bool ownsModel = false)
    {
        Model = model;
        Epsilon = epsilon;
        _ownsModel = ownsModel;
        Random = FastRandom.SeededByClock();
    }


    public bool IsGameDone(GameState gameState)
    {
        return gameState.GameIsDone;
    }


    public void Dispose()
    {
        if (_ownsModel)
            Model.Dispose();
    }


    public void MakeMove(float temp, bool annotatePolicy, params ReadOnlySpan<GameState> gameStates)
    {
        List<int> activeIndices = CollectActiveIndices(gameStates);
        if (activeIndices.Count == 0)
            return;

        for (int batchStart = 0; batchStart < activeIndices.Count; batchStart += MaxActiveBatchSize)
        {
            int batchSize = Math.Min(MaxActiveBatchSize, activeIndices.Count - batchStart);
            GameState[] activeStates = new GameState[batchSize];
            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
                activeStates[batchIndex] = gameStates[activeIndices[batchStart + batchIndex]];

            (UseHandMove[][] moveOptions, float[][] policies) = EvaluateMovePolicies(activeStates);
            for (int stateIndex = 0; stateIndex < activeStates.Length; ++stateIndex)
            {
                UseHandMove[] stateMoveOptions = moveOptions[stateIndex];
                if (stateMoveOptions == null || stateMoveOptions.Length == 0)
                    continue;

                int chosenMoveIndex = AgentUtilities.SampleIndex(policies[stateIndex], Random);
                GameState gameState = activeStates[stateIndex];
                stateMoveOptions[chosenMoveIndex].Apply(gameState);

                if (annotatePolicy)
                    AnnotatePolicy(gameState, policies[stateIndex]);
            }
        }
    }


    public float[][] GetPolicy(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        float[][] result = new float[gameStates.Length][];
        List<int> activeIndices = CollectActiveIndices(gameStates);
        if (activeIndices.Count == 0)
            return result;

        for (int batchStart = 0; batchStart < activeIndices.Count; batchStart += MaxActiveBatchSize)
        {
            int batchSize = Math.Min(MaxActiveBatchSize, activeIndices.Count - batchStart);
            GameState[] activeStates = new GameState[batchSize];
            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
                activeStates[batchIndex] = gameStates[activeIndices[batchStart + batchIndex]];

            (_, float[][] policies) = EvaluateMovePolicies(activeStates);
            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
                result[activeIndices[batchStart + batchIndex]] = policies[batchIndex];
        }

        return result;
    }


    List<int> CollectActiveIndices(ReadOnlySpan<GameState> gameStates)
    {
        List<int> activeIndices = [];
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            gameState.AdvanceToNextPlayerChoice();
            if (!IsGameDone(gameState))
                activeIndices.Add(stateIndex);
        }

        return activeIndices;
    }


    (UseHandMove[][] moveOptions, float[][] policies) EvaluateMovePolicies(ReadOnlySpan<GameState> gameStates)
    {
        UseHandMove[][] moveOptions = new UseHandMove[gameStates.Length][];
        float[][] policies = new float[gameStates.Length][];
        CandidateRange[] candidateRanges = new CandidateRange[gameStates.Length];

        int totalCandidateCount = CollectMoveOptions(gameStates, moveOptions, candidateRanges);
        if (totalCandidateCount == 0)
            return (moveOptions, policies);

        float[] flatCandidateValues = EvaluateSuccessorValues(gameStates, moveOptions);
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            CandidateRange candidateRange = candidateRanges[stateIndex];
            ReadOnlySpan<float> candidateValues = flatCandidateValues.AsSpan(candidateRange.Start, candidateRange.Count);
            policies[stateIndex] = BuildGreedyEpsilonPolicy(candidateValues);
        }

        return (moveOptions, policies);
    }


    int CollectMoveOptions(ReadOnlySpan<GameState> gameStates, UseHandMove[][] moveOptions, CandidateRange[] candidateRanges)
    {
        int totalCandidateCount = 0;
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            Move[] legalMoves = gameStates[stateIndex].GetMoveOptions();
            UseHandMove[] typedMoves = new UseHandMove[legalMoves.Length];
            for (int moveIndex = 0; moveIndex < legalMoves.Length; ++moveIndex)
                typedMoves[moveIndex] = (UseHandMove)legalMoves[moveIndex];

            moveOptions[stateIndex] = typedMoves;
            candidateRanges[stateIndex] = new(totalCandidateCount, typedMoves.Length);
            totalCandidateCount += typedMoves.Length;
        }

        return totalCandidateCount;
    }


    float[] EvaluateSuccessorValues(ReadOnlySpan<GameState> gameStates, UseHandMove[][] moveOptions)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        int totalCandidateCount = 0;
        for (int stateIndex = 0; stateIndex < moveOptions.Length; ++stateIndex)
            totalCandidateCount += moveOptions[stateIndex].Length;

        float[] candidateValues = new float[totalCandidateCount];
        GameStateEmbedder gameStateEmbedder = new(SuccessorEvaluationBatchSize);
        List<int> pendingCandidateIndices = [];
        int candidateIndex = 0;
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            int moveHistoryStep = gameState.MoveState.MoveStep;
            UseHandMove[] stateMoveOptions = moveOptions[stateIndex];

            for (int moveIndex = 0; moveIndex < stateMoveOptions.Length; ++moveIndex)
            {
                stateMoveOptions[moveIndex].Apply(gameState);
                gameState.AdvanceToNextPlayerChoice();
                if (gameState.GameIsDone)
                    candidateValues[candidateIndex] = GetReward(gameState);
                else
                {
                    pendingCandidateIndices.Add(candidateIndex);
                    gameStateEmbedder.AddGameState(gameState);
                    if (pendingCandidateIndices.Count == SuccessorEvaluationBatchSize)
                    {
                        FlushPendingSuccessorValues(
                            gameStateEmbedder: gameStateEmbedder,
                            pendingCandidateIndices: pendingCandidateIndices,
                            candidateValues: candidateValues);
                        gameStateEmbedder = new(SuccessorEvaluationBatchSize);
                    }
                }
                gameState.MoveState.RevertToStep(moveHistoryStep);
                candidateIndex++;
            }
        }

        FlushPendingSuccessorValues(
            gameStateEmbedder: gameStateEmbedder,
            pendingCandidateIndices: pendingCandidateIndices,
            candidateValues: candidateValues);

        return candidateValues;
    }


    void FlushPendingSuccessorValues(GameStateEmbedder gameStateEmbedder, List<int> pendingCandidateIndices, float[] candidateValues)
    {
        if (pendingCandidateIndices.Count == 0)
            return;

        GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(ValueNetwork.EvalDevice);
        Tensor values = Model.GetAdvantages(gameStateTensors).to(CPU);
        float[] embeddedValues = values.data<float>().ToArray();
        for (int valueIndex = 0; valueIndex < pendingCandidateIndices.Count; ++valueIndex)
            candidateValues[pendingCandidateIndices[valueIndex]] = embeddedValues[valueIndex];

        pendingCandidateIndices.Clear();
    }


    float[] BuildGreedyEpsilonPolicy(ReadOnlySpan<float> candidateValues)
    {
        float[] policy = new float[candidateValues.Length];
        if (candidateValues.Length == 0)
            return policy;

        float bestValue = candidateValues[0];
        int bestCount = 1;
        for (int candidateIndex = 1; candidateIndex < candidateValues.Length; ++candidateIndex)
        {
            if (candidateValues[candidateIndex] > bestValue)
            {
                bestValue = candidateValues[candidateIndex];
                bestCount = 1;
            }
            else if (candidateValues[candidateIndex] == bestValue)
            {
                bestCount++;
            }
        }

        float uniformMass = Epsilon / candidateValues.Length;
        float greedyMass = (1f - Epsilon) / bestCount;
        for (int candidateIndex = 0; candidateIndex < candidateValues.Length; ++candidateIndex)
            policy[candidateIndex] = uniformMass;

        for (int candidateIndex = 0; candidateIndex < candidateValues.Length; ++candidateIndex)
        {
            if (candidateValues[candidateIndex] == bestValue)
                policy[candidateIndex] += greedyMass;
        }

        return policy;
    }


    static void AnnotatePolicy(GameState gameState, ReadOnlySpan<float> probabilities)
    {
        AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation(probabilities);
        annotation.Apply(gameState);
    }


    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalScore >= 300f)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalScore / 1000f;
    }


    readonly record struct CandidateRange(int Start, int Count);
}
