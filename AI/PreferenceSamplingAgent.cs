namespace Ramen.AI;

using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public sealed class PreferenceSamplingAgent : IAgent, IDisposable
{
    readonly bool _ownsModel;

    public readonly PreferenceValueModel Model;
    public readonly FastRandom Random;

    public int MaxActiveBatchSize { get; set; } = 16;

    public float TopMoveProbabilityMassTarget { get; set; } = float.NaN;

    public int TopMoveProbabilityCount { get; set; } = 4;

    public PreferenceSamplingAgent(PreferenceValueModel model, bool ownsModel = false)
    {
        Model = model;
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
        // Gather all active states after advancing through automatic transitions.
        List<int> activeIndices = CollectActiveIndices(gameStates);
        if (activeIndices.Count == 0)
            return;

        // Evaluate active states in chunks so successor-state expansion stays bounded.
        for (int batchStart = 0; batchStart < activeIndices.Count; batchStart += MaxActiveBatchSize)
        {
            int batchSize = Math.Min(MaxActiveBatchSize, activeIndices.Count - batchStart);
            GameState[] activeStates = new GameState[batchSize];
            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
                activeStates[batchIndex] = gameStates[activeIndices[batchStart + batchIndex]];

            (UseHandMove[][] moveOptions, float[][] policies) = EvaluateMovePolicies(temp, activeStates);
            for (int stateIndex = 0; stateIndex < activeStates.Length; ++stateIndex)
            {
                if (moveOptions[stateIndex] == null || moveOptions[stateIndex].Length == 0)
                    continue;

                int chosenMoveIndex = AgentUtilities.SampleIndex(policies[stateIndex], Random);
                GameState gameState = activeStates[stateIndex];
                moveOptions[stateIndex][chosenMoveIndex].Apply(gameState);

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

            (_, float[][] policies) = EvaluateMovePolicies(temp, activeStates);
            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
                result[activeIndices[batchStart + batchIndex]] = policies[batchIndex];
        }

        return result;
    }


    List<int> CollectActiveIndices(ReadOnlySpan<GameState> gameStates)
    {
        List<int> activeIndices = new();
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            gameState.AdvanceToNextPlayerChoice();
            if (!IsGameDone(gameState))
                activeIndices.Add(stateIndex);
        }

        return activeIndices;
    }


    (UseHandMove[][] moveOptions, float[][] policies) EvaluateMovePolicies(float temp, ReadOnlySpan<GameState> gameStates)
    {
        UseHandMove[][] moveOptions = new UseHandMove[gameStates.Length][];
        float[][] policies = new float[gameStates.Length][];
        CandidateRange[] candidateRanges = new CandidateRange[gameStates.Length];

        int totalCandidateCount = CollectMoveOptions(gameStates, moveOptions, candidateRanges);
        if (totalCandidateCount == 0)
            return (moveOptions, policies);

        float[] flatLogits = EvaluateSuccessorLogits(gameStates, moveOptions, candidateRanges, totalCandidateCount);
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            CandidateRange candidateRange = candidateRanges[stateIndex];
            ReadOnlySpan<float> candidateLogits = flatLogits.AsSpan(candidateRange.Start, candidateRange.Count);
            float policyTemperature = temp;
            if (float.IsFinite(TopMoveProbabilityMassTarget))
            {
                policyTemperature = AgentUtilities.GetTemperatureForTargetTopProbabilityMass(
                    logits: candidateLogits,
                    topProbabilityCount: Math.Max(1, TopMoveProbabilityCount),
                    targetProbabilityMass: TopMoveProbabilityMassTarget);
            }

            float[] policy = AgentUtilities.SafeSoftmax(candidateLogits, policyTemperature);
            policies[stateIndex] = policy;
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


    float[] EvaluateSuccessorLogits(ReadOnlySpan<GameState> gameStates, UseHandMove[][] moveOptions, CandidateRange[] candidateRanges, int totalCandidateCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        GameStateEmbedder gameStateEmbedder = new(totalCandidateCount);

        // Enumerate every legal successor state exactly once.
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            UseHandMove[] stateMoveOptions = moveOptions[stateIndex];
            int moveHistoryStep = gameState.MoveState.MoveStep;

            for (int moveIndex = 0; moveIndex < stateMoveOptions.Length; ++moveIndex)
            {
                stateMoveOptions[moveIndex].Apply(gameState);
                gameState.AdvanceToNextPlayerChoice();
                gameStateEmbedder.AddGameState(gameState);
                gameState.MoveState.RevertToStep(moveHistoryStep);
            }
        }

        GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(PreferenceValueModel.EvalDevice);
        Tensor logits = Model.GetLogits(gameStateTensors).to(CPU);
        return logits.data<float>().ToArray();
    }


    static void AnnotatePolicy(GameState gameState, ReadOnlySpan<float> probabilities)
    {
        AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation(probabilities);
        annotation.Apply(gameState);
    }


    readonly record struct CandidateRange(int Start, int Count);
}
