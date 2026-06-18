namespace Ramen.AI;

/// <summary>
/// Extension methods for RamenAgent that provide training-related functionality.
/// </summary>
public static class AgentTrainingExtensions
{
    /// <summary>
    /// Makes moves based on the policy model's predicted probability distribution for a batch of game states.
    /// Also generates a PPO training sample for the chosen move.
    /// </summary>
    public static PolicyTrainingSample[] MakeMoveAndTrainingSample(this PolicyNetworkAgent agent, GameState[] gameStates, bool useSampledSoftmax, int sampleCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var p_funcScope = ProfileScope.New(nameof(MakeMoveAndTrainingSample));

        PolicyTrainingSample[] outputSamples = new PolicyTrainingSample[gameStates.Length];
        if (gameStates.Length == 0)
            return outputSamples;

        for (int i = 0; i < gameStates.Length; ++i)
            gameStates[i].AdvanceToNextPlayerChoice();

        // get a batch of policy probabilities for each state as a single batched tensor
        (GameStateTensors gameStateTensors, Tensor logProbs, Tensor value) = agent.GetPolicyProbDist(temp: 1f, gameStates); // returned tensors are on the gpu.
        Tensor probs = exp(logProbs);

        int moveCount = (int)probs.size(1);
        int minSampleCount = useSampledSoftmax ? Math.Min(2, moveCount) : 1;
        int clampedSampleCount = useSampledSoftmax ? Math.Clamp(sampleCount, minSampleCount, moveCount) : 1;

        // Index 0 is always the chosen move. Remaining indices are sampled negatives for sampled softmax.
        Tensor sampleIndices = multinomial(probs, clampedSampleCount, replacement: false);
        Tensor chosenMoveIndices = sampleIndices.select(dim: 1, index: 0);
        Tensor sampledProbs = probs.gather(dim: 1, sampleIndices);
        Tensor chosenProbs = sampledProbs.select(dim: 1, index: 0);
        Tensor chosenLogProbs = logProbs.gather(dim: 1, chosenMoveIndices.unsqueeze(1)).select(dim: 1, index: 0);

        // Read back the chosen move index and log-prob, so the move can be made and nl prob can be stored in the sample.
        Profiling.Enter("ChosenMoveReadback");
        long[] chosenMoveIndicesManaged = [.. chosenMoveIndices.to(CPU).data<long>()];
        float[] chosenLogProbsManaged = [.. chosenLogProbs.to(CPU).data<float>()];
        float[] policyData = [.. probs.to(CPU).data<float>()];
        Profiling.Exit("ChosenMoveReadback");

        Profiling.Enter("BuildSamples");
        // Sample and apply one move per state while creating one sample per state.
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            // Clone per-sample slices so each sample owns compact tensors instead of views to full step batches.
            Tensor sampleSamplingProb = sampledProbs[stateIndex..(stateIndex + 1)].clone();
            Tensor sampleSamplingLogProb = chosenLogProbs[stateIndex..(stateIndex + 1)].unsqueeze(1).clone();
            GameStateTensors sampleStateTensors = gameStateTensors.GetBatch(stateIndex, stateIndex + 1).Clone();
            Tensor sampleMoveIndices = sampleIndices[stateIndex..(stateIndex + 1)].clone();
            Tensor sampleValue = value[stateIndex..(stateIndex + 1)].clone();
            float[] policy = new float[moveCount];
            Array.Copy(policyData, stateIndex * moveCount, policy, 0, moveCount);
            PolicyTrainingSample sample = new()
            {
                SamplingProb = sampleSamplingProb,
                SamplingLogProb = sampleSamplingLogProb,
                StateTensors = sampleStateTensors,
                MoveIndices = sampleMoveIndices,
                Value = sampleValue,
                ChosenMoveNLProb = -chosenLogProbsManaged[stateIndex],
            };
            sample.DetachFromScope();

            GameState state = gameStates[stateIndex];
            if (!agent.IsGameDone(state))
            {
                UseHandMove move = AgentUtilities.MoveForPolicyIndex(state, (int)chosenMoveIndicesManaged[stateIndex]);
                move.Apply(state);
                AnnotationDataUtils.CreatePolicyAnnotation(policy).Apply(state);
            }

            // save sample in the output buffer
            outputSamples[stateIndex] = sample;
        }
        Profiling.Exit("BuildSamples");
        return outputSamples;
    }
}

public class PolicyTrainingSample : ITensorGroup
{
    // The chosen move is always at index 0. Remaining moves are sampled negatives when sampled softmax is enabled.
    public GameStateTensors StateTensors;
    public Tensor MoveIndices;
    public Tensor SamplingProb;
    public Tensor SamplingLogProb;

    // the networks predicted value
    public Tensor Value;

    // set after trajectory completes
    public Tensor ValueTarget;
    public Tensor PolicyAdvantage;

    /// <summary>
    /// The negative natural log probability of the chosen move. Used for debugging
    /// </summary>
    public float ChosenMoveNLProb;
}
