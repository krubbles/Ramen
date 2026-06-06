namespace Ramen.AI;

/// <summary>
/// Extension methods for RamenAgent that provide training-related functionality.
/// </summary>
public static class AgentTrainingExtensions
{
    /// <summary>
    /// Makes moves based on the policy model's predicted probability distribution for a batch of game states.
    /// Also generates a GRPO training sample with <paramref name="sampleCount"/> sampled moves.
    /// </summary>
    public static PolicyTrainingSample[] MakeMoveAndTrainingSample(this PolicyNetworkAgent agent, GameState[] gameStates, int sampleCount)
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
        (GameStateTensors gameStateTensors, Tensor probs, Tensor value) = agent.GetPolicyProbDist(temp: 1f, gameStates); // returned tensors are on the gpu.


        int moveCount = (int)probs.size(1);
        int clampedSampleCount = Math.Clamp(sampleCount, 1, moveCount);

        // sample move indices using the policy. Index 0 for each item in the batch is the chosen move,
        // the remaining are used for sampled softmax training
        Tensor sampleIndices = multinomial(probs, clampedSampleCount, replacement: false);

        Tensor chosenMoveIndices = sampleIndices.select(dim: 1, index: 0);
        Tensor sampledProbs = probs.gather(dim: 1, sampleIndices);
        Tensor chosenProbs = sampledProbs.select(dim: 1, index: 0);

        // readback the chosen move index and prob, so the move can be made and nl prob can be stored in the sample
        Profiling.Enter("ChosenMoveReadback");
        long[] chosenMoveIndicesManaged = [.. chosenMoveIndices.to(CPU).data<long>()];
        float[] chosenProbsManaged = [.. chosenProbs.to(CPU).data<float>()];
        float[] policyData = [.. probs.to(CPU).data<float>()];
        Profiling.Exit("ChosenMoveReadback");

        Profiling.Enter("BuildSamples");
        // Sample and apply one move per state while creating one sample per state.
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            // Clone per-sample slices so each sample owns compact tensors instead of views to full step batches.
            Tensor sampleSamplingProb = sampledProbs[stateIndex..(stateIndex + 1)].clone();
            GameStateTensors sampleStateTensors = gameStateTensors.GetBatch(stateIndex, stateIndex + 1).Clone();
            Tensor sampleMoveIndices = sampleIndices[stateIndex..(stateIndex + 1)].clone();
            Tensor sampleValue = value[stateIndex..(stateIndex + 1)].clone();
            float[] policy = new float[moveCount];
            Array.Copy(policyData, stateIndex * moveCount, policy, 0, moveCount);
            PolicyTrainingSample sample = new()
            {
                SamplingProb = sampleSamplingProb,
                StateTensors = sampleStateTensors,
                MoveIndices = sampleMoveIndices,
                Value = value[stateIndex..(stateIndex + 1)].clone(),
                ChosenMoveNLProb = -MathF.Log(chosenProbsManaged[stateIndex] + 1e-9f),
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
    // a sampled set of states, with their corresponding move indices and sampling probabilities.
    // the chosen move is always at index 0.
    public GameStateTensors StateTensors;
    public Tensor MoveIndices;
    public Tensor SamplingProb;

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
