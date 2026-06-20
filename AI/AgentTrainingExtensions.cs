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

        using (ProfileScope.New("AdvanceToChoices"))
        {
            for (int i = 0; i < gameStates.Length; ++i)
                gameStates[i].AdvanceToNextPlayerChoice();
        }

        // get a batch of policy probabilities for each state as a single batched tensor
        (GameStateTensors gameStateTensors, Tensor logProbs, Tensor value) = agent.GetPolicyProbDist(temp: 1f, gameStates); // returned tensors are on the gpu.
        Tensor probs = exp(logProbs);

        int moveCount = (int)probs.size(1);
        int minSampleCount = useSampledSoftmax ? Math.Min(2, moveCount) : 1;
        int clampedSampleCount = useSampledSoftmax ? Math.Clamp(sampleCount, minSampleCount, moveCount) : 1;

        // Index 0 is always the chosen move. Remaining indices are sampled negatives for sampled softmax.
        Tensor sampleIndices;
        Tensor chosenMoveIndices;
        Tensor sampledProbs;
        Tensor chosenLogProbs;
        using (ProfileScope.New("SampleMoveIndices"))
        {
            sampleIndices = multinomial(probs, clampedSampleCount, replacement: false);
            chosenMoveIndices = sampleIndices.select(dim: 1, index: 0);
            sampledProbs = probs.gather(dim: 1, sampleIndices);
            chosenLogProbs = logProbs.gather(dim: 1, chosenMoveIndices.unsqueeze(1)).select(dim: 1, index: 0);
        }

        // Read back the chosen move index and log-prob, so the move can be made and nl prob can be stored in the sample.
        long[] chosenMoveIndicesManaged;
        float[] chosenLogProbsManaged;
        using (ProfileScope.New("ChosenMoveReadback"))
        {
            chosenMoveIndicesManaged = [.. chosenMoveIndices.to(CPU).data<long>()];
            chosenLogProbsManaged = [.. chosenLogProbs.to(CPU).data<float>()];
        }

        float[] policyData;
        using (ProfileScope.New("FullPolicyReadback"))
        {
            policyData = [.. probs.to(CPU).data<float>()];
        }

        Profiling.Enter("BuildSamples");
        // Sample and apply one move per state while creating one sample per state.
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            // Clone per-sample slices so each sample owns compact tensors instead of views to full step batches.
            Tensor sampleSamplingProb;
            Tensor sampleSamplingLogProb;
            GameStateTensors sampleStateTensors;
            Tensor sampleMoveIndices;
            Tensor sampleValue;
            using (ProfileScope.New("CloneSampleTensors"))
            {
                sampleSamplingProb = sampledProbs[stateIndex..(stateIndex + 1)].clone();
                sampleSamplingLogProb = chosenLogProbs[stateIndex..(stateIndex + 1)].unsqueeze(1).clone();
                sampleStateTensors = gameStateTensors.GetBatch(stateIndex, stateIndex + 1).Clone();
                sampleMoveIndices = sampleIndices[stateIndex..(stateIndex + 1)].clone();
                sampleValue = value[stateIndex..(stateIndex + 1)].clone();
            }
            float[] policy;
            using (ProfileScope.New("CopyPolicyRow"))
            {
                policy = new float[moveCount];
                Array.Copy(policyData, stateIndex * moveCount, policy, 0, moveCount);
            }
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
                using (ProfileScope.New("ApplyMoveAndAnnotate"))
                {
                    UseHandMove move = AgentUtilities.MoveForPolicyIndex(state, (int)chosenMoveIndicesManaged[stateIndex]);
                    move.Apply(state);
                    AnnotationDataUtils.CreatePolicyAnnotation(policy).Apply(state);
                }
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
