namespace Ramen.AI;

/// <summary>
/// Extension methods for RamenAgent that provide training-related functionality.
/// </summary>
public static class AgentTrainingExtensions
{
    /// <summary>
    /// Lower bound applied to move probabilities before negative sampling. Well below any
    /// meaningful probability, so it only matters once a move's true probability has
    /// rounded to zero.
    /// </summary>
    const float NegativeSampleProbabilityFloor = 1e-16f;

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
        (GameStateTensors gameStateTensors, Tensor logProbs, Tensor value, Tensor candidateIndices) =
            agent.GetPolicyProbDistWithCandidates(temp: 1f, gameStates); // returned tensors are on the gpu.
        Tensor probs = exp(logProbs);

        int moveCount = (int)probs.size(1);
        int minSampleCount = useSampledSoftmax ? Math.Min(2, moveCount) : 1;
        int clampedSampleCount = useSampledSoftmax ? Math.Clamp(sampleCount, minSampleCount, moveCount) : 1;

        // Index 0 is always the chosen move. Remaining indices are sampled negatives for sampled softmax.
        // Under the cascade the layout is instead [candidates | chosen-if-outside | negatives].
        Tensor sampleIndices;
        Tensor chosenMoveIndices;
        Tensor sampledProbs;
        Tensor chosenLogProbs;
        Tensor cpuProbs;
        Tensor slotValidMask = null;
        Tensor exactMass = null;
        Tensor chosenSlotIndex = null;
        if (candidateIndices is not null)
        {
            using (ProfileScope.New("SampleMoveIndices"))
            {
                cpuProbs = probs.to(CPU);
                Tensor cpuCandidates = candidateIndices.to(CPU);
                int candidateCount = (int)cpuCandidates.size(1);
                int negativeCount = Math.Max(1, Math.Min(sampleCount, moveCount - candidateCount - 1));

                Tensor chosen = multinomial(cpuProbs, 1, replacement: true);
                Tensor chosenIsCandidate = cpuCandidates.eq(chosen).any(dim: 1, keepdim: true);

                // Negatives come from outside the exact block, so their proposal is a clean
                // distribution over that complement.
                Tensor inCandidates = zeros(gameStates.Length, moveCount, dtype: ScalarType.Bool);
                inCandidates.scatter_(dim: 1, index: cpuCandidates, src: ones_like(cpuCandidates).to_type(ScalarType.Bool));
                Tensor excluded = inCandidates.logical_or(
                    arange(moveCount, dtype: ScalarType.Int64).unsqueeze(0).eq(chosen));
                Tensor negativeProbs = cpuProbs.max(NegativeSampleProbabilityFloor) *
                    excluded.logical_not().to_type(ScalarType.Float32);
                Tensor negatives = multinomial(negativeProbs, negativeCount, replacement: true);

                // The chosen slot carries the chosen move only when it is not already a
                // candidate; otherwise it is a masked duplicate of candidate zero.
                Tensor chosenSlot = where(chosenIsCandidate, cpuCandidates.narrow(1, 0, 1), chosen);
                sampleIndices = cat([cpuCandidates, chosenSlot, negatives], dim: 1);

                Tensor validCandidates = ones(gameStates.Length, candidateCount, dtype: ScalarType.Float32);
                Tensor validChosenSlot = chosenIsCandidate.logical_not().to_type(ScalarType.Float32);
                Tensor validNegatives = ones(gameStates.Length, negativeCount, dtype: ScalarType.Float32);
                slotValidMask = cat([validCandidates, validChosenSlot, validNegatives], dim: 1);

                Tensor candidateMass = cpuProbs.gather(dim: 1, index: cpuCandidates).sum(dim: 1, keepdim: true);
                Tensor chosenProb = cpuProbs.gather(dim: 1, index: chosen);
                exactMass = candidateMass + chosenProb * validChosenSlot;

                // Where the chosen move is a candidate its slot is its position in the
                // candidate block; otherwise it is the dedicated chosen slot.
                Tensor candidatePosition = cpuCandidates.eq(chosen).to_type(ScalarType.Int64).argmax(dim: 1, keepdim: true);
                chosenSlotIndex = where(
                    chosenIsCandidate,
                    candidatePosition,
                    full_like(candidatePosition, candidateCount));

                chosenMoveIndices = chosen.select(dim: 1, index: 0);
                sampledProbs = cpuProbs.gather(dim: 1, index: sampleIndices);
                Tensor chosenDevice = chosenMoveIndices.to(logProbs.device);
                chosenLogProbs = logProbs.gather(dim: 1, index: chosenDevice.unsqueeze(1)).select(dim: 1, index: 0);
            }
        }
        else
        using (ProfileScope.New("SampleMoveIndices"))
        {
            // Sample on CPU: MPS multinomial can rarely return a zero-probability first draw, which later becomes a huge KLD spike.
            cpuProbs = probs.to(CPU);
            Tensor chosenSampleIndices = multinomial(cpuProbs, 1, replacement: true);
            if (clampedSampleCount == 1)
            {
                sampleIndices = chosenSampleIndices;
            }
            else
            {
                Tensor notChosenMask = arange(moveCount, dtype: ScalarType.Int64, device: CPU).unsqueeze(0) != chosenSampleIndices;
                // Floor before masking, so the chosen move stays exactly zero and cannot be
                // drawn as its own negative. A sharp enough policy rounds every non-chosen
                // probability to zero, and multinomial rejects an all-zero row.
                Tensor negativeProbs = cpuProbs.max(NegativeSampleProbabilityFloor) * notChosenMask.to_type(ScalarType.Float32);
                Tensor negativeSampleIndices = multinomial(negativeProbs, clampedSampleCount - 1, replacement: true);
                sampleIndices = cat([chosenSampleIndices, negativeSampleIndices], dim: 1);
            }
            chosenMoveIndices = sampleIndices.select(dim: 1, index: 0);
            sampledProbs = cpuProbs.gather(dim: 1, sampleIndices);
            Tensor chosenMoveIndicesDevice = chosenMoveIndices.to(logProbs.device);
            chosenLogProbs = logProbs.gather(dim: 1, chosenMoveIndicesDevice.unsqueeze(1)).select(dim: 1, index: 0);
        }

        // Read back the chosen move index and log-prob, so the move can be made and nl prob can be stored in the sample.
        long[] chosenMoveIndicesManaged;
        float[] chosenLogProbsManaged;
        Tensor cpuChosenLogProbs;
        using (ProfileScope.New("ChosenMoveReadback"))
        {
            chosenMoveIndicesManaged = [.. chosenMoveIndices.data<long>()];
            cpuChosenLogProbs = chosenLogProbs.to(CPU);
            chosenLogProbsManaged = [.. cpuChosenLogProbs.data<float>()];
        }

        float[] policyData;
        Tensor cpuValue;
        using (ProfileScope.New("FullPolicyReadback"))
        {
            policyData = [.. cpuProbs.data<float>()];
            cpuValue = value.to(CPU);
        }

        Profiling.Enter("BuildSamples");
        // Sample and apply one move per state while creating one sample per state.
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            // Clone per-sample slices so each sample owns compact tensors instead of views to full step batches.
            Tensor sampleSlotValidMask;
            Tensor sampleExactMass;
            Tensor sampleChosenSlotIndex;
            Tensor sampleSamplingProb;
            Tensor sampleSamplingLogProb;
            GameStateTensors sampleStateTensors;
            Tensor sampleMoveIndices;
            Tensor sampleValue;
            using (ProfileScope.New("CloneSampleTensors"))
            {
                sampleSamplingProb = sampledProbs[stateIndex..(stateIndex + 1)].clone();
                sampleSamplingLogProb = cpuChosenLogProbs[stateIndex..(stateIndex + 1)].unsqueeze(1).clone();
                sampleStateTensors = gameStateTensors.GetBatch(stateIndex, stateIndex + 1).Clone();
                sampleMoveIndices = sampleIndices[stateIndex..(stateIndex + 1)].clone();
                sampleValue = cpuValue[stateIndex..(stateIndex + 1)].clone();
                sampleSlotValidMask = slotValidMask?[stateIndex..(stateIndex + 1)].clone();
                sampleExactMass = exactMass?[stateIndex..(stateIndex + 1)].clone();
                sampleChosenSlotIndex = chosenSlotIndex?[stateIndex..(stateIndex + 1)].clone();
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
                SlotValidMask = sampleSlotValidMask,
                ExactMass = sampleExactMass,
                ChosenSlotIndex = sampleChosenSlotIndex,
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

    // --- two-tier (cascade) sampling ---------------------------------------------------
    // Slot layout is [candidates | chosen-if-outside-candidates | sampled negatives].
    // Candidates and the chosen move form the exact block of the sampled-softmax
    // normalizer; only the negatives carry a log-q correction.

    /// <summary>One per slot: zero for a slot that carries no move, such as the
    /// chosen-move slot when the chosen move is already a candidate.</summary>
    public Tensor SlotValidMask;

    /// <summary>
    /// Old-policy probability mass of the exact block. The negatives are drawn from the
    /// complement of that block, so their proposal is q(a) = pi_old(a) / (1 - ExactMass)
    /// and this is the term that normalizes their correction.
    /// </summary>
    public Tensor ExactMass;

    /// <summary>Slot holding the chosen move, which is not a fixed position under the
    /// cascade layout.</summary>
    public Tensor ChosenSlotIndex;
}
