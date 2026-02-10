namespace Ramen.Training;

using System;
using System.Collections.Generic;
using Ramen.AI;
using Ramen.Game;
using Tensorboard;
using static TorchSharp.torch;

/// <summary>
/// Extension methods for RamenAgent that provide training-related functionality.
/// </summary>
public static class AgentTrainingExtensions
{
    /// <summary>
    /// Makes moves based on the policy model's predicted probability distribution for a batch of game states.
    /// Also generates a GRPO training sample with <paramref name="sampleCount"/> sampled moves.
    /// </summary>
    public static PolicyTrainingSample[] MakeMoveAndTrainingSample(this PolicyOnlyAgent agent, GameState[] gameStates, int sampleCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        using var p_funcScope = ProfileScope.New(nameof(MakeMoveAndTrainingSample));

        // go through each state, advance them to the next player choice, and build a list of states that aren't done. 
        PolicyTrainingSample[] outputSamples = new PolicyTrainingSample[gameStates.Length];
        List<int> activeIndices = new();
        for (int i = 0; i < gameStates.Length; ++i)
        {
            GameState gameState = gameStates[i];
            gameState.AdvanceToNextPlayerChoice();
            if (!agent.IsGameDone(gameState))
                activeIndices.Add(i);
        }

        // all states are done, nothing to do
        if (activeIndices.Count == 0)
            return outputSamples;

        // Build active state batch and evaluate policy once.
        GameState[] activeStates = new GameState[activeIndices.Count];
        for (int i = 0; i < activeStates.Length; ++i)
            activeStates[i] = gameStates[activeIndices[i]];

        // get a batch of policy probabilities for each active state as a single batched tensor
        (GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor probs) = agent.GetPolicyProbDist(temp: 1f, activeStates); // returned tensors are on the gpu.


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
        Profiling.Exit("ChosenMoveReadback");
        
        Profiling.Enter("BuildSamples");
        // Sample and apply one move per active state while creating one sample per state.
        for (int activeIndex = 0; activeIndex < activeStates.Length; ++activeIndex)
        {
            // Clone per-sample slices so each sample owns compact tensors instead of views to full step batches.
            Tensor sampleSamplingProb = sampledProbs[activeIndex..(activeIndex + 1)].clone();
            GameStateTensors sampleStateTensors = gameStateTensors.GetBatch(activeIndex, activeIndex + 1).Clone();
            UseHandTensors sampleUseHandTensors = useHandTensors.GetBatch(activeIndex, activeIndex + 1).Clone();
            Tensor sampleMoveIndices = sampleIndices[activeIndex..(activeIndex + 1)].clone();

            PolicyTrainingSample sample = new()
            {
                SamplingProb = sampleSamplingProb,
                StateTensors = sampleStateTensors,
                UseHandTensors = sampleUseHandTensors,
                MoveIndices = sampleMoveIndices,
                ChosenMoveNLProb = -MathF.Log(chosenProbsManaged[activeIndex] + 1e-9f),
            };
            sample.DetachFromScope();

            // make the chosen move
            GameState state = activeStates[activeIndex];
            UseHandMove move = PolicyOnlyAgent.MoveForIndex(state, (int)chosenMoveIndicesManaged[activeIndex]);
            move.Apply(state);

            // save sample in the output buffer
            outputSamples[activeIndices[activeIndex]] = sample;
        }
        Profiling.Exit("BuildSamples"); 
        return outputSamples;
    }
}
