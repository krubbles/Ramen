namespace Ramen.UnitTests;

using System;
using NUnit.Framework;
using Ramen.AI;
using Ramen.Game;
using static TorchSharp.torch;

public class PolicyModelTests
{
    [Test]
    public void GetPolicyLogits_IndexBasedMatchesFull()
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();

        PolicyModel model = new();
        RamenAgent agent = new(gameState, model);

        GameStateTensors stateTensors = agent.GameStateTensors;
        UseHandTensors useHandTensors = CreateUseHandTensors(gameState);

        int useHandCount = Combinatorics.CalculateCombinationCount(
            setSize: gameState.HandState.HandCardCount,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        int moveCount = useHandCount * 2;
        int[] handIndices = new int[moveCount];
        int[] actionIndices = new int[moveCount];
        for (int i = 0; i < moveCount; ++i)
        {
            handIndices[i] = i / 2;
            actionIndices[i] = i % 2;
        }

        Tensor fullLogits = model.GetPolicyLogits(stateTensors, useHandTensors);
        Tensor indexedLogits = model.GetPolicyLogits(stateTensors, useHandTensors, handIndices, actionIndices);

        float[] fullData = fullLogits.data<float>().ToArray();
        float[] indexedData = indexedLogits.data<float>().ToArray();

        Assert.That(fullData.Length, Is.EqualTo(indexedData.Length));
        for (int i = 0; i < fullData.Length; ++i)
        {
            float delta = MathF.Abs(fullData[i] - indexedData[i]);
            Assert.That(delta, Is.LessThan(1e-5f), $"Mismatch at index {i}");
        }
    }

    static UseHandTensors CreateUseHandTensors(GameState gameState)
    {
        int useHandCount = Combinatorics.CalculateCombinationCount(
            setSize: gameState.HandState.HandCardCount,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        float[] scores = new float[useHandCount];

        int move = 0;
        int[][] cardIndicesEnumerator = Combinatorics.GetCombinations(
            setSize: gameState.HandState.HandCardCount,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        foreach (int[] cardIndices in cardIndicesEnumerator)
        {
            UseHandMove useHandMove = new(false, cardIndices);
            useHandMove.Apply(gameState);
            scores[move++] = (float)gameState.ScoringState.CurrentRoundTotalChips / 300f;
            useHandMove.Revert(gameState);
        }

        UseHandTensors useHandTensors = new()
        {
            Score = tensor(scores).view([1, -1]),
        };
        return useHandTensors;
    }
}
