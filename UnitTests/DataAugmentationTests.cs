namespace Ramen.UnitTests;

using System;
using System.Linq;
using System.Reflection;
using System.Collections.Generic;
using NUnit.Framework;
using Ramen.AI;
using static TorchSharp.torch;

public class DataAugmentationTests
{
    [Test]
    public void AugmentTrainingDataBySuitRemap_Works()
    {
        // Original sample with one card from each suit and a few extra indices
        PolicyTrainingSample sample = new()
        {
            State = new GameStateTensors
            {
                FullHand = tensor([1, 14, 27, 40], dtype: ScalarType.Int32),
                RemainingDeck = tensor([2, 15], dtype: ScalarType.Int32)
            },
            Moves = new UseHandTensors
            {
                PlayedHand = tensor([3], dtype: ScalarType.Int32),
                RemainingHand = tensor([4], dtype: ScalarType.Int32)
            }
        };

        List<PolicyTrainingSample> samples = new() { sample };

        // Include identity so we have all 24 permutations
        DataAugmentation.AugmentTrainingDataBySuitRemap(samples);

        Assert.That(samples.Count, Is.EqualTo(1 + 24), "Expected original + 24 permutations");

        // Reflect the private permutation table to verify each permutation was produced
        FieldInfo field = typeof(DataAugmentation).GetField("SuitRemapPermutations", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.That(field, Is.Not.Null, "Could not find SuitRemapPermutations field via reflection");
        int[][] perms = (int[][])field.GetValue(null);
        Assert.That(perms.Length, Is.EqualTo(24), "Expected 24 suit permutations");

        int[] originalFullHand = sample.State.FullHand.data<int>().ToArray();
        int[] originalRemainingDeck = sample.State.RemainingDeck.data<int>().ToArray();
        int[] originalPlayedHand = sample.Moves.PlayedHand.data<int>().ToArray();
        int[] originalRemainingHand = sample.Moves.RemainingHand.data<int>().ToArray();

        bool HasMatchingSample(int[] expectedFullHand, int[] expectedRemainingDeck, int[] expectedPlayedHand, int[] expectedRemainingHand)
        {
            return samples.Skip(1).Any(s =>
            {
                if (s.State is null || s.Moves is null) return false;
                int[] fh = s.State.FullHand.data<int>().ToArray();
                int[] rd = s.State.RemainingDeck.data<int>().ToArray();
                int[] ph = s.Moves.PlayedHand.data<int>().ToArray();
                int[] rh = s.Moves.RemainingHand.data<int>().ToArray();
                return fh.SequenceEqual(expectedFullHand) && rd.SequenceEqual(expectedRemainingDeck) && ph.SequenceEqual(expectedPlayedHand) && rh.SequenceEqual(expectedRemainingHand);
            });
        }

        // Local remap function matching DataAugmentation.RemapCardIndex
        static int RemapCard(int cardIndex, int[] suitRemap)
        {
            if (cardIndex <= 0)
                return cardIndex;
            int zeroBased = cardIndex - 1;
            int suitIndex = zeroBased / 13;
            int rankIndex = zeroBased % 13;
            int mappedSuitIndex = suitRemap[suitIndex];
            return rankIndex + 1 + mappedSuitIndex * 13;
        }

        // For each permutation, ensure there's a matching sample
        foreach (int[] perm in perms)
        {
            int[] expectedFullHand = originalFullHand.Select(ci => RemapCard(ci, perm)).ToArray();
            int[] expectedRemainingDeck = originalRemainingDeck.Select(ci => RemapCard(ci, perm)).ToArray();
            int[] expectedPlayedHand = originalPlayedHand.Select(ci => RemapCard(ci, perm)).ToArray();
            int[] expectedRemainingHand = originalRemainingHand.Select(ci => RemapCard(ci, perm)).ToArray();

            Assert.That(HasMatchingSample(expectedFullHand, expectedRemainingDeck, expectedPlayedHand, expectedRemainingHand), Is.True, $"Missing permutation: {string.Join(',', perm)}");
        }
    }
}
