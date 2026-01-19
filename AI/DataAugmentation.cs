namespace Ramen.AI;

using static TorchSharp.torch;
public static class DataAugmentation
{
    static readonly int[][] SuitRemapPermutations = CreateSuitRemapPermutations();

    public static void AugmentEvaluationTrainingDataBySuitRemap(bool includeIdentity = false)
    {
        AugmentTrainingDataBySuitRemap(TrainingData.EvaluationTrainingData, includeIdentity);
    }

    public static void AugmentTrainingDataBySuitRemap(List<PolicyTrainingSample> samples, bool includeIdentity = false)
    {
        int originalCount = samples.Count;
        for (int i = 0; i < originalCount; ++i)
        {
            PolicyTrainingSample sample = samples[i];
            for (int p = 0; p < SuitRemapPermutations.Length; ++p)
            {
                int[] permutation = SuitRemapPermutations[p];
                if (!includeIdentity && IsIdentitySuitRemap(permutation))
                    continue;

                PolicyTrainingSample remapped = CreateSuitRemappedSample(sample, permutation);
                samples.Add(remapped);
            }
        }
    }

    static PolicyTrainingSample CreateSuitRemappedSample(PolicyTrainingSample sample, int[] suitRemap)
    {
        PolicyTrainingSample remapped = new()
        {
            State = sample.State?.Clone(),
            Moves = sample.Moves?.Clone(),
            MoveProbDist = sample.MoveProbDist?.clone(),
            Advantage = sample.Advantage?.clone(),
            ChosenMoveNLProb = sample.ChosenMoveNLProb
        };

        if (remapped.State != null)
        {
            Tensor originalFullHand = remapped.State.FullHand;
            remapped.State.FullHand = RemapCardIndexTensor(remapped.State.FullHand, suitRemap);
            originalFullHand?.Dispose();

            Tensor originalRemainingDeck = remapped.State.RemainingDeck;
            remapped.State.RemainingDeck = RemapCardIndexTensor(remapped.State.RemainingDeck, suitRemap);
            originalRemainingDeck?.Dispose();
        }

        if (remapped.Moves != null)
        {
            Tensor originalPlayedHand = remapped.Moves.PlayedHand;
            remapped.Moves.PlayedHand = RemapCardIndexTensor(remapped.Moves.PlayedHand, suitRemap);
            originalPlayedHand?.Dispose();

            Tensor originalRemainingHand = remapped.Moves.RemainingHand;
            remapped.Moves.RemainingHand = RemapCardIndexTensor(remapped.Moves.RemainingHand, suitRemap);
            originalRemainingHand?.Dispose();
        }

        return remapped;
    }

    static Tensor RemapCardIndexTensor(Tensor t, int[] suitRemap)
    {
        if (t is null)
            return null;

        using var scope = NewDisposeScope();

        long[] shape = t.shape;
        ScalarType dtype = t.dtype;
        if (dtype == ScalarType.Int32)
        {
            int[] data = t.data<int>().ToArray();
            for (int i = 0; i < data.Length; ++i)
                data[i] = RemapCardIndex(data[i], suitRemap);
            Tensor remapped = tensor(data, dtype: ScalarType.Int32).view(shape).to(t.device);
            return remapped.MoveToOuterDisposeScope();
        }
        else
        {
            long[] data = t.data<long>().ToArray();
            for (int i = 0; i < data.Length; ++i)
                data[i] = RemapCardIndex((int)data[i], suitRemap);
            Tensor remapped = tensor(data, dtype: ScalarType.Int64).view(shape).to(t.device);
            return remapped.MoveToOuterDisposeScope();
        }
    }

    static int RemapCardIndex(int cardIndex, int[] suitRemap)
    {
        if (cardIndex <= 0)
            return cardIndex;

        int zeroBased = cardIndex - 1;
        int suitIndex = zeroBased / 13;
        int rankIndex = zeroBased % 13;
        int mappedSuitIndex = suitRemap[suitIndex];
        return rankIndex + 1 + mappedSuitIndex * 13;
    }

    static bool IsIdentitySuitRemap(int[] suitRemap)
    {
        return suitRemap[0] == 0 && suitRemap[1] == 1 && suitRemap[2] == 2 && suitRemap[3] == 3;
    }

    static int[][] CreateSuitRemapPermutations()
    {
        int[] suits = [0, 1, 2, 3];
        int[] current = new int[4];
        bool[] suitIsUsed = new bool[4];
        List<int[]> permutations = new(24);
        BuildSuitRemapPermutations(0, suits, suitIsUsed, current, permutations);
        return permutations.ToArray();
    }

    static void BuildSuitRemapPermutations(int depth, int[] suits, bool[] suitIsUsed, int[] current, List<int[]> permutations)
    {
        if (depth == current.Length)
        {
            int[] permutation = new int[4];
            Array.Copy(current, permutation, 4);
            permutations.Add(permutation);
            return;
        }

        for (int i = depth; i < suits.Length; ++i)
        {
            if (suitIsUsed[i])
                continue;
            suitIsUsed[i] = true;
            current[depth] = suits[i];
            BuildSuitRemapPermutations(depth + 1, suits, suitIsUsed, current, permutations);
            suitIsUsed[i] = false;
        }
    }
}