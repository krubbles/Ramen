namespace Ramen.AI;

using Ramen.Game;
using static TorchSharp.torch;
public static class DataAugmentation
{
    static readonly int[][] SuitRemapPermutations = CreateSuitRemapPermutations();

    public static void AugmentEvaluationTrainingDataBySuitRemap()
    {
        AugmentTrainingDataBySuitRemap(TrainingData.PolicyData);
    }

    public static void AugmentTrainingDataBySuitRemap(List<PolicyTrainingSample> samples)
    {
        FastRandom random = FastRandom.SeededByClock();
        int originalCount = samples.Count;
        for (int i = 0; i < originalCount; ++i)
        {
            PolicyTrainingSample sample = samples[i];
            for (int p = 0; p < SuitRemapPermutations.Length; ++p)
            {
                if (random.NextFlip(0.8f))
                    continue;
                int[] permutation = SuitRemapPermutations[p];
                if (IsIdentitySuitRemap(permutation))
                    continue;

                PolicyTrainingSample remapped = CreateSuitRemappedSample(sample, permutation);
                samples.Add(remapped);
            }
            if ((i + 1) % 1000 == 0)
                Console.WriteLine($"Augmented {i + 1}/{originalCount} samples...");
        }
    }

    static PolicyTrainingSample CreateSuitRemappedSample(PolicyTrainingSample sample, int[] suitRemap)
    {
        PolicyTrainingSample remapped = new()
        {
            StateTensors = sample.StateTensors?.Clone(),
            UseHandTensors = sample.UseHandTensors?.Clone(),
            SamplingProb = sample.SamplingProb?.clone(),
            Advantage = sample.Advantage?.clone(),
            ChosenMoveNLProb = sample.ChosenMoveNLProb
        };

        if (remapped.StateTensors != null)
        {
            Tensor originalFullHand = remapped.StateTensors.FullHand;
            remapped.StateTensors.FullHand = RemapCardSuitTensor(remapped.StateTensors.FullHand, suitRemap);
            originalFullHand?.Dispose();

            Tensor originalRemainingDeck = remapped.StateTensors.RemainingDeck;
            remapped.StateTensors.RemainingDeck = RemapCardSuitTensor(remapped.StateTensors.RemainingDeck, suitRemap);
            originalRemainingDeck?.Dispose();
        }

        if (remapped.UseHandTensors != null)
        {
            Tensor originalPlayedHand = remapped.UseHandTensors.PlayedHand;
            remapped.UseHandTensors.PlayedHand = RemapCardSuitTensor(remapped.UseHandTensors.PlayedHand, suitRemap);
            originalPlayedHand?.Dispose();

            Tensor originalRemainingHand = remapped.UseHandTensors.RemainingHand;
            remapped.UseHandTensors.RemainingHand = RemapCardSuitTensor(remapped.UseHandTensors.RemainingHand, suitRemap);
            originalRemainingHand?.Dispose();
        }

        return remapped;
    }

    static Tensor RemapCardSuitTensor(Tensor t, int[] suitRemap)
    {
        if (t is null)
            return null;

        using var scope = NewDisposeScope();

        long[] shape = t.shape;
        ScalarType dtype = t.dtype;
        if (dtype == ScalarType.Int32)
        {
            int[] data = t.data<int>().ToArray();
            for (int i = 0; i + 1 < data.Length; i += 2)
            {
                int rank = data[i];
                int suit = data[i + 1];
                if (rank <= 0 || suit <= 0)
                    continue;
                int suitIndex = suit - 1;
                data[i + 1] = suitRemap[suitIndex] + 1;
            }
            Tensor remapped = tensor(data, dtype: ScalarType.Int32).view(shape).to(t.device);
            return remapped.MoveToOuterDisposeScope();
        }
        else
        {
            long[] data = t.data<long>().ToArray();
            for (int i = 0; i + 1 < data.Length; i += 2)
            {
                long rank = data[i];
                long suit = data[i + 1];
                if (rank <= 0 || suit <= 0)
                    continue;
                int suitIndex = (int)suit - 1;
                data[i + 1] = suitRemap[suitIndex] + 1;
            }
            Tensor remapped = tensor(data, dtype: ScalarType.Int64).view(shape).to(t.device);
            return remapped.MoveToOuterDisposeScope();
        }
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

        for (int i = 0; i < suits.Length; ++i)
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