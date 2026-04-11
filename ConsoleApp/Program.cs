namespace Ramen.ConsoleApp;

using System.Diagnostics;
using System.Globalization;
using System.Text;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Program
{
    const string ExperimentName = "2026-04-10_padded_swiglu_value_expected_reward_400k_suitx24_e10_b256_lr3e4_w192_l2";
    const string BaselineExperimentName = "2026-04-10_simple_value_expected_reward_400k_suitx24_e10_b256_lr3e4";
    const string SecondExperimentName = "2026-04-10_simple_value_expected_reward_400k_suitx24_e10_b256_lr3e4_w192_w96";
    const string ThirdExperimentName = "2026-04-10_simple_value_expected_reward_400k_suitx24_e10_b256_lr3e4_w128_w64_w32";
    const string FourthExperimentName = "2026-04-10_padded_swiglu_value_expected_reward_400k_suitx24_e10_b256_lr3e4";
    const string GameDatabasePath = "/Users/miles/Documents/Ramen/GameDatabases/random_discard_preredraw_winprob_h1_d1_3334x1000.bin";
    const int EpochCount = 10;
    const int BatchSize = 256;
    const int CheckpointsPerEpoch = 5;
    const int SplitSeed = 12345;
    const float LearningRate = 3e-4f;
    const float ValidationFraction = 0.02f;
    const float ScoreThreshold = 1f;
    const int ResidualWidth = 192;
    const int ResidualLayerCount = 2;

    static readonly SuitPermutation[] SuitPermutations = BuildSuitPermutations();
    static readonly int[] SuitPermutationIndexMap = BuildSuitPermutationIndexMap();

    public static void Main()
    {
        // Do not change START
        set_default_device(mps_is_available() ? MPS : CPU);
        TensorManager.Init();
        Console.WriteLine("=== START ===");
        // Do not change END

        RunExperiment();
    }


    static void RunExperiment()
    {
        string repoRoot = FindRepoRoot();
        string analysisDir = Path.Combine(repoRoot, "Analysis", ExperimentName);
        string weightsDir = Path.Combine(analysisDir, "weights");
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string programSnapshotPath = Path.Combine(analysisDir, "Program.cs");
        string readmePath = Path.Combine(analysisDir, "README.md");
        string notebookPath = Path.Combine(analysisDir, "analysis.ipynb");

        Directory.CreateDirectory(analysisDir);
        Directory.CreateDirectory(weightsDir);

        BaseDataset dataset = LoadDataset(GameDatabasePath);
        int[] shuffledBaseIndices = CreateShuffledIndices(dataset.Count, seed: SplitSeed);
        int validationBaseCount = (int)MathF.Round(dataset.Count * ValidationFraction);
        int trainingBaseCount = dataset.Count - validationBaseCount;

        int[] trainingBaseIndices = new int[trainingBaseCount];
        int[] validationBaseIndices = new int[validationBaseCount];
        Array.Copy(shuffledBaseIndices, 0, trainingBaseIndices, 0, trainingBaseCount);
        Array.Copy(shuffledBaseIndices, trainingBaseCount, validationBaseIndices, 0, validationBaseCount);

        int augmentedTrainingCount = trainingBaseCount * SuitPermutations.Length;
        int augmentedValidationCount = validationBaseCount * SuitPermutations.Length;

        CSVBuilder analysis = new();
        analysis.NextRow()
            .SetCell("row_type", "config")
            .SetCell("experiment", ExperimentName)
            .SetCell("game_database_path", GameDatabasePath)
            .SetCell("base_dataset_count", dataset.Count)
            .SetCell("suit_permutation_count", SuitPermutations.Length)
            .SetCell("augmented_dataset_count", dataset.Count * SuitPermutations.Length)
            .SetCell("training_count", augmentedTrainingCount)
            .SetCell("validation_count", augmentedValidationCount)
            .SetCell("epoch_count", EpochCount)
            .SetCell("training_batch_size", BatchSize)
            .SetCell("learning_rate", LearningRate.ToString("F6", CultureInfo.InvariantCulture))
            .SetCell("validation_fraction", ValidationFraction.ToString("F4", CultureInfo.InvariantCulture))
            .SetCell("split_seed", SplitSeed)
            .SetCell("checkpoint_count_per_epoch", CheckpointsPerEpoch)
            .SetCell("score_threshold", ScoreThreshold.ToString("F4", CultureInfo.InvariantCulture))
            .SetCell("score_bucket_count", PaddedSwiGLUValueNetwork.ScoreBucketCount)
            .SetCell("score_embedding_width", PaddedSwiGLUValueNetwork.ScoreEmbeddingWidth)
            .SetCell("hands_discards_embedding_width", PaddedSwiGLUValueNetwork.HandsAndDiscardsEmbeddingWidth)
            .SetCell("input_width", PaddedSwiGLUValueNetwork.InputWidth)
            .SetCell("residual_width", ResidualWidth)
            .SetCell("swiglu_hidden_width", PaddedSwiGLUValueNetwork.SwiGLUHiddenWidth)
            .SetCell("residual_layer_count", ResidualLayerCount);

        Stopwatch experimentStopwatch = Stopwatch.StartNew();

        using PaddedSwiGLUValueNetwork model = new(
            scoreThreshold: ScoreThreshold,
            residualWidth: ResidualWidth,
            residualLayerCount: ResidualLayerCount);
        using AdamW optimizer = optim.AdamW(
            parameters: model.parameters(),
            lr: LearningRate,
            weight_decay: 0f,
            beta1: 0.9f,
            beta2: 0.99f);

        int globalCheckpointIndex = 0;

        for (int epoch = 1; epoch <= EpochCount; ++epoch)
        {
            Console.WriteLine($"Epoch {epoch}/{EpochCount}");
            int[] trainingOrder = BuildAugmentedOrder(
                baseIndices: trainingBaseIndices,
                permutationCount: SuitPermutations.Length,
                seed: SplitSeed + epoch * 7919);
            int batchCount = DivideRoundUp(trainingOrder.Length, BatchSize);
            int[] checkpointBatchIndices = BuildCheckpointBatchIndices(batchCount, CheckpointsPerEpoch);

            model.train();
            float accumulatedTrainingLoss = 0f;
            int accumulatedBatchCount = 0;
            int trainedExamples = 0;
            int nextCheckpointCursor = 0;

            for (int batchIndex = 0; batchIndex < batchCount; ++batchIndex)
            {
                int batchStart = batchIndex * BatchSize;
                int currentBatchSize = Math.Min(BatchSize, trainingOrder.Length - batchStart);

                using var scope = NewDisposeScope();

                GameStateTensors batch = CreateBatch(dataset, trainingOrder, batchStart, currentBatchSize);
                Tensor targets = CreateTargetTensor(dataset, trainingOrder, batchStart, currentBatchSize);

                optimizer.zero_grad();

                Tensor predictions = model.GetAdvantages(batch);
                Tensor loss = functional.mse_loss(predictions, targets);
                loss.backward();
                optimizer.step();

                accumulatedTrainingLoss += loss.item<float>();
                accumulatedBatchCount++;
                trainedExamples += currentBatchSize;

                if (nextCheckpointCursor >= checkpointBatchIndices.Length || batchIndex + 1 != checkpointBatchIndices[nextCheckpointCursor])
                    continue;

                globalCheckpointIndex++;
                float meanTrainingLoss = accumulatedTrainingLoss / Math.Max(1, accumulatedBatchCount);
                Stopwatch validationStopwatch = Stopwatch.StartNew();
                float validationLoss = EvaluateModel(model, dataset, validationBaseIndices);
                validationStopwatch.Stop();

                string weightsPath = Path.Combine(weightsDir, $"step{globalCheckpointIndex:D3}.bin");
                model.save(weightsPath);
                model.save(Path.Combine(weightsDir, "latest.bin"));

                analysis.NextRow()
                    .SetCell("row_type", "checkpoint")
                    .SetCell("experiment", ExperimentName)
                    .SetCell("epoch", epoch)
                    .SetCell("epoch_progress", ((float)(batchIndex + 1) / batchCount).ToString("F4", CultureInfo.InvariantCulture))
                    .SetCell("checkpoint_in_epoch", nextCheckpointCursor + 1)
                    .SetCell("global_checkpoint", globalCheckpointIndex)
                    .SetCell("trained_examples", trainedExamples)
                    .SetCell("training_loss", meanTrainingLoss.ToString("F6", CultureInfo.InvariantCulture))
                    .SetCell("validation_loss", validationLoss.ToString("F6", CultureInfo.InvariantCulture))
                    .SetCell("validation_seconds", validationStopwatch.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture))
                    .SetCell("elapsed_seconds", experimentStopwatch.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture))
                    .SetCell("weights_path", weightsPath);

                File.WriteAllText(analysisCsvPath, analysis.ToString());
                Console.WriteLine(
                    $"  checkpoint {globalCheckpointIndex:D3} " +
                    $"epoch_progress={(float)(batchIndex + 1) / batchCount:F3} " +
                    $"train_loss={meanTrainingLoss:F6} val_loss={validationLoss:F6}");

                accumulatedTrainingLoss = 0f;
                accumulatedBatchCount = 0;
                nextCheckpointCursor++;
            }
        }

        File.Copy(Path.Combine(repoRoot, "ConsoleApp", "Program.cs"), programSnapshotPath, overwrite: true);
        File.WriteAllText(readmePath, BuildReadme(
            datasetCount: dataset.Count,
            trainingCount: augmentedTrainingCount,
            validationCount: augmentedValidationCount));
        File.WriteAllText(notebookPath, BuildNotebookJson());
        File.WriteAllText(analysisCsvPath, analysis.ToString());
    }


    static BaseDataset LoadDataset(string gameDatabasePath)
    {
        int count = CountAnnotatedGames(gameDatabasePath);
        BaseDataset dataset = new(count);
        GameData gameData = CreateOneHandOneDiscardGameData();

        using FileStream fileStream = new(gameDatabasePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        int recordIndex = 0;
        while (fileStream.Position < fileStream.Length)
        {
            GameState gameState = new(gameData);
            gameState.Deserialize(fileStream);

            if (!TryGetTarget(gameState, out float target))
                continue;

            ReadOnlySpan<Card> hand = gameState.HandState.Hand;
            for (int cardIndex = 0; cardIndex < hand.Length; ++cardIndex)
                dataset.FullHand[recordIndex, cardIndex] = hand[cardIndex].ToIndex();

            dataset.HandsAndDiscards[recordIndex] = gameState.HandState.RemainingHands * 5 + gameState.HandState.RemainingDiscards;
            dataset.Score[recordIndex, 0] = (float)gameState.ScoringState.CurrentRoundTotalChips / 300f;
            dataset.Targets[recordIndex] = target;
            recordIndex++;
        }

        return dataset;
    }


    static int CountAnnotatedGames(string gameDatabasePath)
    {
        int count = 0;
        GameData gameData = CreateOneHandOneDiscardGameData();
        using FileStream fileStream = new(gameDatabasePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        while (fileStream.Position < fileStream.Length)
        {
            GameState gameState = new(gameData);
            gameState.Deserialize(fileStream);
            if (TryGetTarget(gameState, out _))
                count++;
        }

        return count;
    }


    static GameStateTensors CreateBatch(BaseDataset dataset, int[] augmentedIndices, int batchStart, int batchSize)
    {
        long[,] hand = new long[batchSize, GameData.HandSize];
        long[] handsAndDiscards = new long[batchSize];
        float[,] score = new float[batchSize, 1];

        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            int augmentedIndex = augmentedIndices[batchStart + batchIndex];
            int baseIndex = augmentedIndex / SuitPermutations.Length;
            int permutationIndex = augmentedIndex % SuitPermutations.Length;
            int mapOffset = permutationIndex * (Card.RankCount * Card.SuitCount + 1);

            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
            {
                long sourceCardIndex = dataset.FullHand[baseIndex, cardIndex];
                hand[batchIndex, cardIndex] = SuitPermutationIndexMap[mapOffset + (int)sourceCardIndex];
            }

            handsAndDiscards[batchIndex] = dataset.HandsAndDiscards[baseIndex];
            score[batchIndex, 0] = dataset.Score[baseIndex, 0];
        }

        return new()
        {
            FullHand = tensor(hand, dtype: ScalarType.Int64, device: ValueNetwork.EvalDevice),
            HandsAndDiscards = tensor(handsAndDiscards, dtype: ScalarType.Int64, device: ValueNetwork.EvalDevice),
            Score = tensor(score, dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice),
        };
    }


    static Tensor CreateTargetTensor(BaseDataset dataset, int[] augmentedIndices, int batchStart, int batchSize)
    {
        float[] targets = new float[batchSize];
        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            int augmentedIndex = augmentedIndices[batchStart + batchIndex];
            int baseIndex = augmentedIndex / SuitPermutations.Length;
            targets[batchIndex] = dataset.Targets[baseIndex];
        }

        return tensor(targets, dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice);
    }


    static float EvaluateModel(PaddedSwiGLUValueNetwork model, BaseDataset dataset, int[] validationBaseIndices)
    {
        using var noGrad = no_grad();
        model.eval();

        int validationCount = validationBaseIndices.Length * SuitPermutations.Length;
        if (validationCount == 0)
            return 0f;

        float totalLoss = 0f;
        int batchCount = 0;

        for (int augmentedStart = 0; augmentedStart < validationCount; augmentedStart += BatchSize)
        {
            int currentBatchSize = Math.Min(BatchSize, validationCount - augmentedStart);
            using var scope = NewDisposeScope();

            long[,] hand = new long[currentBatchSize, GameData.HandSize];
            long[] handsAndDiscards = new long[currentBatchSize];
            float[,] score = new float[currentBatchSize, 1];
            float[] targets = new float[currentBatchSize];

            for (int batchIndex = 0; batchIndex < currentBatchSize; ++batchIndex)
            {
                int augmentedIndex = augmentedStart + batchIndex;
                int baseArrayIndex = augmentedIndex / SuitPermutations.Length;
                int baseIndex = validationBaseIndices[baseArrayIndex];
                int permutationIndex = augmentedIndex % SuitPermutations.Length;
                int mapOffset = permutationIndex * (Card.RankCount * Card.SuitCount + 1);

                for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                {
                    long sourceCardIndex = dataset.FullHand[baseIndex, cardIndex];
                    hand[batchIndex, cardIndex] = SuitPermutationIndexMap[mapOffset + (int)sourceCardIndex];
                }

                handsAndDiscards[batchIndex] = dataset.HandsAndDiscards[baseIndex];
                score[batchIndex, 0] = dataset.Score[baseIndex, 0];
                targets[batchIndex] = dataset.Targets[baseIndex];
            }

            GameStateTensors batch = new()
            {
                FullHand = tensor(hand, dtype: ScalarType.Int64, device: ValueNetwork.EvalDevice),
                HandsAndDiscards = tensor(handsAndDiscards, dtype: ScalarType.Int64, device: ValueNetwork.EvalDevice),
                Score = tensor(score, dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice),
            };
            Tensor targetTensor = tensor(targets, dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice);

            Tensor predictions = model.GetAdvantages(batch);
            Tensor loss = functional.mse_loss(predictions, targetTensor);
            totalLoss += loss.item<float>();
            batchCount++;
        }

        model.train();
        return totalLoss / Math.Max(1, batchCount);
    }


    static bool TryGetTarget(GameState gameState, out float target)
    {
        for (int moveIndex = gameState.MoveState.MoveHistory.Count - 1; moveIndex >= 0; --moveIndex)
        {
            if (gameState.MoveState.MoveHistory[moveIndex] is not AnnotatingDataMove annotation)
                continue;

            if (AnnotationDataUtils.TryDecodeExpectedRewardAnnotation(annotation, out target))
                return true;

            if (annotation.DataTypeID == (ushort)AnnotationDataType.Policy)
                continue;

            if (annotation.Data.Length == sizeof(float))
            {
                target = BitConverter.ToSingle(annotation.Data, 0);
                return true;
            }
        }

        target = 0f;
        return false;
    }


    static int[] BuildAugmentedOrder(int[] baseIndices, int permutationCount, int seed)
    {
        int[] order = new int[baseIndices.Length * permutationCount];
        int cursor = 0;
        for (int index = 0; index < baseIndices.Length; ++index)
        {
            int baseIndex = baseIndices[index];
            for (int permutationIndex = 0; permutationIndex < permutationCount; ++permutationIndex)
            {
                order[cursor] = baseIndex * permutationCount + permutationIndex;
                cursor++;
            }
        }

        Shuffle(order, seed);
        return order;
    }


    static int[] CreateShuffledIndices(int count, int seed)
    {
        int[] indices = new int[count];
        for (int index = 0; index < count; ++index)
            indices[index] = index;
        Shuffle(indices, seed);
        return indices;
    }


    static void Shuffle(int[] indices, int seed)
    {
        Random random = new(seed);
        for (int index = indices.Length - 1; index > 0; --index)
        {
            int swapIndex = random.Next(index + 1);
            (indices[index], indices[swapIndex]) = (indices[swapIndex], indices[index]);
        }
    }


    static int[] BuildCheckpointBatchIndices(int batchCount, int checkpointCount)
    {
        int[] checkpointBatchIndices = new int[checkpointCount];
        for (int checkpointIndex = 0; checkpointIndex < checkpointCount; ++checkpointIndex)
        {
            checkpointBatchIndices[checkpointIndex] =
                (int)Math.Ceiling((checkpointIndex + 1) * batchCount / (float)checkpointCount);
        }

        return checkpointBatchIndices;
    }


    static int DivideRoundUp(int numerator, int denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }


    static int[] BuildSuitPermutationIndexMap()
    {
        int cardIndexCount = Card.RankCount * Card.SuitCount + 1;
        int[] map = new int[SuitPermutations.Length * cardIndexCount];

        for (int permutationIndex = 0; permutationIndex < SuitPermutations.Length; ++permutationIndex)
        {
            int baseOffset = permutationIndex * cardIndexCount;
            map[baseOffset] = 0;

            for (int cardIndex = 1; cardIndex < cardIndexCount; ++cardIndex)
            {
                int zeroBasedIndex = cardIndex - 1;
                int rankIndex = zeroBasedIndex % Card.RankCount + 1;
                Suit suit = (Suit)(zeroBasedIndex / Card.RankCount + 1);
                Suit permutedSuit = SuitPermutations[permutationIndex].Map(suit);
                map[baseOffset + cardIndex] = rankIndex + ((int)permutedSuit - 1) * Card.RankCount;
            }
        }

        return map;
    }


    static SuitPermutation[] BuildSuitPermutations()
    {
        Suit[] suits = [Suit.Diamond, Suit.Club, Suit.Heart, Suit.Spade];
        List<SuitPermutation> permutations = [];
        BuildSuitPermutationsRecursive(
            remainingSuits: suits,
            depth: 0,
            current: new Suit[4],
            output: permutations);
        return [.. permutations];
    }


    static void BuildSuitPermutationsRecursive(
        ReadOnlySpan<Suit> remainingSuits,
        int depth,
        Span<Suit> current,
        List<SuitPermutation> output)
    {
        if (depth == current.Length)
        {
            output.Add(new(
                Diamond: current[0],
                Club: current[1],
                Heart: current[2],
                Spade: current[3]));
            return;
        }

        for (int suitIndex = 0; suitIndex < remainingSuits.Length; ++suitIndex)
        {
            current[depth] = remainingSuits[suitIndex];

            Suit[] nextRemaining = new Suit[remainingSuits.Length - 1];
            int nextIndex = 0;
            for (int innerIndex = 0; innerIndex < remainingSuits.Length; ++innerIndex)
            {
                if (innerIndex == suitIndex)
                    continue;

                nextRemaining[nextIndex] = remainingSuits[innerIndex];
                nextIndex++;
            }

            BuildSuitPermutationsRecursive(
                remainingSuits: nextRemaining,
                depth: depth + 1,
                current: current,
                output: output);
        }
    }


    static GameData CreateOneHandOneDiscardGameData()
    {
        return new()
        {
            Hands = 1,
            Discards = 1,
            RandomizeSeed = true,
        };
    }


    static string FindRepoRoot()
    {
        string current = AppContext.BaseDirectory;
        while (!string.IsNullOrEmpty(current))
        {
            if (File.Exists(Path.Combine(current, "Ramen.sln")))
                return current;

            string? parent = Directory.GetParent(current)?.FullName;
            if (string.Equals(parent, current, StringComparison.Ordinal))
                break;

            current = parent ?? string.Empty;
        }

        throw new InvalidOperationException("Could not locate repository root.");
    }


    static string BuildReadme(int datasetCount, int trainingCount, int validationCount)
    {
        List<string> lines =
        [
            $"Date: {DateTime.Now:yyyy-MM-dd}",
            $"Commit Hash: {GetCommitHash()}",
            string.Empty,
            "# Training Params",
            $"1. Dataset path: {GameDatabasePath}",
            $"2. Base dataset count: {datasetCount}",
            $"3. Suit permutation count: {SuitPermutations.Length}",
            $"4. Augmented training count: {trainingCount}",
            $"5. Augmented validation count: {validationCount}",
            $"6. Epoch count: {EpochCount}",
            $"7. Training batch size: {BatchSize}",
            $"8. Learning rate: {LearningRate}",
            $"9. Validation fraction: {ValidationFraction}",
            $"10. Split seed: {SplitSeed}",
            $"11. Checkpoints per epoch: {CheckpointsPerEpoch}",
            $"12. Score threshold: {ScoreThreshold}",
            $"13. Score bucket count: {PaddedSwiGLUValueNetwork.ScoreBucketCount}",
            $"14. Score embedding width: {PaddedSwiGLUValueNetwork.ScoreEmbeddingWidth}",
            $"15. Hands/discards embedding width: {PaddedSwiGLUValueNetwork.HandsAndDiscardsEmbeddingWidth}",
            $"16. Input width before zero padding: {PaddedSwiGLUValueNetwork.InputWidth}",
            $"17. Residual stream width: {ResidualWidth}",
            $"18. SwiGLU hidden width: {PaddedSwiGLUValueNetwork.SwiGLUHiddenWidth}",
            $"19. Residual layer count: {ResidualLayerCount}",
            string.Empty,
            "# Description",
            "- Trains `PaddedSwiGLUValueNetwork` with MSE on the saved float annotation from the one-hand one-discard pre-discard dataset.",
            "- Treats the dataset's `winprob` annotation as the expected reward target, per the experiment request.",
            "- Splits base states before augmentation, then expands both train and validation sets by all 24 suit permutations.",
            "- Concatenates zero features to expand the 169-wide state vector to a 192-wide residual stream.",
            "- Applies 2 pre-norm 1:1 SwiGLU residual layers with gate and value hidden widths of 284, then GELU and a final linear head.",
            "- Shuffles the augmented training order every epoch and writes a single `analysis.csv` with checkpointed train and validation loss.",
            "- The notebook overlays this run with the earlier `128/64`, `192/96`, partial `128/64/32`, and `384`-wide padded SwiGLU trajectories.",
        ];

        return string.Join(Environment.NewLine, lines) + Environment.NewLine;
    }


    static string BuildNotebookJson()
    {
        return $$"""
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Padded SwiGLU Value Training\n",
        "\n",
        "Overlays the current padded SwiGLU run with the earlier `128/64`, `192/96`, and partial `128/64/32` runs."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import csv\n",
        "from pathlib import Path\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "current_path = Path('analysis.csv')\n",
        "baseline_path = Path('../{{BaselineExperimentName}}/analysis.csv')\n",
        "second_path = Path('../{{SecondExperimentName}}/analysis.csv')\n",
        "third_path = Path('../{{ThirdExperimentName}}/analysis.csv')\n",
        "fourth_path = Path('../{{FourthExperimentName}}/analysis.csv')\n",
        "\n",
        "def load_rows(path):\n",
        "    rows = []\n",
        "    if not path.exists():\n",
        "        return rows\n",
        "    with path.open() as f:\n",
        "        reader = csv.DictReader(f)\n",
        "        for row in reader:\n",
        "            if row['row_type'] == 'checkpoint':\n",
        "                rows.append(row)\n",
        "    return rows\n",
        "\n",
        "current_rows = load_rows(current_path)\n",
        "baseline_rows = load_rows(baseline_path)\n",
        "second_rows = load_rows(second_path)\n",
        "third_rows = load_rows(third_path)\n",
        "fourth_rows = load_rows(fourth_path)\n",
        "\n",
        "print(f'current checkpoints: {len(current_rows)}')\n",
        "print(f'baseline checkpoints: {len(baseline_rows)}')\n",
        "print(f'second checkpoints: {len(second_rows)}')\n",
        "print(f'third checkpoints: {len(third_rows)}')\n",
        "print(f'fourth checkpoints: {len(fourth_rows)}')\n",
        "if current_rows:\n",
        "    print('latest current checkpoint:')\n",
        "    print(current_rows[-1])\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(10, 5))\n",
        "\n",
        "for rows, label in [\n",
        "    (baseline_rows, 'baseline 128/64'),\n",
        "    (second_rows, 'second 192/96'),\n",
        "    (third_rows, 'third 128/64/32'),\n",
        "    (fourth_rows, 'fourth padded SwiGLU 384/l3'),\n",
        "    (current_rows, 'current padded SwiGLU 192/l2'),\n",
        "]:\n",
        "    steps = [int(row['global_checkpoint']) for row in rows]\n",
        "    train_loss = [float(row['training_loss']) for row in rows]\n",
        "    val_loss = [float(row['validation_loss']) for row in rows]\n",
        "    ax.plot(steps, train_loss, label=f'{label} train')\n",
        "    ax.plot(steps, val_loss, label=f'{label} val', alpha=0.25)\n",
        "\n",
        "ax.set_xlabel('checkpoint')\n",
        "ax.set_ylabel('mse')\n",
        "ax.set_ylim(0, 0.002)\n",
        "ax.set_title('Loss')\n",
        "ax.grid(True)\n",
        "ax.legend()\n",
        "plt.show()\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.x"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
""";
    }


    static string GetCommitHash()
    {
        try
        {
            ProcessStartInfo startInfo = new("git", "rev-parse --short HEAD")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = FindRepoRoot(),
            };
            using Process process = Process.Start(startInfo);
            if (process == null)
                return "UNKNOWN";

            string output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit();
            return string.IsNullOrWhiteSpace(output) ? "UNKNOWN" : output;
        }
        catch
        {
            return "UNKNOWN";
        }
    }


    readonly record struct SuitPermutation(
        Suit Diamond,
        Suit Club,
        Suit Heart,
        Suit Spade)
    {
        public Suit Map(Suit suit)
        {
            return suit switch
            {
                Suit.Diamond => Diamond,
                Suit.Club => Club,
                Suit.Heart => Heart,
                Suit.Spade => Spade,
                _ => suit,
            };
        }
    }


    sealed class BaseDataset
    {
        public readonly long[,] FullHand;
        public readonly long[] HandsAndDiscards;
        public readonly float[,] Score;
        public readonly float[] Targets;

        public BaseDataset(int count)
        {
            FullHand = new long[count, GameData.HandSize];
            HandsAndDiscards = new long[count];
            Score = new float[count, 1];
            Targets = new float[count];
        }

        public int Count => Targets.Length;
    }
}
