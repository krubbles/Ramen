namespace Ramen.ConsoleApp;

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

public static class Program
{
    const string ExperimentName = "2026-04-10_quantile_padded_swiglu_value_expected_reward_400k_suitx24_e10_b256_lr3e4_w192_l2_q50";
    const string BaselineExperimentName = "2026-04-10_padded_swiglu_value_expected_reward_400k_suitx24_e10_b256_lr3e4_w192_l2";
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
    const float QuantileHuberKappa = 1f;

    static readonly SuitPermutation[] SuitPermutations = BuildSuitPermutations();
    static readonly int[] SuitPermutationIndexMap = BuildSuitPermutationIndexMap();

    public static void Main()
    {
        set_default_device(mps_is_available() ? MPS : CPU);
        TensorManager.Init();
        Console.WriteLine("=== START ===");

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
        int augmentedValidationCount = validationBaseIndices.Length * SuitPermutations.Length;

        CSVBuilder analysis = new();
        analysis.NextRow()
            .SetCell("row_type", "config")
            .SetCell("experiment", ExperimentName)
            .SetCell("baseline_experiment", BaselineExperimentName)
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
            .SetCell("score_bucket_count", QuantilePaddedSwiGLUValueNetwork.ScoreBucketCount)
            .SetCell("score_embedding_width", QuantilePaddedSwiGLUValueNetwork.ScoreEmbeddingWidth)
            .SetCell("hands_discards_embedding_width", QuantilePaddedSwiGLUValueNetwork.HandsAndDiscardsEmbeddingWidth)
            .SetCell("input_width", QuantilePaddedSwiGLUValueNetwork.InputWidth)
            .SetCell("residual_width", ResidualWidth)
            .SetCell("swiglu_hidden_width", QuantilePaddedSwiGLUValueNetwork.SwiGLUHiddenWidth)
            .SetCell("residual_layer_count", ResidualLayerCount)
            .SetCell("quantile_count", QuantilePaddedSwiGLUValueNetwork.QuantileCount)
            .SetCell("quantile_huber_kappa", QuantileHuberKappa.ToString("F4", CultureInfo.InvariantCulture));

        File.WriteAllText(analysisCsvPath, analysis.ToString());
        File.WriteAllText(readmePath, BuildReadme(
            commitHash: GetCommitHash(repoRoot),
            datasetCount: dataset.Count,
            trainingCount: augmentedTrainingCount,
            validationCount: augmentedValidationCount));
        File.WriteAllText(notebookPath, BuildNotebookJson());
        File.Copy(Path.Combine(repoRoot, "ConsoleApp", "Program.cs"), programSnapshotPath, overwrite: true);

        Stopwatch experimentStopwatch = Stopwatch.StartNew();

        using QuantilePaddedSwiGLUValueNetwork model = new(
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
            float accumulatedTrainingEvMse = 0f;
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

                Tensor quantilePredictions = model.GetQuantiles(batch);
                Tensor loss = GetQuantileRegressionLoss(quantilePredictions, targets);
                Tensor expectedValues = quantilePredictions.mean([quantilePredictions.Dimensions - 1]);
                Tensor evMse = torch.nn.functional.mse_loss(expectedValues, targets);

                loss.backward();
                optimizer.step();

                accumulatedTrainingLoss += loss.item<float>();
                accumulatedTrainingEvMse += evMse.item<float>();
                accumulatedBatchCount++;
                trainedExamples += currentBatchSize;

                if (nextCheckpointCursor >= checkpointBatchIndices.Length || batchIndex + 1 != checkpointBatchIndices[nextCheckpointCursor])
                    continue;

                globalCheckpointIndex++;
                float meanTrainingLoss = accumulatedTrainingLoss / Math.Max(1, accumulatedBatchCount);
                float meanTrainingEvMse = accumulatedTrainingEvMse / Math.Max(1, accumulatedBatchCount);

                Stopwatch validationStopwatch = Stopwatch.StartNew();
                EvaluationMetrics validationMetrics = EvaluateModel(model, dataset, validationBaseIndices);
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
                    .SetCell("training_quantile_loss", meanTrainingLoss.ToString("F6", CultureInfo.InvariantCulture))
                    .SetCell("training_ev_mse", meanTrainingEvMse.ToString("F6", CultureInfo.InvariantCulture))
                    .SetCell("validation_quantile_loss", validationMetrics.QuantileLoss.ToString("F6", CultureInfo.InvariantCulture))
                    .SetCell("validation_ev_mse", validationMetrics.EvMse.ToString("F6", CultureInfo.InvariantCulture))
                    .SetCell("validation_seconds", validationStopwatch.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture))
                    .SetCell("elapsed_seconds", experimentStopwatch.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture))
                    .SetCell("weights_path", weightsPath);

                File.WriteAllText(analysisCsvPath, analysis.ToString());
                Console.WriteLine(
                    $"  checkpoint {globalCheckpointIndex:D3} " +
                    $"epoch_progress={(float)(batchIndex + 1) / batchCount:F3} " +
                    $"train_quantile={meanTrainingLoss:F6} " +
                    $"train_ev_mse={meanTrainingEvMse:F6} " +
                    $"val_quantile={validationMetrics.QuantileLoss:F6} " +
                    $"val_ev_mse={validationMetrics.EvMse:F6}");

                accumulatedTrainingLoss = 0f;
                accumulatedTrainingEvMse = 0f;
                accumulatedBatchCount = 0;
                nextCheckpointCursor++;
            }
        }

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


    static EvaluationMetrics EvaluateModel(QuantilePaddedSwiGLUValueNetwork model, BaseDataset dataset, int[] validationBaseIndices)
    {
        using IDisposable noGrad = no_grad();

        model.eval();

        int validationCount = validationBaseIndices.Length * SuitPermutations.Length;
        if (validationCount == 0)
            return new(QuantileLoss: 0f, EvMse: 0f);

        float totalQuantileLoss = 0f;
        float totalEvMse = 0f;
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

            Tensor quantilePredictions = model.GetQuantiles(batch);
            Tensor quantileLoss = GetQuantileRegressionLoss(quantilePredictions, targetTensor);
            Tensor expectedValues = quantilePredictions.mean([quantilePredictions.Dimensions - 1]);
            Tensor evMse = torch.nn.functional.mse_loss(expectedValues, targetTensor);

            totalQuantileLoss += quantileLoss.item<float>();
            totalEvMse += evMse.item<float>();
            batchCount++;
        }

        model.train();
        return new(
            QuantileLoss: totalQuantileLoss / Math.Max(1, batchCount),
            EvMse: totalEvMse / Math.Max(1, batchCount));
    }


    static Tensor GetQuantileRegressionLoss(Tensor predictedQuantiles, Tensor targets)
    {
        using var scope = NewDisposeScope();

        Tensor taus = (arange(QuantilePaddedSwiGLUValueNetwork.QuantileCount, dtype: ScalarType.Float32, device: predictedQuantiles.device) + 0.5f)
            / QuantilePaddedSwiGLUValueNetwork.QuantileCount;
        Tensor tdErrors = targets.unsqueeze(-1) - predictedQuantiles;
        Tensor absoluteTdErrors = tdErrors.abs();
        Tensor quadraticPart = absoluteTdErrors.clamp_max(QuantileHuberKappa);
        Tensor linearPart = absoluteTdErrors - quadraticPart;
        Tensor huberLoss = 0.5f * quadraticPart.square() + QuantileHuberKappa * linearPart;
        Tensor indicator = tdErrors.lt(0f).to_type(ScalarType.Float32);
        Tensor quantileWeights = (taus.unsqueeze(0) - indicator).abs();
        Tensor loss = (quantileWeights * huberLoss / QuantileHuberKappa).mean();

        loss.MoveToOuterDisposeScope();
        return loss;
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
        DirectoryInfo directory = new(AppContext.BaseDirectory);
        while (directory != null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, ".git")))
                return directory.FullName;
            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate repository root.");
    }


    static string GetCommitHash(string repoRoot)
    {
        string headPath = Path.Combine(repoRoot, ".git", "HEAD");
        if (!File.Exists(headPath))
            return "unknown";

        string headContents = File.ReadAllText(headPath).Trim();
        if (!headContents.StartsWith("ref: ", StringComparison.Ordinal))
            return headContents.Length >= 7 ? headContents[..7] : headContents;

        string refPath = Path.Combine(repoRoot, ".git", headContents[5..].Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(refPath))
            return "unknown";

        string hash = File.ReadAllText(refPath).Trim();
        return hash.Length >= 7 ? hash[..7] : hash;
    }


    static string BuildReadme(string commitHash, int datasetCount, int trainingCount, int validationCount)
    {
        StringBuilder readme = new();
        readme.AppendLine("Date: 2026-04-10");
        readme.AppendLine($"Commit Hash: {commitHash}");
        readme.AppendLine();
        readme.AppendLine("# Training Params");
        readme.AppendLine($"1. Dataset path: {GameDatabasePath}");
        readme.AppendLine($"2. Baseline comparison run: {BaselineExperimentName}");
        readme.AppendLine($"3. Base dataset count: {datasetCount}");
        readme.AppendLine($"4. Suit permutation count: {SuitPermutations.Length}");
        readme.AppendLine($"5. Augmented training count: {trainingCount}");
        readme.AppendLine($"6. Augmented validation count: {validationCount}");
        readme.AppendLine($"7. Epoch count: {EpochCount}");
        readme.AppendLine($"8. Training batch size: {BatchSize}");
        readme.AppendLine($"9. Learning rate: {LearningRate}");
        readme.AppendLine($"10. Validation fraction: {ValidationFraction}");
        readme.AppendLine($"11. Split seed: {SplitSeed}");
        readme.AppendLine($"12. Checkpoints per epoch: {CheckpointsPerEpoch}");
        readme.AppendLine($"13. Score threshold: {ScoreThreshold}");
        readme.AppendLine($"14. Score bucket count: {QuantilePaddedSwiGLUValueNetwork.ScoreBucketCount}");
        readme.AppendLine($"15. Score embedding width: {QuantilePaddedSwiGLUValueNetwork.ScoreEmbeddingWidth}");
        readme.AppendLine($"16. Hands/discards embedding width: {QuantilePaddedSwiGLUValueNetwork.HandsAndDiscardsEmbeddingWidth}");
        readme.AppendLine($"17. Input width before zero padding: {QuantilePaddedSwiGLUValueNetwork.InputWidth}");
        readme.AppendLine($"18. Residual stream width: {ResidualWidth}");
        readme.AppendLine($"19. SwiGLU hidden width: {QuantilePaddedSwiGLUValueNetwork.SwiGLUHiddenWidth}");
        readme.AppendLine($"20. Residual layer count: {ResidualLayerCount}");
        readme.AppendLine($"21. Quantile count: {QuantilePaddedSwiGLUValueNetwork.QuantileCount}");
        readme.AppendLine($"22. Quantile Huber kappa: {QuantileHuberKappa}");
        readme.AppendLine();
        readme.AppendLine("# Description");
        readme.AppendLine("- Trains a quantile-output version of the 192-wide 2-layer padded SwiGLU value network on the saved expected-reward target.");
        readme.AppendLine("- Uses 50 quantiles with QR-DQN style quantile Huber regression against the scalar expected reward target.");
        readme.AppendLine("- Logs both quantile regression loss and the MSE between the predicted expected value and the true target.");
        readme.AppendLine("- Splits base states before augmentation, then expands both train and validation sets by all 24 suit permutations.");
        readme.AppendLine("- The notebook overlays the new EV-MSE curves with the original scalar 192/l2 run on the same graph.");
        return readme.ToString();
    }


    static string BuildNotebookJson()
    {
        object notebook = new
        {
            cells = new object[]
            {
                new
                {
                    cell_type = "markdown",
                    metadata = new { },
                    source = new[]
                    {
                        "# Quantile SwiGLU EV Comparison\n",
                        "This notebook loads the live quantile run, prints the latest checkpoint, and overlays EV-MSE against the original scalar 192/l2 run.\n",
                    }
                },
                new
                {
                    cell_type = "code",
                    execution_count = (int?)null,
                    metadata = new { },
                    outputs = Array.Empty<object>(),
                    source = new[]
                    {
                        "import csv\n",
                        "from pathlib import Path\n",
                        "import matplotlib.pyplot as plt\n",
                        "\n",
                        "current_path = Path('analysis.csv')\n",
                        $"baseline_path = Path('../{BaselineExperimentName}/analysis.csv')\n",
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
                        "\n",
                        "print(f'quantile checkpoints: {len(current_rows)}')\n",
                        "print(f'baseline checkpoints: {len(baseline_rows)}')\n",
                        "if current_rows:\n",
                        "    print('latest quantile checkpoint:')\n",
                        "    print(current_rows[-1])\n",
                        "\n",
                        "fig, ax = plt.subplots(figsize=(10, 5))\n",
                        "\n",
                        "if baseline_rows:\n",
                        "    baseline_steps = [int(row['global_checkpoint']) for row in baseline_rows]\n",
                        "    baseline_train = [float(row['training_loss']) for row in baseline_rows]\n",
                        "    baseline_val = [float(row['validation_loss']) for row in baseline_rows]\n",
                        "    ax.plot(baseline_steps, baseline_train, label='scalar 192/l2 train mse')\n",
                        "    ax.plot(baseline_steps, baseline_val, label='scalar 192/l2 val mse', alpha=0.25)\n",
                        "\n",
                        "if current_rows:\n",
                        "    current_steps = [int(row['global_checkpoint']) for row in current_rows]\n",
                        "    current_train = [float(row['training_ev_mse']) for row in current_rows]\n",
                        "    current_val = [float(row['validation_ev_mse']) for row in current_rows]\n",
                        "    ax.plot(current_steps, current_train, label='quantile 192/l2 train EV mse')\n",
                        "    ax.plot(current_steps, current_val, label='quantile 192/l2 val EV mse', alpha=0.25)\n",
                        "\n",
                        "ax.set_xlabel('checkpoint')\n",
                        "ax.set_ylabel('mse')\n",
                        "ax.set_ylim(0, 0.002)\n",
                        "ax.set_title('Scalar MSE vs Quantile EV MSE')\n",
                        "ax.grid(True)\n",
                        "ax.legend()\n",
                        "plt.show()\n",
                    }
                }
            },
            metadata = new
            {
                kernelspec = new
                {
                    display_name = "Python 3",
                    language = "python",
                    name = "python3"
                },
                language_info = new
                {
                    name = "python",
                    version = "3.11"
                }
            },
            nbformat = 4,
            nbformat_minor = 5
        };

        return JsonSerializer.Serialize(notebook, new JsonSerializerOptions { WriteIndented = true });
    }


    readonly record struct EvaluationMetrics(float QuantileLoss, float EvMse);

    sealed class BaseDataset
    {
        public readonly long[,] FullHand;
        public readonly long[] HandsAndDiscards;
        public readonly float[,] Score;
        public readonly float[] Targets;

        public int Count => Targets.Length;

        public BaseDataset(int count)
        {
            FullHand = new long[count, GameData.HandSize];
            HandsAndDiscards = new long[count];
            Score = new float[count, 1];
            Targets = new float[count];
        }
    }

    readonly record struct SuitPermutation(Suit Diamond, Suit Club, Suit Heart, Suit Spade)
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
}
