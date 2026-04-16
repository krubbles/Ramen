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
    const string ExperimentName = "2026-04-13_quantile_dqn_standardreward_defaultstart_eps0p1_q50_replay500k_suitx24_reuse2_lr2e4_rg64";
    const int TrainingStepCount = 300;
    const int CheckpointInterval = 5;
    const int RolloutGameCount = 64;
    const int TrainingBatchSize = 256;
    const int EvaluationGameCount = 2000;
    const int ReplayCapacity = 500_000;
    const int MinimumReplaySize = 2048;
    const int TargetSyncInterval = 10;
    const int RandomSeed = 12345;
    const float LearningRate = 2e-4f;
    const float Epsilon = 0.1f;
    const float Gamma = 1f;
    const float ScoreThreshold = 1f;
    const float QuantileHuberKappa = 1f;
    const int ResidualWidth = 192;
    const int ResidualLayerCount = 2;
    const int SuccessorEvaluationBatchSize = 256;
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

        CSVBuilder analysis = CreateConfigAnalysis();
        File.WriteAllText(readmePath, BuildReadme(commitHash: GetCommitHash(repoRoot)));
        File.WriteAllText(notebookPath, BuildNotebookJson());
        File.Copy(Path.Combine(repoRoot, "ConsoleApp", "Program.cs"), programSnapshotPath, overwrite: true);

        ReplayBuffer replayBuffer = new(ReplayCapacity);
        Random random = new(RandomSeed);
        Stopwatch experimentStopwatch = Stopwatch.StartNew();

        using QuantilePaddedSwiGLUValueNetwork onlineModel = new(
            scoreThreshold: ScoreThreshold,
            residualWidth: ResidualWidth,
            residualLayerCount: ResidualLayerCount);
        using QuantilePaddedSwiGLUValueNetwork targetModel = new(
            scoreThreshold: ScoreThreshold,
            residualWidth: ResidualWidth,
            residualLayerCount: ResidualLayerCount);
        using AdamW optimizer = optim.AdamW(
            parameters: onlineModel.parameters(),
            lr: LearningRate,
            weight_decay: 0f,
            beta1: 0.9f,
            beta2: 0.99f);
        using QuantileGreedyPlyOneAgent rolloutAgent = new(onlineModel, epsilon: Epsilon);
        using QuantileGreedyPlyOneAgent evaluationAgent = new(onlineModel, epsilon: 0f);
        CopyWeights(source: onlineModel, destination: targetModel);
        File.WriteAllText(analysisCsvPath, analysis.ToString());

        for (int step = 1; step <= TrainingStepCount; ++step)
        {
            Console.WriteLine($"step {step}/{TrainingStepCount}");

            int addedTransitionCount = CollectRollouts(
                replayBuffer: replayBuffer,
                agent: rolloutAgent,
                gameCount: RolloutGameCount,
                random: random);

            if (replayBuffer.Count < MinimumReplaySize)
            {
                analysis.NextRow()
                    .SetCell("row_type", "step")
                    .SetCell("step", step)
                    .SetCell("replay_count", replayBuffer.Count)
                    .SetCell("added_transitions", addedTransitionCount)
                    .SetCell("training_updates", 0)
                    .SetCell("trained_examples", 0)
                    .SetCell("training_loss", "")
                    .SetCell("training_ev_mse", "")
                    .SetCell("eval_win_rate", "")
                    .SetCell("eval_reward_mean", "")
                    .SetCell("elapsed_seconds", experimentStopwatch.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture));
                File.WriteAllText(analysisCsvPath, analysis.ToString());
                Console.WriteLine($"  replay warmup {replayBuffer.Count}/{MinimumReplaySize}");
                continue;
            }

            int targetTrainingExampleCount = addedTransitionCount * 2;
            int updateCount = DivideRoundUp(targetTrainingExampleCount, TrainingBatchSize);
            float totalTrainingLoss = 0f;
            float totalTrainingEvMse = 0f;

            for (int updateIndex = 0; updateIndex < updateCount; ++updateIndex)
            {
                ReplayTransition[] batchTransitions = replayBuffer.SampleBatch(TrainingBatchSize, random);
                TrainingBatch trainingBatch = BuildTrainingBatch(batchTransitions);

                using var scope = NewDisposeScope();
                optimizer.zero_grad();

                Tensor predictedQuantiles = onlineModel.GetQuantiles(trainingBatch.CurrentStates);
                Tensor targetQuantiles = BuildTargetQuantiles(
                    targetModel: targetModel,
                    batchTransitions: batchTransitions);
                Tensor loss = GetQuantileDqnLoss(predictedQuantiles, targetQuantiles);
                Tensor predictedExpectedValues = predictedQuantiles.mean([predictedQuantiles.Dimensions - 1]);
                Tensor targetExpectedValues = targetQuantiles.mean([targetQuantiles.Dimensions - 1]);
                Tensor evMse = torch.nn.functional.mse_loss(predictedExpectedValues, targetExpectedValues);

                loss.backward();
                optimizer.step();

                totalTrainingLoss += loss.item<float>();
                totalTrainingEvMse += evMse.item<float>();
            }

            if (step % TargetSyncInterval == 0)
                CopyWeights(source: onlineModel, destination: targetModel);

            float trainingLoss = totalTrainingLoss / Math.Max(1, updateCount);
            float trainingEvMse = totalTrainingEvMse / Math.Max(1, updateCount);

            analysis.NextRow()
                .SetCell("row_type", "step")
                .SetCell("step", step)
                .SetCell("replay_count", replayBuffer.Count)
                .SetCell("added_transitions", addedTransitionCount)
                .SetCell("training_updates", updateCount)
                .SetCell("trained_examples", updateCount * TrainingBatchSize)
                .SetCell("training_loss", trainingLoss.ToString("F6", CultureInfo.InvariantCulture))
                .SetCell("training_ev_mse", trainingEvMse.ToString("F6", CultureInfo.InvariantCulture))
                .SetCell("eval_win_rate", "")
                .SetCell("eval_reward_mean", "")
                .SetCell("elapsed_seconds", experimentStopwatch.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture));

            Console.WriteLine($"  train_loss={trainingLoss:F6} ev_mse={trainingEvMse:F6}");

            if (step % CheckpointInterval != 0)
            {
                File.WriteAllText(analysisCsvPath, analysis.ToString());
                continue;
            }

            (float evalWinRate, float evalRewardMean) = EvaluateAgentMetrics(
                agent: evaluationAgent,
                gameCount: EvaluationGameCount,
                random: random);
            string weightsPath = Path.Combine(weightsDir, $"step{step:D3}.bin");
            onlineModel.save(weightsPath);
            onlineModel.save(Path.Combine(weightsDir, "latest.bin"));

            analysis.NextRow()
                .SetCell("row_type", "checkpoint")
                .SetCell("step", step)
                .SetCell("replay_count", replayBuffer.Count)
                .SetCell("added_transitions", addedTransitionCount)
                .SetCell("training_updates", updateCount)
                .SetCell("trained_examples", step * updateCount * TrainingBatchSize)
                .SetCell("training_loss", trainingLoss.ToString("F6", CultureInfo.InvariantCulture))
                .SetCell("training_ev_mse", trainingEvMse.ToString("F6", CultureInfo.InvariantCulture))
                .SetCell("eval_win_rate", evalWinRate.ToString("F6", CultureInfo.InvariantCulture))
                .SetCell("eval_reward_mean", evalRewardMean.ToString("F6", CultureInfo.InvariantCulture))
                .SetCell("elapsed_seconds", experimentStopwatch.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture))
                .SetCell("weights_path", weightsPath);

            File.WriteAllText(analysisCsvPath, analysis.ToString());
            Console.WriteLine($"  checkpoint eval_win_rate={evalWinRate:F4} eval_reward_mean={evalRewardMean:F4}");
        }

        File.WriteAllText(analysisCsvPath, analysis.ToString());
    }


    static int CollectRollouts(ReplayBuffer replayBuffer, QuantileGreedyPlyOneAgent agent, int gameCount, Random random)
    {
        GameState[] gameStates = new GameState[gameCount];
        int addedTransitionCount = 0;
        for (int gameIndex = 0; gameIndex < gameCount; ++gameIndex)
            gameStates[gameIndex] = CreateStartingState(seed: random.Next());

        while (true)
        {
            bool anyActive = false;
            for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            {
                gameStates[gameIndex].AdvanceToNextPlayerChoice();
                if (!gameStates[gameIndex].GameIsDone)
                    anyActive = true;
            }

            if (!anyActive)
                return addedTransitionCount;

            float[][] policies = agent.GetPolicy(temp: 1f, gameStates);
            for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            {
                GameState gameState = gameStates[gameIndex];
                if (gameState.GameIsDone)
                    continue;

                ReplayState currentState = ReplayState.FromGameState(gameState);
                Move[] legalMoves = gameState.GetMoveOptions();
                int chosenMoveIndex = AgentUtilities.SampleIndex(policies[gameIndex], agent.Random);
                UseHandMove chosenMove = (UseHandMove)legalMoves[chosenMoveIndex];
                chosenMove.Apply(gameState);
                gameState.AdvanceToNextPlayerChoice();

                bool done = gameState.GameIsDone;
                float reward = done ? GetReward(gameState) : 0f;
                ReplayState nextState = done ? default : ReplayState.FromGameState(gameState);
                AddAugmentedTransitions(replayBuffer, currentState, nextState, reward, done);
                addedTransitionCount += SuitPermutations.Length;
            }
        }
    }


    static Tensor BuildTargetQuantiles(QuantilePaddedSwiGLUValueNetwork targetModel, ReplayTransition[] batchTransitions)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        float[,] targetQuantiles = new float[batchTransitions.Length, QuantilePaddedSwiGLUValueNetwork.QuantileCount];
        List<CandidateValueSource> candidateValueSources = [];
        CandidateRange[] candidateRanges = new CandidateRange[batchTransitions.Length];

        for (int transitionIndex = 0; transitionIndex < batchTransitions.Length; ++transitionIndex)
        {
            ReplayTransition transition = batchTransitions[transitionIndex];
            if (transition.Done)
            {
                for (int quantileIndex = 0; quantileIndex < QuantilePaddedSwiGLUValueNetwork.QuantileCount; ++quantileIndex)
                    targetQuantiles[transitionIndex, quantileIndex] = transition.Reward;
                continue;
            }

            GameState nextState = CreateGameStateFromReplayState(transition.NextState);
            Move[] legalMoves = nextState.GetMoveOptions();
            candidateRanges[transitionIndex] = new(candidateValueSources.Count, legalMoves.Length);
            int rootStep = nextState.MoveState.MoveStep;

            for (int moveIndex = 0; moveIndex < legalMoves.Length; ++moveIndex)
            {
                ((UseHandMove)legalMoves[moveIndex]).Apply(nextState);
                nextState.AdvanceToNextPlayerChoice();
                if (nextState.GameIsDone)
                {
                    candidateValueSources.Add(new(
                        IsTerminal: true,
                        TerminalReward: GetReward(nextState),
                        SuccessorState: default));
                }
                else
                {
                    candidateValueSources.Add(new(
                        IsTerminal: false,
                        TerminalReward: 0f,
                        SuccessorState: ReplayState.FromGameState(nextState)));
                }
                nextState.MoveState.RevertToStep(rootStep);
            }
        }

        float[] flatCandidateExpectedValues = new float[candidateValueSources.Count];
        float[,] nonTerminalCandidateQuantiles = new float[candidateValueSources.Count, QuantilePaddedSwiGLUValueNetwork.QuantileCount];
        EvaluateNonTerminalCandidateQuantiles(
            targetModel: targetModel,
            candidateValueSources: candidateValueSources,
            flatCandidateExpectedValues: flatCandidateExpectedValues,
            nonTerminalCandidateQuantiles: nonTerminalCandidateQuantiles);

        for (int candidateIndex = 0; candidateIndex < candidateValueSources.Count; ++candidateIndex)
        {
            CandidateValueSource candidateValueSource = candidateValueSources[candidateIndex];
            if (candidateValueSource.IsTerminal)
            {
                flatCandidateExpectedValues[candidateIndex] = candidateValueSource.TerminalReward;
                continue;
            }
        }

        for (int transitionIndex = 0; transitionIndex < batchTransitions.Length; ++transitionIndex)
        {
            ReplayTransition transition = batchTransitions[transitionIndex];
            if (transition.Done)
                continue;

            CandidateRange candidateRange = candidateRanges[transitionIndex];
            int bestCandidateIndex = candidateRange.Start;
            float bestExpectedValue = flatCandidateExpectedValues[bestCandidateIndex];
            for (int candidateIndex = candidateRange.Start + 1; candidateIndex < candidateRange.Start + candidateRange.Count; ++candidateIndex)
            {
                float candidateExpectedValue = flatCandidateExpectedValues[candidateIndex];
                if (candidateExpectedValue <= bestExpectedValue)
                    continue;

                bestExpectedValue = candidateExpectedValue;
                bestCandidateIndex = candidateIndex;
            }

            CandidateValueSource bestCandidate = candidateValueSources[bestCandidateIndex];
            if (bestCandidate.IsTerminal)
            {
                for (int quantileIndex = 0; quantileIndex < QuantilePaddedSwiGLUValueNetwork.QuantileCount; ++quantileIndex)
                    targetQuantiles[transitionIndex, quantileIndex] = transition.Reward + Gamma * bestCandidate.TerminalReward;
                continue;
            }

            for (int quantileIndex = 0; quantileIndex < QuantilePaddedSwiGLUValueNetwork.QuantileCount; ++quantileIndex)
                targetQuantiles[transitionIndex, quantileIndex] = transition.Reward + Gamma * nonTerminalCandidateQuantiles[bestCandidateIndex, quantileIndex];
        }

        Tensor result = tensor(targetQuantiles, dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice);
        result.MoveToOuterDisposeScope();
        return result;
    }


    static void EvaluateNonTerminalCandidateQuantiles(
        QuantilePaddedSwiGLUValueNetwork targetModel,
        List<CandidateValueSource> candidateValueSources,
        float[] flatCandidateExpectedValues,
        float[,] nonTerminalCandidateQuantiles)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameStateEmbedder successorEmbedder = new(SuccessorEvaluationBatchSize);
        List<int> pendingCandidateIndices = [];
        for (int candidateIndex = 0; candidateIndex < candidateValueSources.Count; ++candidateIndex)
        {
            CandidateValueSource candidateValueSource = candidateValueSources[candidateIndex];
            if (candidateValueSource.IsTerminal)
                continue;

            pendingCandidateIndices.Add(candidateIndex);
            successorEmbedder.AddGameState(CreateGameStateFromReplayState(candidateValueSource.SuccessorState));
            if (pendingCandidateIndices.Count == SuccessorEvaluationBatchSize)
            {
                FlushNonTerminalCandidateQuantiles(
                    targetModel: targetModel,
                    successorEmbedder: successorEmbedder,
                    pendingCandidateIndices: pendingCandidateIndices,
                    flatCandidateExpectedValues: flatCandidateExpectedValues,
                    nonTerminalCandidateQuantiles: nonTerminalCandidateQuantiles);
                successorEmbedder = new(SuccessorEvaluationBatchSize);
            }
        }

        FlushNonTerminalCandidateQuantiles(
            targetModel: targetModel,
            successorEmbedder: successorEmbedder,
            pendingCandidateIndices: pendingCandidateIndices,
            flatCandidateExpectedValues: flatCandidateExpectedValues,
            nonTerminalCandidateQuantiles: nonTerminalCandidateQuantiles);
    }


    static void FlushNonTerminalCandidateQuantiles(
        QuantilePaddedSwiGLUValueNetwork targetModel,
        GameStateEmbedder successorEmbedder,
        List<int> pendingCandidateIndices,
        float[] flatCandidateExpectedValues,
        float[,] nonTerminalCandidateQuantiles)
    {
        if (pendingCandidateIndices.Count == 0)
            return;

        GameStateTensors successorTensors = successorEmbedder.ToTensors(ValueNetwork.EvalDevice);
        Tensor successorQuantilesTensor = targetModel.GetQuantiles(successorTensors).to(CPU);
        float[] flatQuantiles = successorQuantilesTensor.data<float>().ToArray();
        for (int pendingIndex = 0; pendingIndex < pendingCandidateIndices.Count; ++pendingIndex)
        {
            int candidateIndex = pendingCandidateIndices[pendingIndex];
            float total = 0f;
            int quantileOffset = pendingIndex * QuantilePaddedSwiGLUValueNetwork.QuantileCount;
            for (int quantileIndex = 0; quantileIndex < QuantilePaddedSwiGLUValueNetwork.QuantileCount; ++quantileIndex)
            {
                float quantileValue = flatQuantiles[quantileOffset + quantileIndex];
                nonTerminalCandidateQuantiles[candidateIndex, quantileIndex] = quantileValue;
                total += quantileValue;
            }

            flatCandidateExpectedValues[candidateIndex] = total / QuantilePaddedSwiGLUValueNetwork.QuantileCount;
        }

        pendingCandidateIndices.Clear();
    }


    static (float evalWinRate, float evalRewardMean) EvaluateAgentMetrics(QuantileGreedyPlyOneAgent agent, int gameCount, Random random)
    {
        int winCount = 0;
        float totalReward = 0f;
        GameState[] gameStates = new GameState[gameCount];
        for (int gameIndex = 0; gameIndex < gameCount; ++gameIndex)
            gameStates[gameIndex] = CreateStartingState(seed: random.Next());

        while (true)
        {
            bool anyActive = false;
            for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            {
                gameStates[gameIndex].AdvanceToNextPlayerChoice();
                if (!gameStates[gameIndex].GameIsDone)
                    anyActive = true;
            }

            if (!anyActive)
                break;

            agent.MakeMove(temp: 1f, annotatePolicy: false, gameStates);
        }

        for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
        {
            if (gameStates[gameIndex].ScoringState.CurrentRoundTotalChips >= 300f)
                winCount++;
            totalReward += GetReward(gameStates[gameIndex]);
        }

        return (winCount / (float)gameCount, totalReward / gameCount);
    }


    static CSVBuilder CreateConfigAnalysis()
    {
        CSVBuilder analysis = new();
        analysis.NextRow()
            .SetCell("row_type", "config")
            .SetCell("experiment", ExperimentName)
            .SetCell("training_step_count", TrainingStepCount)
            .SetCell("checkpoint_interval", CheckpointInterval)
            .SetCell("rollout_game_count", RolloutGameCount)
            .SetCell("training_batch_size", TrainingBatchSize)
            .SetCell("evaluation_game_count", EvaluationGameCount)
            .SetCell("replay_capacity", ReplayCapacity)
            .SetCell("minimum_replay_size", MinimumReplaySize)
            .SetCell("suit_permutation_count", SuitPermutations.Length)
            .SetCell("replay_reuse_factor", 2)
            .SetCell("target_sync_interval", TargetSyncInterval)
            .SetCell("random_seed", RandomSeed)
            .SetCell("learning_rate", LearningRate.ToString("F6", CultureInfo.InvariantCulture))
            .SetCell("epsilon", Epsilon.ToString("F4", CultureInfo.InvariantCulture))
            .SetCell("gamma", Gamma.ToString("F4", CultureInfo.InvariantCulture))
            .SetCell("score_threshold", ScoreThreshold.ToString("F4", CultureInfo.InvariantCulture))
            .SetCell("quantile_count", QuantilePaddedSwiGLUValueNetwork.QuantileCount)
            .SetCell("quantile_huber_kappa", QuantileHuberKappa.ToString("F4", CultureInfo.InvariantCulture))
            .SetCell("residual_width", ResidualWidth)
            .SetCell("residual_layer_count", ResidualLayerCount)
            .SetCell("starting_hands", GameData.Default.Hands)
            .SetCell("starting_discards", GameData.Default.Discards)
            .SetCell("starting_score", 0f);
        return analysis;
    }


    static TrainingBatch BuildTrainingBatch(ReplayTransition[] batchTransitions)
    {
        long[,] hand = new long[batchTransitions.Length, GameData.HandSize];
        long[] handsAndDiscards = new long[batchTransitions.Length];
        float[,] score = new float[batchTransitions.Length, 1];

        for (int transitionIndex = 0; transitionIndex < batchTransitions.Length; ++transitionIndex)
        {
            ReplayState currentState = batchTransitions[transitionIndex].CurrentState;
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                hand[transitionIndex, cardIndex] = currentState.FullHand[cardIndex];

            handsAndDiscards[transitionIndex] = currentState.HandsAndDiscards;
            score[transitionIndex, 0] = currentState.Score;
        }

        return new(new()
        {
            FullHand = tensor(hand, dtype: ScalarType.Int64, device: ValueNetwork.EvalDevice),
            HandsAndDiscards = tensor(handsAndDiscards, dtype: ScalarType.Int64, device: ValueNetwork.EvalDevice),
            Score = tensor(score, dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice),
        });
    }


    static Tensor GetQuantileDqnLoss(Tensor predictedQuantiles, Tensor targetQuantiles)
    {
        using var scope = NewDisposeScope();

        Tensor taus = (arange(QuantilePaddedSwiGLUValueNetwork.QuantileCount, dtype: ScalarType.Float32, device: predictedQuantiles.device) + 0.5f)
            / QuantilePaddedSwiGLUValueNetwork.QuantileCount;
        Tensor tdErrors = targetQuantiles.unsqueeze(1) - predictedQuantiles.unsqueeze(2);
        Tensor absoluteTdErrors = tdErrors.abs();
        Tensor quadraticPart = absoluteTdErrors.clamp_max(QuantileHuberKappa);
        Tensor linearPart = absoluteTdErrors - quadraticPart;
        Tensor huberLoss = 0.5f * quadraticPart.square() + QuantileHuberKappa * linearPart;
        Tensor indicator = tdErrors.lt(0f).to_type(ScalarType.Float32);
        Tensor quantileWeights = (taus.view(1, QuantilePaddedSwiGLUValueNetwork.QuantileCount, 1) - indicator).abs();
        Tensor loss = (quantileWeights * huberLoss / QuantileHuberKappa).mean();

        loss.MoveToOuterDisposeScope();
        return loss;
    }


    static float GetExpectedValue(float[] flatQuantiles, int candidateIndex)
    {
        int offset = candidateIndex * QuantilePaddedSwiGLUValueNetwork.QuantileCount;
        float total = 0f;
        for (int quantileIndex = 0; quantileIndex < QuantilePaddedSwiGLUValueNetwork.QuantileCount; ++quantileIndex)
            total += flatQuantiles[offset + quantileIndex];
        return total / QuantilePaddedSwiGLUValueNetwork.QuantileCount;
    }


    static float GetReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalChips >= 300f)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalChips / 1000f;
    }


    static void CopyWeights(QuantilePaddedSwiGLUValueNetwork source, QuantilePaddedSwiGLUValueNetwork destination)
    {
        string tempPath = Path.GetTempFileName();
        source.save(tempPath);
        destination.load(tempPath);
        File.Delete(tempPath);
    }


    static GameState CreateStartingState(int seed)
    {
        GameData gameData = new()
        {
            RandomizeSeed = false,
            Seed = seed,
        };

        GameState gameState = new(gameData);
        gameState.AdvanceToNextPlayerChoice();
        new SetCurrentRoundScoreMove(0f).Apply(gameState);
        return gameState;
    }


    static void AddAugmentedTransitions(ReplayBuffer replayBuffer, ReplayState currentState, ReplayState nextState, float reward, bool done)
    {
        for (int permutationIndex = 0; permutationIndex < SuitPermutations.Length; ++permutationIndex)
        {
            replayBuffer.Add(new(
                CurrentState: currentState.Permute(permutationIndex),
                NextState: done ? default : nextState.Permute(permutationIndex),
                Reward: reward,
                Done: done));
        }
    }


    static GameState CreateGameStateFromReplayState(ReplayState replayState)
    {
        GameState gameState = new(CreateReplayGameData());
        new StartRoundMove().Apply(gameState);
        new DrawSpecificHandMove(replayState.ToCards()).Apply(gameState);
        new SetRemainingHandsAndDiscardsMove(
            remainingHands: replayState.RemainingHands,
            remainingDiscards: replayState.RemainingDiscards).Apply(gameState);
        new SetCurrentRoundScoreMove(replayState.Score * 300f).Apply(gameState);
        return gameState;
    }


    static GameData CreateReplayGameData()
    {
        return new()
        {
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


    static string BuildReadme(string commitHash)
    {
        StringBuilder readme = new();
        readme.AppendLine("Date: 2026-04-12");
        readme.AppendLine($"Commit Hash: {commitHash}");
        readme.AppendLine();
        readme.AppendLine("# Training Params");
        readme.AppendLine($"1. Training steps: {TrainingStepCount}");
        readme.AppendLine($"2. Checkpoint interval: {CheckpointInterval}");
        readme.AppendLine($"3. Rollout game count per step: {RolloutGameCount}");
        readme.AppendLine($"4. Training batch size: {TrainingBatchSize}");
        readme.AppendLine($"5. Evaluation game count: {EvaluationGameCount}");
        readme.AppendLine($"6. Replay capacity: {ReplayCapacity}");
        readme.AppendLine($"7. Minimum replay size: {MinimumReplaySize}");
        readme.AppendLine($"8. Suit permutation count: {SuitPermutations.Length}");
        readme.AppendLine($"9. Replay reuse factor: 2");
        readme.AppendLine($"10. Target sync interval: {TargetSyncInterval}");
        readme.AppendLine($"11. Random seed: {RandomSeed}");
        readme.AppendLine($"12. Learning rate: {LearningRate}");
        readme.AppendLine($"13. Epsilon: {Epsilon}");
        readme.AppendLine($"14. Gamma: {Gamma}");
        readme.AppendLine($"15. Score threshold: {ScoreThreshold}");
        readme.AppendLine($"16. Quantile count: {QuantilePaddedSwiGLUValueNetwork.QuantileCount}");
        readme.AppendLine($"17. Quantile Huber kappa: {QuantileHuberKappa}");
        readme.AppendLine($"18. Residual width: {ResidualWidth}");
        readme.AppendLine($"19. Residual layer count: {ResidualLayerCount}");
        readme.AppendLine();
        readme.AppendLine("# Description");
        readme.AppendLine("- Trains a quantile DQN from the standard round start state with default hands and discards.");
        readme.AppendLine("- Uses a new greedy-e ply-1 agent for rollout policy with epsilon 0.1.");
        readme.AppendLine("- Uses the standard reward function: 1 + 0.2 * remaining hands on a cleared blind, else score / 1000.");
        readme.AppendLine("- Augments every collected replay transition by all 24 suit permutations before insertion into replay.");
        readme.AppendLine("- Trains on 2x the post-augmentation transition count each step.");
        readme.AppendLine("- Bootstraps target quantiles from the target network by taking the best successor state under one-step lookahead, while scoring terminal candidate moves by their immediate binary reward.");
        readme.AppendLine("- Logs training loss, EV MSE against Bellman targets, replay size, and evaluation win rate in a single analysis.csv.");
        return readme.ToString();
    }


    static string BuildNotebookJson()
    {
        string oldAnalysisCsvPath = "/Users/miles/Desktop/dev/repos/BalatroAI/Analysis/2026-04-12_quantile_dqn_standardreward_defaultstart_eps0p1_q50_replay500k_suitx24_reuse2_lr3e5_rg64_resume015_memfix/analysis.csv";
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
                        "# Quantile DQN\n",
                        "This notebook loads the live DQN run and plots training loss, EV MSE, and evaluation win rate.\n",
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
                        "analysis_path = Path('analysis.csv')\n",
                        $"old_analysis_path = Path(r'{oldAnalysisCsvPath}')\n",
                        "\n",
                        "def load_rows(path):\n",
                        "    rows = []\n",
                        "    if not path.exists():\n",
                        "        return rows\n",
                        "    with path.open() as f:\n",
                        "        reader = csv.DictReader(f)\n",
                        "        for row in reader:\n",
                        "            rows.append(row)\n",
                        "    return rows\n",
                        "\n",
                        "rows = load_rows(analysis_path)\n",
                        "old_rows = load_rows(old_analysis_path)\n",
                        "checkpoint_rows = [row for row in rows if row['row_type'] == 'checkpoint']\n",
                        "step_rows = [row for row in rows if row['row_type'] == 'step' and row['training_loss']]\n",
                        "old_checkpoint_rows = [row for row in old_rows if row['row_type'] == 'checkpoint']\n",
                        "old_step_rows = [row for row in old_rows if row['row_type'] == 'step' and row['training_loss']]\n",
                        "\n",
                        "print(f'checkpoint rows: {len(checkpoint_rows)}')\n",
                        "if checkpoint_rows:\n",
                        "    print('latest checkpoint:')\n",
                        "    print(checkpoint_rows[-1])\n",
                        "\n",
                        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
                        "\n",
                        "if step_rows:\n",
                        "    steps = [int(row['step']) for row in step_rows]\n",
                        "    train_loss = [float(row['training_loss']) for row in step_rows]\n",
                        "    train_ev_mse = [float(row['training_ev_mse']) for row in step_rows]\n",
                        "    axes[0].plot(steps, train_loss, label='new train quantile loss')\n",
                        "    axes[0].plot(steps, train_ev_mse, label='new train ev mse')\n",
                        "\n",
                        "if old_step_rows:\n",
                        "    old_steps = [int(row['step']) for row in old_step_rows]\n",
                        "    old_train_loss = [float(row['training_loss']) for row in old_step_rows]\n",
                        "    old_train_ev_mse = [float(row['training_ev_mse']) for row in old_step_rows]\n",
                        "    axes[0].plot(old_steps, old_train_loss, alpha=0.45, linestyle='--', label='old train quantile loss')\n",
                        "    axes[0].plot(old_steps, old_train_ev_mse, alpha=0.45, linestyle='--', label='old train ev mse')\n",
                        "\n",
                        "if checkpoint_rows:\n",
                        "    reward_means = [float(row['eval_reward_mean']) for row in checkpoint_rows if row['eval_reward_mean']]\n",
                        "    reward_steps = [int(row['step']) for row in checkpoint_rows if row['eval_reward_mean']]\n",
                        "    win_rates = [float(row['eval_win_rate']) for row in checkpoint_rows if row['eval_win_rate']]\n",
                        "    win_rate_steps = [int(row['step']) for row in checkpoint_rows if row['eval_win_rate']]\n",
                        "    if reward_means:\n",
                        "        axes[1].plot(reward_steps, reward_means, label='new eval avg reward')\n",
                        "    if win_rates:\n",
                        "        axes[1].plot(win_rate_steps, win_rates, alpha=0.3, label='new eval win rate')\n",
                        "\n",
                        "if old_checkpoint_rows:\n",
                        "    old_reward_means = [float(row['eval_reward_mean']) for row in old_checkpoint_rows if row['eval_reward_mean']]\n",
                        "    old_reward_steps = [int(row['step']) for row in old_checkpoint_rows if row['eval_reward_mean']]\n",
                        "    old_win_rates = [float(row['eval_win_rate']) for row in old_checkpoint_rows if row['eval_win_rate']]\n",
                        "    old_win_rate_steps = [int(row['step']) for row in old_checkpoint_rows if row['eval_win_rate']]\n",
                        "    if old_reward_means:\n",
                        "        axes[1].plot(old_reward_steps, old_reward_means, alpha=0.45, linestyle='--', label='old eval avg reward')\n",
                        "    if old_win_rates:\n",
                        "        axes[1].plot(old_win_rate_steps, old_win_rates, alpha=0.2, linestyle='--', label='old eval win rate')\n",
                        "\n",
                        "axes[0].set_title('Training Metrics')\n",
                        "axes[0].set_xlabel('step')\n",
                        "axes[0].grid(True)\n",
                        "axes[0].legend()\n",
                        "axes[1].set_title('Eval Metrics')\n",
                        "axes[1].set_xlabel('step')\n",
                        "axes[1].grid(True)\n",
                        "axes[1].legend()\n",
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


    readonly record struct ReplayState(long[] FullHand, int RemainingHands, int RemainingDiscards, long HandsAndDiscards, float Score)
    {
        public static ReplayState FromGameState(GameState gameState)
        {
            long[] fullHand = new long[GameData.HandSize];
            ReadOnlySpan<Card> hand = gameState.HandState.Hand;
            for (int cardIndex = 0; cardIndex < hand.Length; ++cardIndex)
                fullHand[cardIndex] = hand[cardIndex].ToIndex();

            return new(
                FullHand: fullHand,
                RemainingHands: gameState.HandState.RemainingHands,
                RemainingDiscards: gameState.HandState.RemainingDiscards,
                HandsAndDiscards: gameState.HandState.RemainingHands * 5 + gameState.HandState.RemainingDiscards,
                Score: (float)gameState.ScoringState.CurrentRoundTotalChips / 300f);
        }


        public ReplayState Permute(int permutationIndex)
        {
            int mapOffset = permutationIndex * (Card.RankCount * Card.SuitCount + 1);
            long[] permutedHand = new long[FullHand.Length];
            for (int cardIndex = 0; cardIndex < FullHand.Length; ++cardIndex)
                permutedHand[cardIndex] = SuitPermutationIndexMap[mapOffset + (int)FullHand[cardIndex]];

            return new(
                FullHand: permutedHand,
                RemainingHands: RemainingHands,
                RemainingDiscards: RemainingDiscards,
                HandsAndDiscards: HandsAndDiscards,
                Score: Score);
        }


        public Card[] ToCards()
        {
            List<Card> cards = [];
            for (int cardIndex = 0; cardIndex < FullHand.Length; ++cardIndex)
            {
                Card card = CardFromIndex((int)FullHand[cardIndex]);
                if (!card.IsNull)
                    cards.Add(card);
            }
            return [.. cards];
        }
    }


    readonly record struct ReplayTransition(ReplayState CurrentState, ReplayState NextState, float Reward, bool Done);


    readonly record struct CandidateValueSource(bool IsTerminal, float TerminalReward, ReplayState SuccessorState);


    readonly record struct CandidateRange(int Start, int Count);


    readonly record struct TrainingBatch(GameStateTensors CurrentStates);


    sealed class ReplayBuffer
    {
        readonly ReplayTransition[] _entries;

        int _nextIndex;

        public int Count { get; private set; }

        public ReplayBuffer(int capacity)
        {
            _entries = new ReplayTransition[capacity];
        }


        public void Add(ReplayTransition transition)
        {
            _entries[_nextIndex] = transition;
            _nextIndex = (_nextIndex + 1) % _entries.Length;
            Count = Math.Min(Count + 1, _entries.Length);
        }


        public ReplayTransition[] SampleBatch(int batchSize, Random random)
        {
            ReplayTransition[] batch = new ReplayTransition[batchSize];
            for (int batchIndex = 0; batchIndex < batch.Length; ++batchIndex)
                batch[batchIndex] = _entries[random.Next(Count)];
            return batch;
        }
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


    static Card CardFromIndex(int cardIndex)
    {
        if (cardIndex == 0)
            return Card.Null;

        int zeroBasedIndex = cardIndex - 1;
        int rank = zeroBasedIndex % Card.RankCount + 2;
        Suit suit = (Suit)(zeroBasedIndex / Card.RankCount + 1);
        return new(rank, suit);
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
