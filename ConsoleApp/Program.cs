namespace Ramen.ConsoleApp;

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Ramen.AI;
using Ramen.AgentTools;
using Ramen.Game;
using Ramen.Training;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public static class Program
{
    const string DeckComparisonEvaluationExperimentName = "2026-04-25_simple_flush_ppo_step355_temp0_standard_vs_checkered_eval3000";
    const int DeckComparisonEvaluationGameCount = 3000;
    const string TraceEvaluationExperimentName = "2026-04-23_simple_flush_ppo_step355_temp0_randdeck_trace5_games";
    const int TraceEvaluationGameCount = 5;
    const string OutcomeEvaluationExperimentName = "2026-04-23_simple_flush_ppo_step355_temp0_randdeck_eval10k_outcomes";
    const int OutcomeEvaluationGameCount = 10000;
    const int OutcomeEvaluationBatchSize = 500;
    const int LatestCheckpointEvaluationGameCount = 1000;
    const int TemperatureOneEvaluationGameCount = 300;
    const int TemperatureZeroEvaluationGameCount = 300;
    const int FirstRoundEvaluationGameCount = 10000;
    const string PpoExperimentName = "2026-05-21_ppo_firstround_winhandbonus_trunk768x4_leaf256x1_randdeck52wr_r16384_b1024_e4_lr7e5_clip2_ent0_val0p5_ss40";

    static readonly ExperimentConfig PpoConfig = new(
        ExperimentName: PpoExperimentName,
        RolloutSize: 16384,
        RolloutParallelGameCount: 64,
        BatchSize: 1024,
        TrainingEpochsPerStep: 4,
        StepCount: 200,
        SampledSoftmaxCount: 40,
        LearningRate: 7e-5f,
        AdamBeta1: 0.9f,
        AdamBeta2: 0.97f,
        WeightDecay: 0f,
        PpoEpsilon: 0.2f,
        EntropyCoefficient: 0f,
        ValueLossCoefficient: 0.5f,
        ValueReplayBufferCapacity: 0,
        SnapshotFrequency: 5,
        RandomSeed: 1337,
        InitialHandsPerRound: 4,
        InitialDiscardsPerRound: 3,
        UseRandomDeckInitializer: true,
        TrainOnlyFirstRound: true,
        ResumeSourceExperimentName: "",
        NotebookReferenceExperimentName: "");

    const float PpoContinuationLearningRate = 7e-5f;

    public static void Main(string[] args)
    {
        TensorManager.Init();
        if (args.Length > 0 && args[0] == "ppo-train")
        {
            RunPpoExperiment();
            return;
        }
        if (args.Length > 0 && args[0] == "ppo-eval-latest")
        {
            RunLatestCheckpointRewardEvaluation();
            return;
        }
        if (args.Length > 0 && args[0] == "ppo-eval-latest-temp1-300")
        {
            RunLatestCheckpointTemperatureOneRewardEvaluation();
            return;
        }
        if (args.Length > 0 && args[0] == "ppo-eval-latest-temp0-300")
        {
            RunLatestCheckpointTemperatureZeroRewardEvaluation();
            return;
        }
        if (args.Length > 0 && args[0] == "ppo-eval-latest-temp0-300-buckets")
        {
            RunLatestCheckpointTemperatureZeroBucketEvaluation();
            return;
        }
        if (args.Length > 0 && args[0] == "ppo-eval-latest-first-round-1k")
        {
            RunLatestCheckpointFirstRoundEvaluation();
            return;
        }

        ConsoleBalatroApp app = new();
        app.Run();
    }


    static void RunDeckComparisonEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string commitHash = GetGitCommitHash(repoRoot);
        PreparedOutcomeEvaluationPaths preparedPaths = PrepareDeckComparisonEvaluation(
            repoRoot: repoRoot,
            commitHash: commitHash);

        if (GetEnvironmentFlag("BALATRO_PREPARE_ONLY"))
        {
            Console.WriteLine($"Prepared experiment directory at {preparedPaths.AnalysisDir}");
            return;
        }

        using PpoPolicyValueModel model = new();
        string outcomeEvaluationCheckpointPath = GetOutcomeEvaluationCheckpointPath();
        model.Load(outcomeEvaluationCheckpointPath);

        Stopwatch standardStopwatch = Stopwatch.StartNew();
        OutcomeDistribution standardDistribution = EvaluateOutcomeDistribution(
            model: model,
            gameCount: DeckComparisonEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            gameData: CreateStandardEvaluationGameData());
        standardStopwatch.Stop();

        Stopwatch checkeredStopwatch = Stopwatch.StartNew();
        OutcomeDistribution checkeredDistribution = EvaluateOutcomeDistribution(
            model: model,
            gameCount: DeckComparisonEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            gameData: CreateCheckeredEvaluationGameData());
        checkeredStopwatch.Stop();

        WriteDeckComparisonDistributionCsv(
            filePath: preparedPaths.AnalysisCsvPath,
            standardDistribution: standardDistribution,
            standardWallSeconds: (float)standardStopwatch.Elapsed.TotalSeconds,
            checkeredDistribution: checkeredDistribution,
            checkeredWallSeconds: (float)checkeredStopwatch.Elapsed.TotalSeconds);

        PrintDistribution("standard", standardDistribution, standardStopwatch);
        PrintDistribution("checkered", checkeredDistribution, checkeredStopwatch);
    }


    static PreparedOutcomeEvaluationPaths PrepareDeckComparisonEvaluation(string repoRoot, string commitHash)
    {
        string analysisDir = Path.Combine(repoRoot, "Analysis", DeckComparisonEvaluationExperimentName);
        string readmePath = Path.Combine(analysisDir, "README.md");
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string copiedProgramPath = Path.Combine(analysisDir, "Program.cs");
        string outcomeEvaluationCheckpointPath = GetOutcomeEvaluationCheckpointPath();

        Directory.CreateDirectory(analysisDir);

        string readme = $"""
Date: {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}
Commit Hash: {commitHash}

# Training Params
1. Loaded checkpoint: `{outcomeEvaluationCheckpointPath}`
2. Evaluation games per deck: `{DeckComparisonEvaluationGameCount}`
3. Evaluation batch size: `{OutcomeEvaluationBatchSize}`
4. Policy temperature: `0` / greedy argmax
5. Decks: `standard`, `checkered`

# Description
- This experiment evaluates the step 355 PPO snapshot on standard and checkered decks.
- Checkered means standard rank counts with clubs converted to spades and diamonds converted to hearts.
- `analysis.csv` contains end-state distributions for both deck types.
""";

        File.WriteAllText(readmePath, readme);
        File.Copy(
            sourceFileName: Path.Combine(repoRoot, "ConsoleApp", "Program.cs"),
            destFileName: copiedProgramPath,
            overwrite: true);

        return new(
            AnalysisDir: analysisDir,
            AnalysisCsvPath: analysisCsvPath);
    }


    static void RunTraceEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string commitHash = GetGitCommitHash(repoRoot);
        PreparedTraceEvaluationPaths preparedPaths = PrepareTraceEvaluation(
            repoRoot: repoRoot,
            commitHash: commitHash);

        if (GetEnvironmentFlag("BALATRO_PREPARE_ONLY"))
        {
            Console.WriteLine($"Prepared experiment directory at {preparedPaths.AnalysisDir}");
            return;
        }

        using PpoPolicyValueModel model = new();
        string outcomeEvaluationCheckpointPath = GetOutcomeEvaluationCheckpointPath();
        model.Load(outcomeEvaluationCheckpointPath);

        string traceText = TraceGreedyGames(
            model: model,
            gameCount: TraceEvaluationGameCount);
        File.WriteAllText(preparedPaths.TracePath, traceText);
        Console.Write(traceText);
    }


    static PreparedTraceEvaluationPaths PrepareTraceEvaluation(string repoRoot, string commitHash)
    {
        string analysisDir = Path.Combine(repoRoot, "Analysis", TraceEvaluationExperimentName);
        string readmePath = Path.Combine(analysisDir, "README.md");
        string tracePath = Path.Combine(analysisDir, "trace.md");
        string copiedProgramPath = Path.Combine(analysisDir, "Program.cs");
        string outcomeEvaluationCheckpointPath = GetOutcomeEvaluationCheckpointPath();

        Directory.CreateDirectory(analysisDir);

        string readme = $"""
Date: {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}
Commit Hash: {commitHash}

# Training Params
1. Loaded checkpoint: `{outcomeEvaluationCheckpointPath}`
2. Evaluation games: `{TraceEvaluationGameCount}`
3. Policy temperature: `0` / greedy argmax
4. Deck initializer: `uniform random 52 cards with replacement`
5. Starting hands per round: `4`
6. Starting discards per round: `3`
7. Trace output: [trace.md](trace.md)

# Description
- This trace plays 5 greedy games from the step 355 snapshot on randomized decks.
- For each policy move, it records the pre-move player-choice state, the chosen move, and the immediate post-move state before automatic redraw.
""";

        File.WriteAllText(readmePath, readme);
        File.Copy(
            sourceFileName: Path.Combine(repoRoot, "ConsoleApp", "Program.cs"),
            destFileName: copiedProgramPath,
            overwrite: true);

        return new(
            AnalysisDir: analysisDir,
            TracePath: tracePath);
    }


    static string TraceGreedyGames(PpoPolicyValueModel model, int gameCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        using PolicyNetworkAgent agent = new(model);
        GameData gameData = CreateOutcomeEvaluationGameData();
        StringBuilder trace = new();

        trace.AppendLine("# Step 355 Greedy Trace");
        trace.AppendLine();
        trace.AppendLine($"Checkpoint: `{GetOutcomeEvaluationCheckpointPath()}`");
        trace.AppendLine($"Games: `{gameCount}`");
        trace.AppendLine($"Policy temperature: `0` / greedy argmax");
        trace.AppendLine($"Deck initializer: `uniform random 52 cards with replacement`");
        trace.AppendLine();

        for (int gameIndex = 1; gameIndex <= gameCount; ++gameIndex)
        {
            GameState gameState = new(gameData);
            trace.AppendLine($"## Game {gameIndex}");
            trace.AppendLine();
            trace.AppendLine("Initial full deck:");
            trace.AppendLine($"- By rank: `{SerializeRankCounts(gameState.DeckState.FullDeck)}`");
            trace.AppendLine($"- By suit: `{SerializeSuitCounts(gameState.DeckState.FullDeck)}`");
            trace.AppendLine();

            int moveNumber = 1;
            while (!gameState.GameIsDone)
            {
                gameState.AdvanceToNextPlayerChoice();
                if (gameState.GameIsDone)
                    break;

                string preMoveState = DescribeTraceState(gameState);
                float[][] policies = agent.GetPolicy(temp: 1f, gameState);
                int chosenMoveIndex = GetArgmaxIndex(policies[0]);
                UseHandMove move = AgentUtilities.MoveForPolicyIndex(gameState, chosenMoveIndex);
                string moveText = DescribeTraceMove(gameState, move);
                move.Apply(gameState);
                string postMoveState = DescribeTraceState(gameState);

                trace.AppendLine($"### Move {moveNumber}");
                trace.AppendLine();
                trace.AppendLine($"Pre move: `{preMoveState}`");
                trace.AppendLine($"Move: `{moveText}`");
                trace.AppendLine($"Post move, pre-redraw: `{postMoveState}`");
                trace.AppendLine();

                moveNumber++;
            }

            string outcome = gameState.ScoringState.CurrentRoundTotalScore >= 300f
                ? $"win with {gameState.HandState.RemainingHands} hands remaining"
                : "loss";
            trace.AppendLine($"Final outcome: `{outcome}`");
            trace.AppendLine($"Final state: `{DescribeTraceState(gameState)}`");
            trace.AppendLine();
        }

        return trace.ToString();
    }


    static string DescribeTraceState(GameState gameState)
    {
        return
            $"{gameState} | " +
            $"Stage {gameState.Stage} | " +
            $"Score {gameState.ScoringState.CurrentRoundTotalScore:F1}";
    }


    static string DescribeTraceMove(GameState gameState, UseHandMove move)
    {
        Card[] cards = new Card[move.CardIndices.Length];
        for (int cardIndex = 0; cardIndex < cards.Length; ++cardIndex)
            cards[cardIndex] = gameState.HandState.Hand[move.CardIndices[cardIndex]];

        return $"{(move.IsDiscard ? "Discard" : "Play")} {SerializeCards(cards)}";
    }


    static string SerializeCards(ReadOnlySpan<Card> cards)
    {
        return CardParseUtils.SerializeHand(cards);
    }


    static string SerializeRankCounts(ReadOnlySpan<Card> cards)
    {
        int[] counts = new int[15];
        for (int cardIndex = 0; cardIndex < cards.Length; ++cardIndex)
            counts[cards[cardIndex].Rank]++;

        string[] parts = new string[13];
        for (int rank = 2; rank <= 14; ++rank)
            parts[rank - 2] = $"{CardParseUtils.CharForRank(rank)}:{counts[rank]}";

        return string.Join(", ", parts);
    }


    static string SerializeSuitCounts(ReadOnlySpan<Card> cards)
    {
        int diamondCount = 0;
        int clubCount = 0;
        int heartCount = 0;
        int spadeCount = 0;

        for (int cardIndex = 0; cardIndex < cards.Length; ++cardIndex)
        {
            if (cards[cardIndex].Suit == Suit.Diamond)
                diamondCount++;
            else if (cards[cardIndex].Suit == Suit.Club)
                clubCount++;
            else if (cards[cardIndex].Suit == Suit.Heart)
                heartCount++;
            else if (cards[cardIndex].Suit == Suit.Spade)
                spadeCount++;
        }

        return $"D:{diamondCount}, C:{clubCount}, H:{heartCount}, S:{spadeCount}";
    }


    static void RunOutcomeEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string commitHash = GetGitCommitHash(repoRoot);
        PreparedOutcomeEvaluationPaths preparedPaths = PrepareOutcomeEvaluation(
            repoRoot: repoRoot,
            commitHash: commitHash);

        if (GetEnvironmentFlag("BALATRO_PREPARE_ONLY"))
        {
            Console.WriteLine($"Prepared experiment directory at {preparedPaths.AnalysisDir}");
            return;
        }

        using PpoPolicyValueModel model = new();
        string outcomeEvaluationCheckpointPath = GetOutcomeEvaluationCheckpointPath();
        model.Load(outcomeEvaluationCheckpointPath);

        Stopwatch stopwatch = Stopwatch.StartNew();
        OutcomeDistribution distribution = EvaluateOutcomeDistribution(
            model: model,
            gameCount: OutcomeEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            gameData: CreateOutcomeEvaluationGameData());
        stopwatch.Stop();

        WriteOutcomeDistributionCsv(
            filePath: preparedPaths.AnalysisCsvPath,
            distribution: distribution,
            wallSeconds: (float)stopwatch.Elapsed.TotalSeconds);

        Console.WriteLine($"checkpoint {outcomeEvaluationCheckpointPath}");
        Console.WriteLine($"games {OutcomeEvaluationGameCount}");
        Console.WriteLine($"batch_size {OutcomeEvaluationBatchSize}");
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"loss {distribution.LossCount} ({GetFractionText(distribution.LossCount, distribution.GameCount)})");
        Console.WriteLine($"win_hands_0 {distribution.WinHands0Count} ({GetFractionText(distribution.WinHands0Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_1 {distribution.WinHands1Count} ({GetFractionText(distribution.WinHands1Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_2 {distribution.WinHands2Count} ({GetFractionText(distribution.WinHands2Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_3 {distribution.WinHands3Count} ({GetFractionText(distribution.WinHands3Count, distribution.GameCount)})");
    }


    static void RunLatestCheckpointRewardEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string experimentDir = Path.Combine(repoRoot, "Analysis", PpoConfig.ExperimentName);
        LatestCheckpointInfo checkpoint = FindLatestCheckpointInDirectory(experimentDir);

        using PpoPolicyValueModel model = new();
        model.Load(checkpoint.CheckpointPath);

        Random rolloutRandom = new(PpoConfig.RandomSeed);
        using PpoRolloutDataset rollout = PpoTraining.GenerateRollout(
            model: model,
            config: PpoConfig,
            random: rolloutRandom);

        Stopwatch stopwatch = Stopwatch.StartNew();
        float averageReward = EvaluateAverageReward(
            model: model,
            gameCount: LatestCheckpointEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            temp: 0f,
            gameData: CreateOutcomeEvaluationGameData());
        stopwatch.Stop();

        Console.WriteLine($"checkpoint {checkpoint.CheckpointPath}");
        Console.WriteLine($"experiment {checkpoint.ExperimentName}");
        Console.WriteLine($"rollout_average_reward {rollout.AverageReward.ToString("F6", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"games {LatestCheckpointEvaluationGameCount}");
        Console.WriteLine($"temp 0");
        Console.WriteLine($"average_reward {averageReward.ToString("F6", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
    }


    static void RunLatestCheckpointTemperatureOneRewardEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string experimentDir = Path.Combine(repoRoot, "Analysis", PpoConfig.ExperimentName);
        LatestCheckpointInfo checkpoint = FindLatestCheckpointInDirectory(experimentDir);

        using PpoPolicyValueModel model = new();
        model.Load(checkpoint.CheckpointPath);

        Stopwatch stopwatch = Stopwatch.StartNew();
        float averageReward = EvaluateAverageReward(
            model: model,
            gameCount: TemperatureOneEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            temp: 1f,
            gameData: CreateConsoleEvaluationGameData());
        stopwatch.Stop();

        Console.WriteLine($"checkpoint {checkpoint.CheckpointPath}");
        Console.WriteLine($"experiment {checkpoint.ExperimentName}");
        Console.WriteLine($"games {TemperatureOneEvaluationGameCount}");
        Console.WriteLine($"temp 1");
        Console.WriteLine($"average_reward {averageReward.ToString("F6", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
    }


    static void RunLatestCheckpointTemperatureZeroRewardEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string experimentDir = Path.Combine(repoRoot, "Analysis", PpoConfig.ExperimentName);
        LatestCheckpointInfo checkpoint = FindLatestCheckpointInDirectory(experimentDir);

        using PpoPolicyValueModel model = new();
        model.Load(checkpoint.CheckpointPath);

        Stopwatch stopwatch = Stopwatch.StartNew();
        float averageReward = EvaluateAverageReward(
            model: model,
            gameCount: TemperatureZeroEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            temp: 0f,
            gameData: CreateConsoleEvaluationGameData());
        stopwatch.Stop();

        Console.WriteLine($"checkpoint {checkpoint.CheckpointPath}");
        Console.WriteLine($"experiment {checkpoint.ExperimentName}");
        Console.WriteLine($"games {TemperatureZeroEvaluationGameCount}");
        Console.WriteLine($"temp 0");
        Console.WriteLine($"average_reward {averageReward.ToString("F6", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
    }


    static void RunLatestCheckpointTemperatureZeroBucketEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string experimentDir = Path.Combine(repoRoot, "Analysis", PpoConfig.ExperimentName);
        LatestCheckpointInfo checkpoint = FindLatestCheckpointInDirectory(experimentDir);

        using PpoPolicyValueModel model = new();
        model.Load(checkpoint.CheckpointPath);

        Stopwatch stopwatch = Stopwatch.StartNew();
        float[] rewards = EvaluateRewards(
            model: model,
            gameCount: TemperatureZeroEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            temp: 0f,
            gameData: CreateConsoleEvaluationGameData());
        stopwatch.Stop();

        float rewardSum = 0f;
        for (int rewardIndex = 0; rewardIndex < rewards.Length; ++rewardIndex)
            rewardSum += rewards[rewardIndex];

        Console.WriteLine($"checkpoint {checkpoint.CheckpointPath}");
        Console.WriteLine($"experiment {checkpoint.ExperimentName}");
        Console.WriteLine($"games {TemperatureZeroEvaluationGameCount}");
        Console.WriteLine($"temp 0");
        Console.WriteLine($"average_reward {(rewardSum / rewards.Length).ToString("F6", CultureInfo.InvariantCulture)}");
        Console.WriteLine("reward_buckets");
        WriteRewardBuckets(rewards, bucketSize: 5);
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
    }


    static void RunLatestCheckpointFirstRoundEvaluation()
    {
        string repoRoot = FindRepoRoot();
        string experimentDir = Path.Combine(repoRoot, "Analysis", PpoConfig.ExperimentName);
        LatestCheckpointInfo checkpoint = FindLatestCheckpointInDirectory(experimentDir);

        using PpoPolicyValueModel model = new();
        model.Load(checkpoint.CheckpointPath);

        Stopwatch stopwatch = Stopwatch.StartNew();
        int[] rewardCounts = EvaluateFirstRoundRewards(
            model: model,
            gameCount: FirstRoundEvaluationGameCount,
            batchSize: OutcomeEvaluationBatchSize,
            gameData: CreateConsoleEvaluationGameData());
        stopwatch.Stop();

        int rewardSum = 0;
        int completedCount = 0;
        for (int reward = 0; reward < rewardCounts.Length; ++reward)
        {
            rewardSum += reward * rewardCounts[reward];
            completedCount += rewardCounts[reward];
        }

        Console.WriteLine($"checkpoint {checkpoint.CheckpointPath}");
        Console.WriteLine($"experiment {checkpoint.ExperimentName}");
        Console.WriteLine($"games {FirstRoundEvaluationGameCount}");
        Console.WriteLine("temp 0");
        Console.WriteLine("reward_function first_round_win + remaining_hands");
        Console.WriteLine($"average_reward {((float)rewardSum / completedCount).ToString("F6", CultureInfo.InvariantCulture)}");
        Console.WriteLine("reward_buckets");
        for (int reward = 0; reward < rewardCounts.Length; ++reward)
        {
            float fraction = (float)rewardCounts[reward] / completedCount;
            Console.WriteLine($"{reward}: {rewardCounts[reward],4} {fraction.ToString("P1", CultureInfo.InvariantCulture)}");
        }
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
    }


    static PreparedOutcomeEvaluationPaths PrepareOutcomeEvaluation(string repoRoot, string commitHash)
    {
        string analysisDir = Path.Combine(repoRoot, "Analysis", OutcomeEvaluationExperimentName);
        string readmePath = Path.Combine(analysisDir, "README.md");
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string copiedProgramPath = Path.Combine(analysisDir, "Program.cs");
        string outcomeEvaluationCheckpointPath = GetOutcomeEvaluationCheckpointPath();

        Directory.CreateDirectory(analysisDir);

        string readme = $"""
Date: {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}
Commit Hash: {commitHash}

# Training Params
1. Loaded checkpoint: `{outcomeEvaluationCheckpointPath}`
2. Evaluation games: `{OutcomeEvaluationGameCount}`
3. Evaluation batch size: `{OutcomeEvaluationBatchSize}`
4. Policy temperature: `0` / greedy argmax
5. Deck initializer: `uniform random 52 cards with replacement`
6. Starting hands per round: `4`
7. Starting discards per round: `3`

# Description
- This experiment evaluates the step 355 PPO snapshot with greedy policy selection on randomized decks.
- `analysis.csv` contains the final end-state distribution: losses, and wins grouped by remaining hands.
""";

        File.WriteAllText(readmePath, readme);
        File.Copy(
            sourceFileName: Path.Combine(repoRoot, "ConsoleApp", "Program.cs"),
            destFileName: copiedProgramPath,
            overwrite: true);

        return new(
            AnalysisDir: analysisDir,
            AnalysisCsvPath: analysisCsvPath);
    }


    static string GetOutcomeEvaluationCheckpointPath()
    {
        string repoRoot = FindRepoRoot();
        LatestCheckpointInfo checkpoint = FindLatestPpoCheckpoint(repoRoot);
        return checkpoint.CheckpointPath;
    }


    static OutcomeDistribution EvaluateOutcomeDistribution(PpoPolicyValueModel model, int gameCount, int batchSize, GameData gameData)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        using PolicyNetworkAgent agent = new(model);
        GameState[] gameStates = new GameState[batchSize];
        int gamesStarted = 0;
        OutcomeDistribution distribution = new(GameCount: gameCount);

        for (int slot = 0; slot < gameStates.Length; ++slot)
        {
            if (gamesStarted >= gameCount)
                break;

            gameStates[slot] = new(gameData);
            gamesStarted++;
        }

        while (distribution.CompletedCount < gameCount)
        {
            List<int> activeSlots = [];
            for (int slot = 0; slot < gameStates.Length; ++slot)
            {
                GameState gameState = gameStates[slot];
                if (gameState == null)
                    continue;

                gameState.AdvanceToNextPlayerChoice();
                if (gameState.GameIsDone)
                {
                    distribution = AddOutcome(distribution, gameState);

                    if (gamesStarted < gameCount)
                    {
                        gameStates[slot] = new(gameData);
                        gamesStarted++;
                    }
                    else
                    {
                        gameStates[slot] = null;
                    }

                    continue;
                }

                activeSlots.Add(slot);
            }

            if (activeSlots.Count == 0)
                continue;

            GameState[] activeStates = new GameState[activeSlots.Count];
            for (int index = 0; index < activeSlots.Count; ++index)
                activeStates[index] = gameStates[activeSlots[index]];

            float[][] policies = agent.GetPolicy(temp: 1f, activeStates);
            for (int index = 0; index < activeSlots.Count; ++index)
            {
                int chosenMoveIndex = GetArgmaxIndex(policies[index]);
                AgentUtilities.MoveForPolicyIndex(activeStates[index], chosenMoveIndex).Apply(activeStates[index]);
            }
        }

        return distribution;
    }


    static float EvaluateAverageReward(PpoPolicyValueModel model, int gameCount, int batchSize, float temp, GameData gameData)
    {
        float[] rewards = EvaluateRewards(
            model: model,
            gameCount: gameCount,
            batchSize: batchSize,
            temp: temp,
            gameData: gameData);
        float rewardSum = 0f;
        for (int rewardIndex = 0; rewardIndex < rewards.Length; ++rewardIndex)
            rewardSum += rewards[rewardIndex];

        return rewardSum / gameCount;
    }


    static float[] EvaluateRewards(PpoPolicyValueModel model, int gameCount, int batchSize, float temp, GameData gameData)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        using PolicyNetworkAgent agent = new(model);
        GameState[] gameStates = new GameState[gameCount];
        for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            gameStates[gameIndex] = new(gameData);

        while (true)
        {
            bool allGamesDone = true;
            for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            {
                if (!agent.IsGameDone(gameStates[gameIndex]))
                {
                    allGamesDone = false;
                    break;
                }
            }

            if (allGamesDone)
                break;

            if (temp <= 0f)
                MakeGreedyConsoleAutoMoves(model, agent, gameStates, batchSize);
            else
                agent.MakeMove(temp: temp, annotatePolicy: false, gameStates);
        }

        float[] rewards = new float[gameCount];
        for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            rewards[gameIndex] = PpoTraining.GetStandardReward(gameStates[gameIndex]);

        return rewards;
    }


    static int[] EvaluateFirstRoundRewards(PpoPolicyValueModel model, int gameCount, int batchSize, GameData gameData)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        using PolicyNetworkAgent agent = new(model);
        GameState[] gameStates = new GameState[gameCount];
        for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            gameStates[gameIndex] = new(gameData);

        while (true)
        {
            bool allGamesDone = true;
            for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
            {
                GameState gameState = gameStates[gameIndex];
                if (!IsFirstRoundEvaluationDone(gameState))
                {
                    allGamesDone = false;
                    break;
                }
            }

            if (allGamesDone)
                break;

            MakeGreedyFirstRoundMoves(agent, gameStates, batchSize);
        }

        int[] rewardCounts = new int[5];
        for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
        {
            int reward = GetFirstRoundReward(gameStates[gameIndex]);
            rewardCounts[reward]++;
        }

        return rewardCounts;
    }


    static bool IsFirstRoundEvaluationDone(GameState gameState)
    {
        return gameState.GameIsDone || gameState.Stage == StageOfGame.EndRound;
    }


    static int GetFirstRoundReward(GameState gameState)
    {
        if (gameState.Stage != StageOfGame.EndRound)
            return 0;

        return 1 + gameState.HandState.RemainingHands;
    }


    static void WriteRewardBuckets(float[] rewards, int bucketSize)
    {
        SortedDictionary<int, int> bucketCounts = new();
        for (int rewardIndex = 0; rewardIndex < rewards.Length; ++rewardIndex)
        {
            int bucketStart = (int)MathF.Floor(rewards[rewardIndex] / bucketSize) * bucketSize;
            if (!bucketCounts.TryAdd(bucketStart, 1))
                bucketCounts[bucketStart]++;
        }

        foreach (KeyValuePair<int, int> bucketCount in bucketCounts)
        {
            int bucketEnd = bucketCount.Key + bucketSize;
            float fraction = (float)bucketCount.Value / rewards.Length;
            Console.WriteLine($"{bucketCount.Key,4}-{bucketEnd,4}: {bucketCount.Value,4} {fraction.ToString("P1", CultureInfo.InvariantCulture)}");
        }
    }


    static void MakeGreedyConsoleAutoMoves(PpoPolicyValueModel model, IAgent agent, GameState[] gameStates, int batchSize)
    {
        List<int> roundIndices = [];
        List<int> shopIndices = [];
        for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
        {
            GameState gameState = gameStates[gameIndex];
            while (!gameState.GameIsDone &&
                gameState.Stage != StageOfGame.InRoundPlayerChoice &&
                gameState.Stage != StageOfGame.InShop)
            {
                gameState.GetMoveOptions()[0].Apply(gameState);
            }

            if (gameState.GameIsDone)
                continue;
            if (gameState.Stage == StageOfGame.InRoundPlayerChoice)
                roundIndices.Add(gameIndex);
            else if (gameState.Stage == StageOfGame.InShop)
                shopIndices.Add(gameIndex);
        }

        ApplyGreedyRoundMoves(agent, gameStates, roundIndices, batchSize);
        ApplyGreedyShopMoves(model, gameStates, shopIndices, batchSize);
    }


    static void MakeGreedyFirstRoundMoves(IAgent agent, GameState[] gameStates, int batchSize)
    {
        List<int> roundIndices = [];
        for (int gameIndex = 0; gameIndex < gameStates.Length; ++gameIndex)
        {
            GameState gameState = gameStates[gameIndex];
            while (!IsFirstRoundEvaluationDone(gameState) && gameState.Stage != StageOfGame.InRoundPlayerChoice)
                gameState.GetMoveOptions()[0].Apply(gameState);

            if (!IsFirstRoundEvaluationDone(gameState))
                roundIndices.Add(gameIndex);
        }

        ApplyGreedyRoundMoves(agent, gameStates, roundIndices, batchSize);
    }


    static void ApplyGreedyRoundMoves(IAgent agent, GameState[] gameStates, List<int> roundIndices, int batchSize)
    {
        for (int batchStart = 0; batchStart < roundIndices.Count; batchStart += batchSize)
        {
            int activeBatchSize = Math.Min(batchSize, roundIndices.Count - batchStart);
            GameState[] activeStates = new GameState[activeBatchSize];
            for (int batchIndex = 0; batchIndex < activeStates.Length; ++batchIndex)
                activeStates[batchIndex] = gameStates[roundIndices[batchStart + batchIndex]];

            float[][] policies = agent.GetPolicy(temp: 1f, activeStates);
            for (int batchIndex = 0; batchIndex < activeStates.Length; ++batchIndex)
            {
                int chosenMoveIndex = GetArgmaxIndex(policies[batchIndex]);
                AgentUtilities.MoveForPolicyIndex(activeStates[batchIndex], chosenMoveIndex).Apply(activeStates[batchIndex]);
            }
        }
    }


    static void ApplyGreedyShopMoves(PpoPolicyValueModel model, GameState[] gameStates, List<int> shopIndices, int batchSize)
    {
        for (int batchStart = 0; batchStart < shopIndices.Count; batchStart += batchSize)
        {
            using var scope = NewDisposeScope();

            int activeBatchSize = Math.Min(batchSize, shopIndices.Count - batchStart);
            GameState[] activeStates = new GameState[activeBatchSize];
            for (int batchIndex = 0; batchIndex < activeStates.Length; ++batchIndex)
                activeStates[batchIndex] = gameStates[shopIndices[batchStart + batchIndex]];

            GameStateEmbedder embedder = new(activeStates.Length);
            for (int stateIndex = 0; stateIndex < activeStates.Length; ++stateIndex)
                embedder.AddGameState(activeStates[stateIndex]);

            GameStateTensors gameStateTensors = embedder.ToTensors(PpoPolicyValueModel.EvalDevice);
            Tensor logits = model.GetStorePolicyLogits(gameStateTensors);
            Tensor illegalMask = BuildStoreMask(gameStateTensors);
            Tensor selectedIndices = (logits + illegalMask).argmax(1).to(CPU);
            long[] selectedIndexData = selectedIndices.data<long>().ToArray();
            gameStateTensors.Dispose();

            for (int stateIndex = 0; stateIndex < activeStates.Length; ++stateIndex)
                ApplyStoreMove(activeStates[stateIndex], (int)selectedIndexData[stateIndex]);
        }
    }


    static Tensor BuildStoreMask(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Device device = PpoPolicyValueModel.EvalDevice;
        Tensor money = gameStateTensors.Money.to(device).to_type(ScalarType.Int64);
        Tensor rerollPrice = gameStateTensors.RerollPrice.to(device).to_type(ScalarType.Int64);
        Tensor storeJokers = gameStateTensors.StoreJokers.to(device).to_type(ScalarType.Int64);
        Tensor storePrices = gameStateTensors.StorePrices.to(device).to_type(ScalarType.Int64);
        Tensor ownedJokers = gameStateTensors.OwnedJokers.to(device).to_type(ScalarType.Int64);

        Tensor exitMask = zeros([money.size(0), 1], dtype: ScalarType.Float32, device: device);
        Tensor rerollMask = money.lt(rerollPrice).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor nullStoreMask = storeJokers.eq(0).to_type(ScalarType.Float32);
        Tensor unaffordableStoreMask = money.unsqueeze(-1).lt(storePrices).to_type(ScalarType.Float32);
        Tensor ownedJokerCount = ownedJokers.ne(0).to_type(ScalarType.Int64).sum(dim: 1, keepdim: true);
        Tensor rosterFullMask = ownedJokerCount.ge(GameStateEmbedder.MaxOwnedJokerCount).to_type(ScalarType.Float32).expand_as(nullStoreMask);
        Tensor storeMask = (nullStoreMask + unaffordableStoreMask + rosterFullMask).clamp(0f, 1f);
        Tensor stacked = cat([exitMask, rerollMask, storeMask], dim: 1) * -1e9f;
        stacked.MoveToOuterDisposeScope();
        return stacked;
    }


    static void ApplyStoreMove(GameState gameState, int selectedIndex)
    {
        Move move = selectedIndex switch
        {
            1 => new RerollMove(),
            2 => new BuyShopOfferMove(0),
            3 => new BuyShopOfferMove(1),
            _ => new ExitShopMove(),
        };
        move.Apply(gameState);
    }


    static GameData CreateOutcomeEvaluationGameData()
    {
        GameData gameData = new()
        {
            Seed = GameData.Default.Seed,
            Hands = 4,
            Discards = 3,
            RandomizeSeed = GameData.Default.RandomizeSeed,
            StartingHandBaseScore = [.. GameData.Default.StartingHandBaseScore],
            PlanetScores = [.. GameData.Default.PlanetScores],
            InitStartingDeck = GameData.InitErraticStartingDeck,
        };

        return gameData;
    }


    static GameData CreateConsoleEvaluationGameData()
    {
        return new();
    }


    static GameData CreateStandardEvaluationGameData()
    {
        GameData gameData = new()
        {
            Seed = GameData.Default.Seed,
            Hands = 4,
            Discards = 3,
            RandomizeSeed = GameData.Default.RandomizeSeed,
            StartingHandBaseScore = [.. GameData.Default.StartingHandBaseScore],
            PlanetScores = [.. GameData.Default.PlanetScores],
            InitStartingDeck = GameData.Default.InitStartingDeck,
        };

        return gameData;
    }


    static GameData CreateCheckeredEvaluationGameData()
    {
        GameData gameData = new()
        {
            Seed = GameData.Default.Seed,
            Hands = 4,
            Discards = 3,
            RandomizeSeed = GameData.Default.RandomizeSeed,
            StartingHandBaseScore = [.. GameData.Default.StartingHandBaseScore],
            PlanetScores = [.. GameData.Default.PlanetScores],
            InitStartingDeck = InitializeCheckeredDeck,
        };

        return gameData;
    }


    static void InitializeCheckeredDeck(GameState gameState)
    {
        for (int rank = 2; rank <= 14; ++rank)
        {
            AddCardToFullDeck(gameState, new(rank, Suit.Spade));
            AddCardToFullDeck(gameState, new(rank, Suit.Spade));
            AddCardToFullDeck(gameState, new(rank, Suit.Heart));
            AddCardToFullDeck(gameState, new(rank, Suit.Heart));
        }
    }


    static void AddCardToFullDeck(GameState gameState, Card card)
    {
        System.Reflection.MethodInfo method = typeof(DeckState).GetMethod(
            "AddCardToFullDeck",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        method.Invoke(gameState.DeckState, [card]);
    }


    static OutcomeDistribution AddOutcome(OutcomeDistribution distribution, GameState gameState)
    {
        int lossCount = distribution.LossCount;
        int winHands0Count = distribution.WinHands0Count;
        int winHands1Count = distribution.WinHands1Count;
        int winHands2Count = distribution.WinHands2Count;
        int winHands3Count = distribution.WinHands3Count;

        if (gameState.ScoringState.CurrentRoundTotalScore < 300f)
        {
            lossCount++;
        }
        else if (gameState.HandState.RemainingHands == 0)
        {
            winHands0Count++;
        }
        else if (gameState.HandState.RemainingHands == 1)
        {
            winHands1Count++;
        }
        else if (gameState.HandState.RemainingHands == 2)
        {
            winHands2Count++;
        }
        else if (gameState.HandState.RemainingHands == 3)
        {
            winHands3Count++;
        }

        return distribution with
        {
            CompletedCount = distribution.CompletedCount + 1,
            LossCount = lossCount,
            WinHands0Count = winHands0Count,
            WinHands1Count = winHands1Count,
            WinHands2Count = winHands2Count,
            WinHands3Count = winHands3Count,
        };
    }


    static void WriteOutcomeDistributionCsv(string filePath, OutcomeDistribution distribution, float wallSeconds)
    {
        string csv = $"""
category,count,fraction,games_completed,wall_seconds
loss,{distribution.LossCount},{GetFractionText(distribution.LossCount, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}
win_hands_0,{distribution.WinHands0Count},{GetFractionText(distribution.WinHands0Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}
win_hands_1,{distribution.WinHands1Count},{GetFractionText(distribution.WinHands1Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}
win_hands_2,{distribution.WinHands2Count},{GetFractionText(distribution.WinHands2Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}
win_hands_3,{distribution.WinHands3Count},{GetFractionText(distribution.WinHands3Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}
""";

        File.WriteAllText(filePath, csv);
    }


    static void WriteDeckComparisonDistributionCsv(
        string filePath,
        OutcomeDistribution standardDistribution,
        float standardWallSeconds,
        OutcomeDistribution checkeredDistribution,
        float checkeredWallSeconds)
    {
        StringBuilder csv = new();
        csv.AppendLine("deck,category,count,fraction,games_completed,wall_seconds");
        AppendDeckDistributionRows(csv, "standard", standardDistribution, standardWallSeconds);
        AppendDeckDistributionRows(csv, "checkered", checkeredDistribution, checkeredWallSeconds);
        File.WriteAllText(filePath, csv.ToString());
    }


    static void AppendDeckDistributionRows(StringBuilder csv, string deckName, OutcomeDistribution distribution, float wallSeconds)
    {
        csv.AppendLine($"{deckName},loss,{distribution.LossCount},{GetFractionText(distribution.LossCount, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}");
        csv.AppendLine($"{deckName},win_hands_0,{distribution.WinHands0Count},{GetFractionText(distribution.WinHands0Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}");
        csv.AppendLine($"{deckName},win_hands_1,{distribution.WinHands1Count},{GetFractionText(distribution.WinHands1Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}");
        csv.AppendLine($"{deckName},win_hands_2,{distribution.WinHands2Count},{GetFractionText(distribution.WinHands2Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}");
        csv.AppendLine($"{deckName},win_hands_3,{distribution.WinHands3Count},{GetFractionText(distribution.WinHands3Count, distribution.GameCount)},{distribution.CompletedCount},{wallSeconds.ToString("F4", CultureInfo.InvariantCulture)}");
    }


    static void PrintDistribution(string deckName, OutcomeDistribution distribution, Stopwatch stopwatch)
    {
        Console.WriteLine($"deck {deckName}");
        Console.WriteLine($"games {distribution.GameCount}");
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"loss {distribution.LossCount} ({GetFractionText(distribution.LossCount, distribution.GameCount)})");
        Console.WriteLine($"win_hands_0 {distribution.WinHands0Count} ({GetFractionText(distribution.WinHands0Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_1 {distribution.WinHands1Count} ({GetFractionText(distribution.WinHands1Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_2 {distribution.WinHands2Count} ({GetFractionText(distribution.WinHands2Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_3 {distribution.WinHands3Count} ({GetFractionText(distribution.WinHands3Count, distribution.GameCount)})");
    }


    static string GetFractionText(int count, int total)
    {
        float fraction = total == 0 ? 0f : (float)count / total;
        return fraction.ToString("F6", CultureInfo.InvariantCulture);
    }


    static int GetArgmaxIndex(float[] values)
    {
        int bestIndex = 0;
        float bestValue = values[0];
        for (int index = 1; index < values.Length; ++index)
        {
            if (values[index] <= bestValue)
                continue;

            bestValue = values[index];
            bestIndex = index;
        }

        return bestIndex;
    }


    static int SampleMoveIndex(ReadOnlySpan<float> probs, Random random)
    {
        float sample = (float)random.NextDouble();
        float cumulative = 0f;

        for (int moveIndex = 0; moveIndex < probs.Length; ++moveIndex)
        {
            cumulative += probs[moveIndex];
            if (sample <= cumulative)
                return moveIndex;
        }

        return probs.Length - 1;
    }


    static void RunPpoExperiment()
    {
        string repoRoot = FindRepoRoot();
        string commitHash = GetGitCommitHash(repoRoot);
        ResumeConfig resume = GetResumeConfig(repoRoot, PpoConfig);
        PreparedExperimentPaths preparedPaths = PrepareExperiment(
            repoRoot: repoRoot,
            commitHash: commitHash,
            config: PpoConfig,
            resume: resume);

        if (GetEnvironmentFlag("BALATRO_PREPARE_ONLY"))
        {
            Console.WriteLine($"Prepared experiment directory at {preparedPaths.AnalysisDir}");
            return;
        }

        List<StepMetrics> metrics = [];
        float priorWallClockSeconds = 0f;
        Stopwatch stopwatch = Stopwatch.StartNew();
        using PpoPolicyValueModel model = new();
        if (!string.IsNullOrWhiteSpace(resume.CheckpointPath))
            model.Load(resume.CheckpointPath);

        using AdamW optimizer = optim.AdamW(
            parameters: model.parameters(),
            lr: PpoConfig.LearningRate,
            weight_decay: PpoConfig.WeightDecay,
            beta1: PpoConfig.AdamBeta1,
            beta2: PpoConfig.AdamBeta2);
        Random rolloutRandom = new(PpoConfig.RandomSeed);
        Random shuffleRandom = new(PpoConfig.RandomSeed + 1);

        for (int step = resume.ResumeStep + 1; step <= PpoConfig.StepCount; ++step)
        {
            int continuationStep = step - resume.ResumeStep;
            float learningRate = PpoTraining.GetLearningRate(continuationStep, PpoContinuationLearningRate);
            PpoTraining.SetOptimizerLearningRate(optimizer, learningRate);

            using PpoRolloutDataset rollout = PpoTraining.GenerateRollout(
                model: model,
                config: PpoConfig,
                random: rolloutRandom);
            TrainingMetrics trainingMetrics = PpoTraining.TrainStep(
                model: model,
                optimizer: optimizer,
                rollout: rollout,
                shuffleRandom: shuffleRandom,
                config: PpoConfig);
            StepMetrics stepMetrics = new(
                Step: step,
                WallClockSeconds: priorWallClockSeconds + GetElapsedSeconds(stopwatch),
                AverageReward: rollout.AverageReward,
                AverageMoveEntropy: rollout.AverageMoveEntropy,
                ValueMseMean: trainingMetrics.ValueMseMean,
                PolicyLossMean: trainingMetrics.PolicyLossMean,
                ClipFractionMean: trainingMetrics.ClipFractionMean,
                CompletedGameCount: rollout.CompletedGameCount,
                LearningRate: learningRate,
                ValueReplayCount: 0);
            metrics.Add(stepMetrics);
            WriteRewardCsv(preparedPaths.AnalysisCsvPath, metrics);

            Console.WriteLine(
                $"step {step}/{PpoConfig.StepCount} | " +
                $"wall {stepMetrics.WallClockSeconds:F2}s | " +
                $"reward {stepMetrics.AverageReward:F4} | " +
                $"entropy {stepMetrics.AverageMoveEntropy:F4} | " +
                $"value_mse {stepMetrics.ValueMseMean:F4} | " +
                $"policy_loss {stepMetrics.PolicyLossMean:F4} | " +
                $"clip_frac {stepMetrics.ClipFractionMean:F4} | " +
                $"lr {stepMetrics.LearningRate:0.0e0} | " +
                $"completed_games {stepMetrics.CompletedGameCount}");

            if (step % PpoConfig.SnapshotFrequency == 0 || step == PpoConfig.StepCount)
            {
                string checkpointPath = Path.Combine(
                    preparedPaths.LocalWeightsDir,
                    $"{step.ToString(CultureInfo.InvariantCulture)}.bin");
                model.Save(checkpointPath);
            }

            TensorManager.DisposeAll();
            GC.Collect();
        }
    }


    static void WriteRewardCsv(string filePath, IReadOnlyList<StepMetrics> metrics)
    {
        CSVBuilder output = new();
        for (int metricIndex = 0; metricIndex < metrics.Count; ++metricIndex)
        {
            StepMetrics metric = metrics[metricIndex];
            output
                .NextRow()
                .SetCell("average_reward", metric.AverageReward);
        }

        File.WriteAllText(filePath, output.ToString());
    }


    static PreparedTraceExperimentPaths PrepareTraceExperiment(string repoRoot, string commitHash, LatestCheckpointInfo checkpoint, TraceExperimentConfig config)
    {
        string analysisDir = Path.Combine(repoRoot, "Analysis", config.ExperimentName);
        string readmePath = Path.Combine(analysisDir, "README.md");
        string notebookPath = Path.Combine(analysisDir, "analysis.ipynb");
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string copiedProgramPath = Path.Combine(analysisDir, "Program.cs");

        Directory.CreateDirectory(analysisDir);

        WriteTraceReadme(
            filePath: readmePath,
            commitHash: commitHash,
            checkpoint: checkpoint,
            config: config);
        WriteTraceNotebook(
            filePath: notebookPath,
            csvPath: analysisCsvPath);
        File.Copy(
            sourceFileName: Path.Combine(repoRoot, "ConsoleApp", "Program.cs"),
            destFileName: copiedProgramPath,
            overwrite: true);

        return new(
            AnalysisDir: analysisDir,
            AnalysisCsvPath: analysisCsvPath);
    }


    static void WriteTraceReadme(string filePath, string commitHash, LatestCheckpointInfo checkpoint, TraceExperimentConfig config)
    {
        string readme = $"""
Date: {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}
Commit Hash: {commitHash}

# Training Params
1. Loaded checkpoint: `{checkpoint.CheckpointPath}`
2. Source experiment: `{checkpoint.ExperimentName}`
3. Top move trajectory count: `{config.PolicyTopMoveTrajectoryCount}`
4. Top move count considered by policy rank: `{config.AdditionalTopMoveCount + 1}`
5. Initial trajectories for non-top moves: `{config.InitialOtherMoveTrajectoryCount}`
6. Additional trajectories per surviving move each round: `{config.AdditionalTrajectoryCountPerRound}`
7. Max trajectories per real move: `{config.MaxTrajectoryCount}`
8. Entropy limit per simulated continuation: `{config.MaxEntropy}`
9. Max turns traced: `{config.MaxTurnsToTrace}`
10. Notebook: [analysis.ipynb](analysis.ipynb)

# Description
- This run traces the first real move from a trajectory-pruning agent initialized from the latest saved checkpoint.
- The CSV includes both candidate-summary rows and raw per-trajectory value rows for one lowest-std move and one highest-std move from the traced turn.
- This diagnostic is intended to explain whether near-zero reported trajectory standard deviations are real or just formatting artifacts.
""";

        File.WriteAllText(filePath, readme);
    }


    static void WriteTraceNotebook(string filePath, string csvPath)
    {
        Dictionary<string, object> notebook = [];
        List<Dictionary<string, object>> cells = [];

        cells.Add(CreateMarkdownCell("""
# Trajectory Pruning Agent Trace

This notebook loads `analysis.csv`, prints the ranked candidate table for each traced turn, and shows raw trajectory values for the debug-labeled moves.
"""));

        string notebookCode = """
from pathlib import Path
import csv

csv_path = Path(r"__CSV_PATH__")
rows = []
with csv_path.open("r", newline="") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        parsed_row = dict()
        for key, value in row.items():
            if value is None or value == "":
                parsed_row[key] = None
                continue
            try:
                parsed_row[key] = float(value)
            except ValueError:
                parsed_row[key] = value
        rows.append(parsed_row)

summary_rows = [row for row in rows if row["row_type"] == "summary"]
raw_rows = [row for row in rows if row["row_type"] == "raw_trajectory"]

turn_to_rows = dict()
for row in summary_rows:
    turn_index = int(row["turn_index"])
    if turn_index not in turn_to_rows:
        turn_to_rows[turn_index] = []
    turn_to_rows[turn_index].append(row)

for turn_index in sorted(turn_to_rows.keys()):
    turn_rows = sorted(turn_to_rows[turn_index], key=lambda row: int(row["original_policy_rank"]))
    chosen_row = next(row for row in turn_rows if int(row["is_selected"]) == 1)
    print(f"Turn {turn_index}")
    print(f"State: {chosen_row['state_text']}")
    print(f"Chosen: {chosen_row['move_text']} | top: {chosen_row['top_move_text']} | seconds: {chosen_row['move_search_seconds']:.2f}")
    print("Rank | Considered | Chosen | Samples | Mean | Std | Move")
    for row in turn_rows:
        print(
            f"{int(row['original_policy_rank']):>4} | "
            f"{int(row['is_considered'])!s:>10} | "
            f"{int(row['is_selected'])!s:>6} | "
            f"{int(row['trajectory_count']):>7} | "
            f"{row['reward_mean']:.4f} | "
            f"{row['reward_std']:.4f} | "
            f"{row['move_text']}"
        )
    print()

debug_groups = dict()
for row in raw_rows:
    label = row["diagnostic_label"]
    if label not in debug_groups:
        debug_groups[label] = []
    debug_groups[label].append(row)

for label in sorted(debug_groups.keys()):
    group = sorted(debug_groups[label], key=lambda row: int(row["trajectory_index"]))
    print(f"Diagnostic: {label}")
    print(f"Move: {group[0]['move_text']}")
    print("Trajectory rewards:")
    print([row["trajectory_value"] for row in group])
    print("Trajectory hash paths:")
    print([row["trajectory_hash_path"] for row in group])
    print()
""";
        cells.Add(CreateCodeCell(notebookCode.Replace("__CSV_PATH__", csvPath, StringComparison.Ordinal)));

        notebook["cells"] = cells;
        notebook["metadata"] = new Dictionary<string, object>
        {
            ["kernelspec"] = new Dictionary<string, object>
            {
                ["display_name"] = "Python 3",
                ["language"] = "python",
                ["name"] = "python3",
            },
            ["language_info"] = new Dictionary<string, object>
            {
                ["name"] = "python",
                ["version"] = "3.11",
            },
        };
        notebook["nbformat"] = 4;
        notebook["nbformat_minor"] = 5;

        JsonSerializerOptions options = new()
        {
            WriteIndented = true,
        };
        File.WriteAllText(filePath, JsonSerializer.Serialize(notebook, options));
    }


    static void WriteTraceCsv(string filePath, IReadOnlyList<TraceCsvRow> rows)
    {
        CSVBuilder output = new();
        for (int rowIndex = 0; rowIndex < rows.Count; ++rowIndex)
        {
            TraceCsvRow row = rows[rowIndex];
            output
                .NextRow()
                .SetCell("row_type", row.RowType)
                .SetCell("turn_index", row.TurnIndex)
                .SetCell("state_text", row.StateText)
                .SetCell("score", row.Score)
                .SetCell("remaining_hands", row.RemainingHands)
                .SetCell("remaining_discards", row.RemainingDiscards)
                .SetCell("remaining_deck", row.RemainingDeck)
                .SetCell("original_policy_rank", row.OriginalPolicyRank)
                .SetCell("policy_probability", row.PolicyProbability.ToString("F8", CultureInfo.InvariantCulture))
                .SetCell("move_index", row.MoveIndex)
                .SetCell("move_text", row.MoveText)
                .SetCell("is_top_move", row.IsTopMove ? 1 : 0)
                .SetCell("is_considered", row.IsConsidered ? 1 : 0)
                .SetCell("is_selected", row.IsSelected ? 1 : 0)
                .SetCell("trajectory_count", row.TrajectoryCount)
                .SetCell("reward_mean", row.RewardMean.ToString("F8", CultureInfo.InvariantCulture))
                .SetCell("reward_std", row.RewardStd.ToString("F8", CultureInfo.InvariantCulture))
                .SetCell("threshold_value", row.ThresholdValue.ToString("F8", CultureInfo.InvariantCulture))
                .SetCell("top_move_text", row.TopMoveText)
                .SetCell("chosen_move_text", row.ChosenMoveText)
                .SetCell("considered_moves_text", row.ConsideredMovesText)
                .SetCell("move_search_seconds", row.MoveSearchSeconds.ToString("F8", CultureInfo.InvariantCulture))
                .SetCell("diagnostic_label", row.DiagnosticLabel)
                .SetCell("trajectory_index", row.TrajectoryIndex)
                .SetCell("trajectory_value", row.TrajectoryValue.ToString("F8", CultureInfo.InvariantCulture))
                .SetCell("trajectory_hash_path", row.TrajectoryHashPath);
        }

        File.WriteAllText(filePath, output.ToString());
    }


    static void AppendTraceRows(List<TraceCsvRow> rows, GameTurnTrace turnTrace)
    {
        for (int candidateIndex = 0; candidateIndex < turnTrace.Candidates.Count; ++candidateIndex)
        {
            CandidateMoveTrace candidate = turnTrace.Candidates[candidateIndex];
            rows.Add(new(
                RowType: "summary",
                TurnIndex: turnTrace.TurnIndex,
                StateText: turnTrace.StateText,
                Score: turnTrace.Score,
                RemainingHands: turnTrace.RemainingHands,
                RemainingDiscards: turnTrace.RemainingDiscards,
                RemainingDeck: turnTrace.RemainingDeck,
                OriginalPolicyRank: candidate.OriginalPolicyRank,
                PolicyProbability: candidate.PolicyProbability,
                MoveIndex: candidate.MoveIndex,
                MoveText: candidate.MoveText,
                IsTopMove: candidate.IsTopMove,
                IsConsidered: candidate.IsConsidered,
                IsSelected: candidate.MoveIndex == turnTrace.ChosenMoveIndex,
                TrajectoryCount: candidate.TrajectoryStats.Count,
                RewardMean: candidate.TrajectoryStats.Mean,
                RewardStd: candidate.TrajectoryStats.SampleStandardDeviation,
                ThresholdValue: turnTrace.ThresholdValue,
                TopMoveText: turnTrace.TopMoveText,
                ChosenMoveText: turnTrace.ChosenMoveText,
                ConsideredMovesText: turnTrace.ConsideredMovesText,
                MoveSearchSeconds: turnTrace.ElapsedSeconds,
                DiagnosticLabel: string.Empty,
                TrajectoryIndex: -1,
                TrajectoryValue: 0f,
                TrajectoryHashPath: string.Empty));
        }

        IEnumerable<CandidateMoveTrace> diagnosticCandidatePool = turnTrace.Candidates
            .Where(candidate => candidate.TrajectoryStats.Count >= 50);
        if (!diagnosticCandidatePool.Any())
            diagnosticCandidatePool = turnTrace.Candidates;

        CandidateMoveTrace zeroStdCandidate = diagnosticCandidatePool
            .OrderBy(candidate => candidate.TrajectoryStats.SampleStandardDeviation)
            .ThenByDescending(candidate => candidate.TrajectoryStats.Count)
            .ThenBy(candidate => candidate.OriginalPolicyRank)
            .First();
        CandidateMoveTrace highStdCandidate = diagnosticCandidatePool
            .OrderByDescending(candidate => candidate.TrajectoryStats.SampleStandardDeviation)
            .ThenByDescending(candidate => candidate.TrajectoryStats.Count)
            .ThenBy(candidate => candidate.OriginalPolicyRank)
            .First();

        AppendDiagnosticTrajectoryRows(rows, turnTrace, zeroStdCandidate, "lowest_std");
        if (!ReferenceEquals(highStdCandidate, zeroStdCandidate))
            AppendDiagnosticTrajectoryRows(rows, turnTrace, highStdCandidate, "highest_std");
    }


    static void AppendDiagnosticTrajectoryRows(List<TraceCsvRow> rows, GameTurnTrace turnTrace, CandidateMoveTrace candidate, string diagnosticLabel)
    {
        for (int trajectoryIndex = 0; trajectoryIndex < candidate.TrajectorySamples.Count; ++trajectoryIndex)
        {
            TrajectorySampleTrace trajectorySample = candidate.TrajectorySamples[trajectoryIndex];
            rows.Add(new(
                RowType: "raw_trajectory",
                TurnIndex: turnTrace.TurnIndex,
                StateText: turnTrace.StateText,
                Score: turnTrace.Score,
                RemainingHands: turnTrace.RemainingHands,
                RemainingDiscards: turnTrace.RemainingDiscards,
                RemainingDeck: turnTrace.RemainingDeck,
                OriginalPolicyRank: candidate.OriginalPolicyRank,
                PolicyProbability: candidate.PolicyProbability,
                MoveIndex: candidate.MoveIndex,
                MoveText: candidate.MoveText,
                IsTopMove: candidate.IsTopMove,
                IsConsidered: candidate.IsConsidered,
                IsSelected: candidate.MoveIndex == turnTrace.ChosenMoveIndex,
                TrajectoryCount: candidate.TrajectoryStats.Count,
                RewardMean: candidate.TrajectoryStats.Mean,
                RewardStd: candidate.TrajectoryStats.SampleStandardDeviation,
                ThresholdValue: turnTrace.ThresholdValue,
                TopMoveText: turnTrace.TopMoveText,
                ChosenMoveText: turnTrace.ChosenMoveText,
                ConsideredMovesText: turnTrace.ConsideredMovesText,
                MoveSearchSeconds: turnTrace.ElapsedSeconds,
                DiagnosticLabel: diagnosticLabel,
                TrajectoryIndex: trajectoryIndex,
                TrajectoryValue: trajectorySample.Value,
                TrajectoryHashPath: trajectorySample.HashPath));
        }
    }


    internal static LatestCheckpointInfo FindLatestPpoCheckpoint(string repoRoot)
    {
        string experimentDir = Path.Combine(repoRoot, "Analysis", PpoConfig.ExperimentName);
        return FindLatestCheckpointInDirectory(experimentDir);
    }


    static LatestCheckpointInfo FindLatestCheckpoint(string repoRoot)
    {
        string analysisRoot = Path.Combine(repoRoot, "Analysis");
        return FindLatestCheckpointInDirectory(analysisRoot);
    }


    internal static LatestCheckpointInfo FindLatestCheckpointInDirectory(string searchRoot)
    {
        string[] checkpointPaths = Directory.GetFiles(searchRoot, "*.bin", SearchOption.AllDirectories);
        if (checkpointPaths.Length == 0)
            throw new FileNotFoundException($"No checkpoints found beneath {searchRoot}");

        string bestPath = checkpointPaths[0];
        DateTime bestTime = File.GetLastWriteTimeUtc(bestPath);
        for (int checkpointIndex = 1; checkpointIndex < checkpointPaths.Length; ++checkpointIndex)
        {
            string candidatePath = checkpointPaths[checkpointIndex];
            DateTime candidateTime = File.GetLastWriteTimeUtc(candidatePath);
            if (candidateTime <= bestTime)
                continue;

            bestPath = candidatePath;
            bestTime = candidateTime;
        }

        DirectoryInfo experimentDirectory = Directory.GetParent(Directory.GetParent(bestPath)!.FullName)!;
        return new(
            CheckpointPath: bestPath,
            ExperimentName: experimentDirectory.Name);
    }


    public static string DescribeState(GameState gameState)
    {
        return $"{gameState} | Score {gameState.ScoringState.CurrentRoundTotalScore:F1} | Deck {gameState.DeckState.RemainingDeckCardCount}";
    }


    static PreparedExperimentPaths PrepareExperiment(string repoRoot, string commitHash, ExperimentConfig config, ResumeConfig resume)
    {
        string analysisDir = Path.Combine(repoRoot, "Analysis", config.ExperimentName);
        string localWeightsDir = Path.Combine(analysisDir, "weights");
        string readmePath = Path.Combine(analysisDir, "README.md");
        string notebookPath = Path.Combine(analysisDir, "analysis.ipynb");
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string copiedProgramPath = Path.Combine(analysisDir, "Program.cs");

        Directory.CreateDirectory(analysisDir);
        Directory.CreateDirectory(localWeightsDir);

        WriteReadme(
            filePath: readmePath,
            commitHash: commitHash,
            config: config,
            localWeightsDir: localWeightsDir,
            resume: resume);
        WriteNotebook(
            filePath: notebookPath,
            csvPath: analysisCsvPath,
            experimentName: config.ExperimentName,
            referenceCsvPath: string.IsNullOrWhiteSpace(config.NotebookReferenceExperimentName) ? "" : Path.Combine(repoRoot, "Analysis", config.NotebookReferenceExperimentName, "analysis.csv"),
            referenceLabel: string.IsNullOrWhiteSpace(config.NotebookReferenceExperimentName) ? "" : config.NotebookReferenceExperimentName);
        File.Copy(
            sourceFileName: Path.Combine(repoRoot, "ConsoleApp", "Program.cs"),
            destFileName: copiedProgramPath,
            overwrite: true);

        return new(
            AnalysisDir: analysisDir,
            LocalWeightsDir: localWeightsDir,
            AnalysisCsvPath: analysisCsvPath);
    }


    static void WriteReadme(string filePath, string commitHash, ExperimentConfig config, string localWeightsDir, ResumeConfig resume)
    {
        string runDate = DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        string extraTrainingLines = "";
        if (!string.IsNullOrWhiteSpace(resume.CheckpointPath))
            extraTrainingLines += $"""

- Resumed from checkpoint: `{resume.CheckpointPath}`
- Resume step: `{resume.ResumeStep}`
""";

        if (!string.IsNullOrWhiteSpace(config.NotebookReferenceExperimentName))
            extraTrainingLines += $"""

- Reference experiment: `{config.NotebookReferenceExperimentName}`
""";

        string descriptionText = string.IsNullOrWhiteSpace(resume.CheckpointPath)
            ? "- This experiment starts PPO from scratch with a fresh model initialization."
            : $"- This experiment continues PPO from the step `{resume.ResumeStep}` checkpoint using the current multi-round rollout pipeline.";
        string rewardText = config.TrainOnlyFirstRound
            ? "first-round terminal reward: lose = 0, win = 1 + 0.1 * remaining hands"
            : "reward = (rounds survived / 3)^2";
        string rolloutText = config.TrainOnlyFirstRound
            ? "Each rollout follows first-round in-round decision states until the round resolves, then applies the terminal round reward."
            : "Each rollout follows games across both in-round and in-shop decision states until terminal reward, with round and store minibatches mixed into the same optimizer step.";
        string policyTrainingText = config.TrainOnlyFirstRound
            ? "Round policy training uses the PPO clipped-surrogate objective with entropy regularization."
            : "Round and store policy training both use the PPO clipped-surrogate objective with entropy regularization.";
        string readme = $"""
Date: {runDate}
Commit Hash: {commitHash}

# Training Params
1. Policy/value model: `PpoPolicyValueModel`
2. Optimizer: `AdamW`
3. Learning rate: `{config.LearningRate.ToString("0.0e0", CultureInfo.InvariantCulture)}`
4. Adam beta1: `{config.AdamBeta1}`
5. Adam beta2: `{config.AdamBeta2}`
6. Weight decay: `{config.WeightDecay}`
7. PPO clip threshold: `{config.PpoEpsilon}`
8. Entropy coefficient: `{config.EntropyCoefficient}`
9. Rollout size: `{config.RolloutSize}` positions
10. Parallel rollout games: `{config.RolloutParallelGameCount}`
11. Batch size: `{config.BatchSize}`
12. Training epochs per step: `{config.TrainingEpochsPerStep}`
13. Total training steps: `{config.StepCount}`
14. Sampled softmax candidates: `{config.SampledSoftmaxCount}`
15. Policy/value update pattern: `1 policy batch, 1 value batch, then optimizer step`
16. Reward function: `{rewardText}`
17. Value target: `average of later position value predictions, with terminal position assigned true reward`
18. Normalization: `LayerNorm`
19. Value head: `Linear(768 -> 1)` attached to the trunk residual stream
20. Value loss coefficient: `{config.ValueLossCoefficient}`
21. Snapshot frequency: every `{config.SnapshotFrequency}` step(s)
22. Snapshot weights directory: `{localWeightsDir}`
23. Notebook: [analysis.ipynb](analysis.ipynb)
24. Starting hands per round: `{config.InitialHandsPerRound}`
25. Starting discards per round: `{config.InitialDiscardsPerRound}`
26. Deck initializer: `{(config.UseRandomDeckInitializer ? "uniform random 52 cards with replacement" : "default deck")}`
{extraTrainingLines}

# Description
- {descriptionText[2..]}
- {rolloutText}
- Reward is only assigned when the first round reaches its terminal end-round state.
- {policyTrainingText}
- The value network trains on the same on-policy minibatches as the policy update, sharing the same trunk forward pass for each sample.
- The CSV contains only an `average_reward` timeseries, one row per PPO step.
- The notebook graphs the reward timeseries over the run.
""";
        File.WriteAllText(filePath, readme);
    }


    static ResumeConfig GetResumeConfig(string repoRoot, ExperimentConfig config)
    {
        if (string.IsNullOrWhiteSpace(config.ResumeSourceExperimentName))
            return new(
                CheckpointPath: "",
                ResumeStep: 0);

        string weightsDir = Path.Combine(repoRoot, "Analysis", config.ResumeSourceExperimentName, "weights");
        if (!Directory.Exists(weightsDir))
            throw new DirectoryNotFoundException($"Resume weights directory not found: {weightsDir}");

        string[] checkpointPaths = Directory.GetFiles(weightsDir, "*.bin");
        if (checkpointPaths.Length == 0)
            throw new FileNotFoundException($"No checkpoints found in {weightsDir}");

        int bestStep = -1;
        string bestPath = "";
        for (int checkpointIndex = 0; checkpointIndex < checkpointPaths.Length; ++checkpointIndex)
        {
            string checkpointPath = checkpointPaths[checkpointIndex];
            string fileName = Path.GetFileNameWithoutExtension(checkpointPath);
            if (!int.TryParse(fileName, out int step))
                continue;
            if (step <= bestStep)
                continue;

            bestStep = step;
            bestPath = checkpointPath;
        }

        if (bestStep < 0)
            throw new InvalidOperationException($"No numeric checkpoint names found in {weightsDir}");

        return new(
            CheckpointPath: bestPath,
            ResumeStep: bestStep);
    }


    static void WriteNotebook(string filePath, string csvPath, string experimentName, string referenceCsvPath, string referenceLabel)
    {
        _ = referenceCsvPath;
        _ = referenceLabel;
        Dictionary<string, object> notebook = [];
        List<Dictionary<string, object>> cells = [];

        cells.Add(CreateMarkdownCell($"""
# {experimentName}

This notebook visualizes the reward timeseries stored in `analysis.csv`.
"""));

        cells.Add(CreateCodeCell("""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

csv_path = Path(r"__CSV_PATH__")

rewards = []
if csv_path.exists():
    with csv_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            value = row.get("average_reward")
            if value is not None and value != "":
                rewards.append(float(value))

steps = list(range(1, len(rewards) + 1))
print(f"Loaded {len(rewards)} reward points")
""".Replace("__CSV_PATH__", csvPath, StringComparison.Ordinal)
            .Replace("__ROLLOUT_SIZE__", PpoConfig.RolloutSize.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal)));

        cells.Add(CreateCodeCell("""
plt.style.use("seaborn-v0_8-whitegrid")

figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
figure.patch.set_facecolor("white")
axis.plot(steps, rewards, linewidth=2.0, color="#1f8f5f")
axis.set_title("Average Reward")
axis.set_xlabel("PPO Step")
axis.set_ylabel("Average Reward")
axis.grid(True, alpha=0.22, linewidth=0.8)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
plt.show()
"""));

        notebook["cells"] = cells;
        notebook["metadata"] = new Dictionary<string, object>
        {
            ["kernelspec"] = new Dictionary<string, object>
            {
                ["display_name"] = "Python 3",
                ["language"] = "python",
                ["name"] = "python3",
            },
            ["language_info"] = new Dictionary<string, object>
            {
                ["name"] = "python",
                ["version"] = "3.11",
            },
        };
        notebook["nbformat"] = 4;
        notebook["nbformat_minor"] = 5;

        JsonSerializerOptions options = new()
        {
            WriteIndented = true,
        };
        File.WriteAllText(filePath, JsonSerializer.Serialize(notebook, options));
    }


    static Dictionary<string, object> CreateMarkdownCell(string source)
    {
        return new()
        {
            ["cell_type"] = "markdown",
            ["metadata"] = new Dictionary<string, object>(),
            ["source"] = SplitNotebookLines(source),
        };
    }


    static Dictionary<string, object> CreateCodeCell(string source)
    {
        return new()
        {
            ["cell_type"] = "code",
            ["metadata"] = new Dictionary<string, object>(),
            ["execution_count"] = null,
            ["outputs"] = new List<object>(),
            ["source"] = SplitNotebookLines(source),
        };
    }


    static List<string> SplitNotebookLines(string source)
    {
        string[] rawLines = source.Replace("\r\n", "\n").Split('\n');
        List<string> lines = new(rawLines.Length);
        for (int lineIndex = 0; lineIndex < rawLines.Length; ++lineIndex)
        {
            string suffix = lineIndex == rawLines.Length - 1 ? string.Empty : "\n";
            lines.Add(rawLines[lineIndex] + suffix);
        }

        return lines;
    }


    static bool GetEnvironmentFlag(string key)
    {
        string value = Environment.GetEnvironmentVariable(key);
        if (string.IsNullOrWhiteSpace(value))
            return false;

        return value.Equals("1", StringComparison.OrdinalIgnoreCase) ||
               value.Equals("true", StringComparison.OrdinalIgnoreCase) ||
               value.Equals("yes", StringComparison.OrdinalIgnoreCase);
    }


    static float GetElapsedSeconds(Stopwatch stopwatch)
    {
        return (float)stopwatch.Elapsed.TotalSeconds;
    }


    static string FindRepoRoot()
    {
        string currentDirectory = Directory.GetCurrentDirectory();
        DirectoryInfo directory = new(currentDirectory);

        while (directory is not null)
        {
            string solutionPath = Path.Combine(directory.FullName, "Ramen.sln");
            if (File.Exists(solutionPath))
                return directory.FullName;

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException("Could not find repo root containing Ramen.sln.");
    }


    static string GetGitCommitHash(string repoRoot)
    {
        ProcessStartInfo startInfo = new()
        {
            FileName = "git",
            Arguments = "rev-parse --short HEAD",
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };

        using Process process = Process.Start(startInfo);
        process.WaitForExit();

        string output = process.StandardOutput.ReadToEnd().Trim();
        if (!string.IsNullOrWhiteSpace(output))
            return output;

        return "unknown";
    }
}

public readonly record struct TraceExperimentConfig
(
    string ExperimentName,
    int PolicyTopMoveTrajectoryCount,
    int AdditionalTopMoveCount,
    int InitialOtherMoveTrajectoryCount,
    int AdditionalTrajectoryCountPerRound,
    int MaxTrajectoryCount,
    float MaxEntropy,
    int RandomSeed,
    int MaxTurnsToTrace
);

public readonly record struct PreparedTraceExperimentPaths
(
    string AnalysisDir,
    string AnalysisCsvPath
);

public readonly record struct LatestCheckpointInfo
(
    string CheckpointPath,
    string ExperimentName
);

public readonly record struct TraceCsvRow
(
    string RowType,
    int TurnIndex,
    string StateText,
    float Score,
    int RemainingHands,
    int RemainingDiscards,
    int RemainingDeck,
    int OriginalPolicyRank,
    float PolicyProbability,
    int MoveIndex,
    string MoveText,
    bool IsTopMove,
    bool IsConsidered,
    bool IsSelected,
    int TrajectoryCount,
    float RewardMean,
    float RewardStd,
    float ThresholdValue,
    string TopMoveText,
    string ChosenMoveText,
    string ConsideredMovesText,
    float MoveSearchSeconds,
    string DiagnosticLabel,
    int TrajectoryIndex,
    float TrajectoryValue,
    string TrajectoryHashPath
);

public readonly record struct ResumeConfig
(
    string CheckpointPath,
    int ResumeStep
);

public readonly record struct PreparedTraceEvaluationPaths
(
    string AnalysisDir,
    string TracePath
);

public readonly record struct PreparedOutcomeEvaluationPaths
(
    string AnalysisDir,
    string AnalysisCsvPath
);

public readonly record struct OutcomeDistribution
(
    int GameCount,
    int CompletedCount = 0,
    int LossCount = 0,
    int WinHands0Count = 0,
    int WinHands1Count = 0,
    int WinHands2Count = 0,
    int WinHands3Count = 0
);

public readonly record struct PreparedExperimentPaths
(
    string AnalysisDir,
    string LocalWeightsDir,
    string AnalysisCsvPath
);
