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
using static TorchSharp.torch.nn;

public static class Program
{
    const string DeckComparisonEvaluationExperimentName = "2026-04-25_simple_flush_ppo_step355_temp0_standard_vs_checkered_eval3000";
    const int DeckComparisonEvaluationGameCount = 3000;
    const string TraceEvaluationExperimentName = "2026-04-23_simple_flush_ppo_step355_temp0_randdeck_trace5_games";
    const int TraceEvaluationGameCount = 5;
    const string OutcomeEvaluationExperimentName = "2026-04-23_simple_flush_ppo_step355_temp0_randdeck_eval10k_outcomes";
    const string OutcomeEvaluationCheckpointPath = "/Users/miles/Desktop/dev/repos/BalatroAI/Analysis/2026-04-23_simple_flush_ppo_stdreward_resume315_randdeck52wr_r32768_b1024_e4_lr1e4_20more_eps0p3_ent3e5_ss40_trunk512_addgelu_vhead/weights/355.bin";
    const int OutcomeEvaluationGameCount = 10000;
    const int OutcomeEvaluationBatchSize = 500;

    static readonly ExperimentConfig PpoConfig = new(
        ExperimentName: "2026-04-23_simple_flush_ppo_stdreward_resume315_randdeck52wr_r32768_b1024_e4_lr1e4_20more_eps0p3_ent3e5_ss40_trunk512_addgelu_vhead",
        RolloutSize: 32768,
        RolloutParallelGameCount: 64,
        BatchSize: 1024,
        TrainingEpochsPerStep: 4,
        StepCount: 355,
        SampledSoftmaxCount: 40,
        LearningRate: 3e-5f,
        AdamBeta1: 0.9f,
        AdamBeta2: 0.97f,
        WeightDecay: 0f,
        PpoEpsilon: 0.3f,
        EntropyCoefficient: 1e-5f,
        ValueReplayBufferCapacity: 0,
        SnapshotFrequency: 5,
        RandomSeed: 1337,
        InitialHandsPerRound: 4,
        InitialDiscardsPerRound: 3,
        UseRandomDeckInitializer: true,
        ResumeSourceExperimentName: "2026-04-23_simple_flush_ppo_stdreward_resume315_randdeck52wr_r32768_b1024_e4_lr1e4_20more_eps0p3_ent3e5_ss40_trunk512_addgelu_vhead",
        NotebookReferenceExperimentName: "2026-04-23_simple_flush_ppo_stdreward_resume285_randdeck52wr_r32768_b1024_e4_lr1e4_30more_eps0p3_ent3e5_ss40_trunk512_addgelu_vhead");

    const float PpoContinuationLearningRate = 3e-5f;

    public static void Main()
    {
        set_default_device(mps_is_available() ? MPS : CPU);
        TensorManager.Init();

        RunDeckComparisonEvaluation();
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

        using PpoPolicyValueModel model = new(useTorchScriptCompile: true);
        model.Load(OutcomeEvaluationCheckpointPath);

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

        Directory.CreateDirectory(analysisDir);

        string readme = $"""
Date: {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}
Commit Hash: {commitHash}

# Training Params
1. Loaded checkpoint: `{OutcomeEvaluationCheckpointPath}`
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

        using PpoPolicyValueModel model = new(useTorchScriptCompile: true);
        model.Load(OutcomeEvaluationCheckpointPath);

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

        Directory.CreateDirectory(analysisDir);

        string readme = $"""
Date: {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}
Commit Hash: {commitHash}

# Training Params
1. Loaded checkpoint: `{OutcomeEvaluationCheckpointPath}`
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

        using PolicyOnlyAgent agent = new(model);
        GameData gameData = CreateOutcomeEvaluationGameData();
        StringBuilder trace = new();

        trace.AppendLine("# Step 355 Greedy Trace");
        trace.AppendLine();
        trace.AppendLine($"Checkpoint: `{OutcomeEvaluationCheckpointPath}`");
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
                UseHandMove move = PolicyOnlyAgent.MoveForIndex(gameState, chosenMoveIndex);
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

        using PpoPolicyValueModel model = new(useTorchScriptCompile: true);
        model.Load(OutcomeEvaluationCheckpointPath);

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

        Console.WriteLine($"checkpoint {OutcomeEvaluationCheckpointPath}");
        Console.WriteLine($"games {OutcomeEvaluationGameCount}");
        Console.WriteLine($"batch_size {OutcomeEvaluationBatchSize}");
        Console.WriteLine($"wall_seconds {stopwatch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"loss {distribution.LossCount} ({GetFractionText(distribution.LossCount, distribution.GameCount)})");
        Console.WriteLine($"win_hands_0 {distribution.WinHands0Count} ({GetFractionText(distribution.WinHands0Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_1 {distribution.WinHands1Count} ({GetFractionText(distribution.WinHands1Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_2 {distribution.WinHands2Count} ({GetFractionText(distribution.WinHands2Count, distribution.GameCount)})");
        Console.WriteLine($"win_hands_3 {distribution.WinHands3Count} ({GetFractionText(distribution.WinHands3Count, distribution.GameCount)})");
    }


    static PreparedOutcomeEvaluationPaths PrepareOutcomeEvaluation(string repoRoot, string commitHash)
    {
        string analysisDir = Path.Combine(repoRoot, "Analysis", OutcomeEvaluationExperimentName);
        string readmePath = Path.Combine(analysisDir, "README.md");
        string analysisCsvPath = Path.Combine(analysisDir, "analysis.csv");
        string copiedProgramPath = Path.Combine(analysisDir, "Program.cs");

        Directory.CreateDirectory(analysisDir);

        string readme = $"""
Date: {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}
Commit Hash: {commitHash}

# Training Params
1. Loaded checkpoint: `{OutcomeEvaluationCheckpointPath}`
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


    static OutcomeDistribution EvaluateOutcomeDistribution(PpoPolicyValueModel model, int gameCount, int batchSize, GameData gameData)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        using PolicyOnlyAgent agent = new(model);
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
                PolicyOnlyAgent.MoveForIndex(activeStates[index], chosenMoveIndex).Apply(activeStates[index]);
            }
        }

        return distribution;
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

        foreach ((string jokerName, Joker joker) in GameData.Default.Jokers)
            gameData.Jokers.Add(jokerName, joker);

        return gameData;
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

        foreach ((string jokerName, Joker joker) in GameData.Default.Jokers)
            gameData.Jokers.Add(jokerName, joker);

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

        foreach ((string jokerName, Joker joker) in GameData.Default.Jokers)
            gameData.Jokers.Add(jokerName, joker);

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

        List<StepMetrics> metrics = LoadExistingMetrics(
            filePath: preparedPaths.AnalysisCsvPath,
            maxStepInclusive: resume.ResumeStep);
        float priorWallClockSeconds = metrics.Count == 0 ? 0f : metrics[^1].WallClockSeconds;
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
        using PpoMiniBatchBuffers batchBuffers = new(
            batchSize: PpoConfig.BatchSize,
            sampledSoftmaxCount: PpoConfig.SampledSoftmaxCount);
        Random rolloutRandom = new(PpoConfig.RandomSeed);
        Random shuffleRandom = new(PpoConfig.RandomSeed + 1);

        for (int step = resume.ResumeStep + 1; step <= PpoConfig.StepCount; ++step)
        {
            int continuationStep = step - resume.ResumeStep;
            float learningRate = GetPpoLearningRate(continuationStep);
            SetOptimizerLearningRate(optimizer, learningRate);

            using PpoRolloutDataset rollout = GenerateRollout(
                model: model,
                config: PpoConfig,
                random: rolloutRandom);
            TrainingMetrics trainingMetrics = TrainStep(
                model: model,
                optimizer: optimizer,
                rollout: rollout,
                batchBuffers: batchBuffers,
                shuffleRandom: shuffleRandom,
                config: PpoConfig);
            StepMetrics stepMetrics = new(
                Step: step,
                WallClockSeconds: priorWallClockSeconds + GetElapsedSeconds(stopwatch),
                AverageReward: rollout.AverageReward,
                AverageMoveEntropy: rollout.AverageMoveEntropy,
                ValueMseMean: trainingMetrics.ValueMseMean,
                PolicyLossMean: trainingMetrics.PolicyLossMean,
                CompletedGameCount: rollout.CompletedGameCount,
                LearningRate: learningRate,
                ValueReplayCount: 0);
            metrics.Add(stepMetrics);
            WriteMetricsCsv(preparedPaths.AnalysisCsvPath, metrics);

            Console.WriteLine(
                $"step {step}/{PpoConfig.StepCount} | " +
                $"wall {stepMetrics.WallClockSeconds:F2}s | " +
                $"reward {stepMetrics.AverageReward:F4} | " +
                $"entropy {stepMetrics.AverageMoveEntropy:F4} | " +
                $"value_mse {stepMetrics.ValueMseMean:F4} | " +
                $"policy_loss {stepMetrics.PolicyLossMean:F4} | " +
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


    static LatestCheckpointInfo FindLatestCheckpoint(string repoRoot)
    {
        string analysisRoot = Path.Combine(repoRoot, "Analysis");
        string[] checkpointPaths = Directory.GetFiles(analysisRoot, "*.bin", SearchOption.AllDirectories);
        if (checkpointPaths.Length == 0)
            throw new FileNotFoundException($"No checkpoints found beneath {analysisRoot}");

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


    static PpoRolloutDataset GenerateRollout(PpoPolicyValueModel model, ExperimentConfig config, Random random)
    {
        PpoRolloutDataset rollout = new(config.RolloutSize, config.SampledSoftmaxCount);
        GameData rolloutGameData = CreateConfiguredGameData(config);
        GameState[] gameStates = new GameState[config.RolloutParallelGameCount];
        List<TrajectoryPosition>[] activeTrajectories = new List<TrajectoryPosition>[config.RolloutParallelGameCount];
        float rewardSum = 0f;
        float entropySum = 0f;
        int rewardGameCount = 0;

        for (int slot = 0; slot < gameStates.Length; ++slot)
        {
            gameStates[slot] = new(rolloutGameData);
            activeTrajectories[slot] = [];
        }

        while (rollout.Count < config.RolloutSize)
        {
            List<int> activeIndices = [];
            List<TrajectoryPosition> activePositions = [];

            for (int slot = 0; slot < gameStates.Length; ++slot)
            {
                GameState gameState = gameStates[slot];
                gameState.AdvanceToNextPlayerChoice();

                if (gameState.GameIsDone)
                {
                    FinalizeCompletedTrajectory(
                        rollout: rollout,
                        trajectory: activeTrajectories[slot],
                        terminalState: gameState,
                        rewardSum: ref rewardSum,
                        rewardGameCount: ref rewardGameCount,
                        entropySum: ref entropySum);
                    if (rollout.Count >= config.RolloutSize)
                        break;

                    gameStates[slot] = new(rolloutGameData);
                    activeTrajectories[slot] = [];
                    gameState = gameStates[slot];
                    gameState.AdvanceToNextPlayerChoice();
                }

                if (rollout.Count >= config.RolloutSize)
                    break;

                activeIndices.Add(slot);
                activePositions.Add(new(gameState));
            }

            if (rollout.Count >= config.RolloutSize)
                break;

            if (activeIndices.Count == 0)
                continue;

            using var scope = NewDisposeScope();

            (GameStateTensors stateTensors, UseHandTensors useHandTensors) = BuildRolloutBatch(activePositions);
            (Tensor logits, Tensor values) = model.GetPolicyLogitsAndValues(stateTensors, useHandTensors);
            Tensor illegalMask = BuildIllegalMoveMask(
                remainingHands: stateTensors.RemainingHands.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64),
                remainingDiscards: stateTensors.RemainingDiscards.to(PpoPolicyValueModel.EvalDevice).to_type(ScalarType.Int64));
            Tensor probs = functional.softmax(logits + illegalMask, dim: 1).to(CPU);
            float[] flatProbs = [.. probs.data<float>()];
            float[] flatValues = [.. values.to(CPU).data<float>()];

            for (int activeIndex = 0; activeIndex < activeIndices.Count; ++activeIndex)
            {
                int rowOffset = activeIndex * PpoPolicyValueModel.MoveCount;
                Span<float> rowSpan = flatProbs.AsSpan(rowOffset, PpoPolicyValueModel.MoveCount);
                activePositions[activeIndex].PositionValueEstimate = flatValues[activeIndex];
                int chosenMoveIndex = SampleMoveIndex(rowSpan, random);
                FillSampledSoftmaxTargets(
                    position: activePositions[activeIndex],
                    chosenMoveIndex: chosenMoveIndex,
                    sampledSoftmaxCount: config.SampledSoftmaxCount,
                    fullProbs: rowSpan,
                    random: random);

                activeTrajectories[activeIndices[activeIndex]].Add(activePositions[activeIndex]);
                UseHandMove move = PolicyOnlyAgent.MoveForIndex(
                    state: gameStates[activeIndices[activeIndex]],
                    index: chosenMoveIndex);
                move.Apply(gameStates[activeIndices[activeIndex]]);
            }
        }

        rollout.SetMetrics(
            averageReward: rewardGameCount == 0 ? 0f : rewardSum / rewardGameCount,
            averageMoveEntropy: rollout.Count == 0 ? 0f : entropySum / rollout.Count,
            completedGameCount: rewardGameCount);
        return rollout;
    }


    static GameData CreateConfiguredGameData(ExperimentConfig config)
    {
        GameData gameData = new()
        {
            Seed = GameData.Default.Seed,
            Hands = config.InitialHandsPerRound,
            Discards = config.InitialDiscardsPerRound,
            RandomizeSeed = GameData.Default.RandomizeSeed,
            StartingHandBaseScore = [.. GameData.Default.StartingHandBaseScore],
            PlanetScores = [.. GameData.Default.PlanetScores],
            InitStartingDeck = config.UseRandomDeckInitializer ? GameData.InitErraticStartingDeck : GameData.InitErraticStartingDeck,
        };

        foreach ((string jokerName, Joker joker) in GameData.Default.Jokers)
            gameData.Jokers.Add(jokerName, joker);

        return gameData;
    }


    static void FinalizeCompletedTrajectory(
        PpoRolloutDataset rollout,
        List<TrajectoryPosition> trajectory,
        GameState terminalState,
        ref float rewardSum,
        ref int rewardGameCount,
        ref float entropySum)
    {
        if (trajectory.Count == 0)
            return;

        float finalReward = GetStandardReward(terminalState);
        float[] suffixValues = new float[trajectory.Count + 1];
        for (int index = trajectory.Count - 1; index >= 0; --index)
            suffixValues[index] = suffixValues[index + 1] + trajectory[index].PositionValueEstimate;

        int remainingCapacity = rollout.Capacity - rollout.Count;
        int addCount = Math.Min(remainingCapacity, trajectory.Count);
        if (addCount <= 0)
            return;

        rewardSum += finalReward;
        rewardGameCount++;

        for (int index = 0; index < addCount; ++index)
        {
            float futureValueSum = suffixValues[index + 1] + finalReward;
            int futureCount = trajectory.Count - index;
            float valueTarget = futureValueSum / futureCount;
            rollout.AddSample(trajectory[index], valueTarget);
            entropySum += trajectory[index].PolicyEntropy;
        }
    }


    static TrainingMetrics TrainStep(
        PpoPolicyValueModel model,
        AdamW optimizer,
        PpoRolloutDataset rollout,
        PpoMiniBatchBuffers batchBuffers,
        Random shuffleRandom,
        ExperimentConfig config)
    {
        float totalPolicyLoss = 0f;
        float totalValueMse = 0f;
        int batchCount = 0;

        for (int epoch = 0; epoch < config.TrainingEpochsPerStep; ++epoch)
        {
            int[] shuffledIndices = BuildShuffledIndices(rollout.Count, shuffleRandom);

            for (int batchStart = 0; batchStart < rollout.Count; batchStart += config.BatchSize)
            {
                using var scope = NewDisposeScope();

                rollout.FillBatch(
                    shuffledIndices: shuffledIndices,
                    batchStart: batchStart,
                    batchSize: config.BatchSize,
                    batchBuffers: batchBuffers);

                GameStateTensors stateTensors = new()
                {
                    FullHand = batchBuffers.Batch.StateTensors.FullHand.to(PpoPolicyValueModel.EvalDevice),
                    RemainingDeck = batchBuffers.Batch.StateTensors.RemainingDeck.to(PpoPolicyValueModel.EvalDevice),
                    RemainingHands = batchBuffers.Batch.StateTensors.RemainingHands.to(PpoPolicyValueModel.EvalDevice),
                    RemainingDiscards = batchBuffers.Batch.StateTensors.RemainingDiscards.to(PpoPolicyValueModel.EvalDevice),
                    Score = batchBuffers.Batch.StateTensors.Score.to(PpoPolicyValueModel.EvalDevice),
                };
                UseHandTensors useHandTensors = new()
                {
                    Score = batchBuffers.Batch.UseHandTensors.Score.to(PpoPolicyValueModel.EvalDevice),
                };
                Tensor sampledMoveIndices = batchBuffers.Batch.SampledMoveIndices.to(PpoPolicyValueModel.EvalDevice);
                Tensor sampledMoveLogQ = batchBuffers.Batch.SampledMoveLogQ.to(PpoPolicyValueModel.EvalDevice);
                Tensor sampledMoveValidMask = batchBuffers.Batch.SampledMoveValidMask.to(PpoPolicyValueModel.EvalDevice);
                Tensor oldSampledProbs = batchBuffers.Batch.OldSampledMoveProbs.to(PpoPolicyValueModel.EvalDevice);
                Tensor valueTargets = batchBuffers.Batch.ValueTargets.to(PpoPolicyValueModel.EvalDevice);

                optimizer.zero_grad();

                (Tensor sampledLogits, Tensor values) = model.GetSelectedPolicyLogitsAndValues(
                    gameStateTensors: stateTensors,
                    useHandTensors: useHandTensors,
                    moveIndices: sampledMoveIndices);
                Tensor correctedLogits = sampledLogits - sampledMoveLogQ;
                Tensor invalidMask = (1f - sampledMoveValidMask) * -1e9f;
                Tensor maskedCorrectedLogits = correctedLogits + invalidMask;
                Tensor logProbs = functional.log_softmax(maskedCorrectedLogits, dim: 1);
                Tensor probs = exp(logProbs) * sampledMoveValidMask;
                Tensor entropy = -(probs * logProbs).sum(dim: 1).mean();
                Tensor logPiNew = logProbs.select(1, 0);
                Tensor logPiOld = oldSampledProbs.select(1, 0).clamp_min(1e-9f).log();
                Tensor ratio = exp(logPiNew - logPiOld);
                Tensor advantages = (valueTargets - values).detach();
                Tensor clippedRatio = clamp(
                    ratio,
                    min: 1f - config.PpoEpsilon,
                    max: 1f + config.PpoEpsilon);
                Tensor surrogate = min(ratio * advantages, clippedRatio * advantages);
                Tensor policyLoss = -surrogate.mean() - config.EntropyCoefficient * entropy;
                Tensor valueLoss = functional.mse_loss(values, valueTargets);
                Tensor totalLoss = policyLoss + valueLoss;
                totalLoss.backward();

                optimizer.step();

                totalPolicyLoss += policyLoss.item<float>();
                totalValueMse += valueLoss.item<float>();
                batchCount++;
            }
        }

        float divisor = Math.Max(batchCount, 1);
        return new(
            PolicyLossMean: totalPolicyLoss / divisor,
            ValueMseMean: totalValueMse / divisor);
    }


    static (GameStateTensors stateTensors, UseHandTensors useHandTensors) BuildRolloutBatch(IReadOnlyList<TrajectoryPosition> positions)
    {
        long[,] fullHand = new long[positions.Count, GameData.HandSize];
        long[,] remainingDeck = new long[positions.Count, 52];
        long[] remainingHands = new long[positions.Count];
        long[] remainingDiscards = new long[positions.Count];
        float[,] score = new float[positions.Count, 1];
        float[,] useHandScores = new float[positions.Count, PpoPolicyValueModel.UseableHandCount];

        for (int batchIndex = 0; batchIndex < positions.Count; ++batchIndex)
        {
            TrajectoryPosition position = positions[batchIndex];
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                fullHand[batchIndex, cardIndex] = position.FullHand[cardIndex];

            for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                remainingDeck[batchIndex, cardIndex] = position.RemainingDeck[cardIndex];

            remainingHands[batchIndex] = position.RemainingHands;
            remainingDiscards[batchIndex] = position.RemainingDiscards;
            score[batchIndex, 0] = position.Score;

            for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
                useHandScores[batchIndex, handIndex] = position.UseHandScores[handIndex];
        }

        return (
            stateTensors: new()
            {
                FullHand = tensor(fullHand, dtype: ScalarType.Int64),
                RemainingDeck = tensor(remainingDeck, dtype: ScalarType.Int64),
                RemainingHands = tensor(remainingHands, dtype: ScalarType.Int64),
                RemainingDiscards = tensor(remainingDiscards, dtype: ScalarType.Int64),
                Score = tensor(score, dtype: ScalarType.Float32),
            },
            useHandTensors: new()
            {
                Score = tensor(useHandScores, dtype: ScalarType.Float32),
            });
    }


    static void FillSampledSoftmaxTargets(
        TrajectoryPosition position,
        int chosenMoveIndex,
        int sampledSoftmaxCount,
        ReadOnlySpan<float> fullProbs,
        Random random)
    {
        Array.Clear(position.SampledMoveIndices, 0, position.SampledMoveIndices.Length);
        Array.Clear(position.OldSampledMoveProbs, 0, position.OldSampledMoveProbs.Length);
        Array.Clear(position.SampledMoveLogQ, 0, position.SampledMoveLogQ.Length);
        Array.Clear(position.SampledMoveValidMask, 0, position.SampledMoveValidMask.Length);

        position.SampledMoveIndices[0] = chosenMoveIndex;
        position.SampledMoveValidMask[0] = 1f;
        position.PolicyEntropy = GetEntropy(fullProbs);

        int[] candidatePool = new int[PpoPolicyValueModel.MoveCount];
        int poolCount = BuildNegativeMovePool(
            remainingHands: position.RemainingHands,
            remainingDiscards: position.RemainingDiscards,
            targetMoveIndex: chosenMoveIndex,
            output: candidatePool);
        int negativeSampleCount = Math.Min(sampledSoftmaxCount - 1, poolCount);
        float negativeLogQ = negativeSampleCount == 0 ? 0f : MathF.Log(negativeSampleCount / (float)poolCount);

        for (int sampleIndex = 1; sampleIndex <= negativeSampleCount; ++sampleIndex)
        {
            int selectedIndex = random.Next(poolCount);
            position.SampledMoveIndices[sampleIndex] = candidatePool[selectedIndex];
            position.SampledMoveLogQ[sampleIndex] = negativeLogQ;
            position.SampledMoveValidMask[sampleIndex] = 1f;
            poolCount--;
            candidatePool[selectedIndex] = candidatePool[poolCount];
        }

        int validCount = negativeSampleCount + 1;
        float maxCorrectedLogit = float.NegativeInfinity;
        float[] correctedLogits = new float[validCount];

        for (int sampleIndex = 0; sampleIndex < validCount; ++sampleIndex)
        {
            int moveIndex = (int)position.SampledMoveIndices[sampleIndex];
            float correctedLogit = MathF.Log(MathF.Max(fullProbs[moveIndex], 1e-9f)) - position.SampledMoveLogQ[sampleIndex];
            correctedLogits[sampleIndex] = correctedLogit;
            if (correctedLogit > maxCorrectedLogit)
                maxCorrectedLogit = correctedLogit;
        }

        float expSum = 0f;
        for (int sampleIndex = 0; sampleIndex < validCount; ++sampleIndex)
            expSum += MathF.Exp(correctedLogits[sampleIndex] - maxCorrectedLogit);

        for (int sampleIndex = 0; sampleIndex < validCount; ++sampleIndex)
            position.OldSampledMoveProbs[sampleIndex] = MathF.Exp(correctedLogits[sampleIndex] - maxCorrectedLogit) / MathF.Max(expSum, 1e-9f);
    }


    static int BuildNegativeMovePool(long remainingHands, long remainingDiscards, int targetMoveIndex, int[] output)
    {
        int count = 0;
        if (remainingHands > 0 && remainingDiscards > 0)
        {
            for (int moveIndex = 0; moveIndex < PpoPolicyValueModel.MoveCount; ++moveIndex)
            {
                if (moveIndex == targetMoveIndex)
                    continue;

                output[count++] = moveIndex;
            }

            return count;
        }

        int actionOffset = remainingHands > 0 ? 0 : 1;
        for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
        {
            int moveIndex = handIndex * 2 + actionOffset;
            if (moveIndex == targetMoveIndex)
                continue;

            output[count++] = moveIndex;
        }

        return count;
    }


    static Tensor BuildIllegalMoveMask(Tensor remainingHands, Tensor remainingDiscards)
    {
        using var scope = NewDisposeScope();

        Tensor noHandsMask = remainingHands
            .eq(0)
            .to_type(ScalarType.Float32)
            .unsqueeze(-1);
        Tensor noDiscardsMask = remainingDiscards
            .eq(0)
            .to_type(ScalarType.Float32)
            .unsqueeze(-1);
        Tensor playMask = noHandsMask.expand(remainingHands.size(0), PpoPolicyValueModel.UseableHandCount);
        Tensor discardMask = noDiscardsMask.expand(remainingHands.size(0), PpoPolicyValueModel.UseableHandCount);
        Tensor stacked = stack([playMask, discardMask], dim: 2).view([remainingHands.size(0), PpoPolicyValueModel.MoveCount]);
        Tensor illegalMask = stacked * -1e9f;
        illegalMask.MoveToOuterDisposeScope();
        return illegalMask;
    }


    static int[] BuildShuffledIndices(int count, Random random)
    {
        int[] indices = new int[count];
        for (int index = 0; index < count; ++index)
            indices[index] = index;

        for (int index = indices.Length - 1; index > 0; --index)
        {
            int swapIndex = random.Next(index + 1);
            (indices[index], indices[swapIndex]) = (indices[swapIndex], indices[index]);
        }

        return indices;
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


    static float GetEntropy(ReadOnlySpan<float> probs)
    {
        float entropy = 0f;
        for (int moveIndex = 0; moveIndex < probs.Length; ++moveIndex)
        {
            float prob = probs[moveIndex];
            if (prob <= 0f)
                continue;

            entropy -= prob * MathF.Log(MathF.Max(prob, 1e-9f));
        }

        return entropy;
    }


    static float GetPpoLearningRate(int continuationStep)
    {
        return PpoContinuationLearningRate;
    }


    static void SetOptimizerLearningRate(AdamW optimizer, float learningRate)
    {
        foreach (TorchSharp.torch.optim.ILearningRateController learningRateController in optimizer.ParamGroups)
        {
            learningRateController.LearningRate = learningRate;
            learningRateController.InitialLearningRate = learningRate;
        }
    }


    public static float GetStandardReward(GameState gameState)
    {
        if (gameState.ScoringState.CurrentRoundTotalScore >= 300f)
            return 1f + gameState.HandState.RemainingHands * 0.2f;

        return (float)gameState.ScoringState.CurrentRoundTotalScore / 1000f;
    }


    static void WriteMetricsCsv(string filePath, IReadOnlyList<StepMetrics> metrics)
    {
        CSVBuilder output = new();
        for (int metricIndex = 0; metricIndex < metrics.Count; ++metricIndex)
        {
            StepMetrics metric = metrics[metricIndex];
            output
                .NextRow()
                .SetCell("step", metric.Step)
                .SetCell("wall_clock_seconds", metric.WallClockSeconds)
                .SetCell("average_reward", metric.AverageReward)
                .SetCell("average_move_entropy", metric.AverageMoveEntropy)
                .SetCell("value_mse_mean", metric.ValueMseMean)
                .SetCell("policy_loss_mean", metric.PolicyLossMean)
                .SetCell("completed_game_count", metric.CompletedGameCount)
                .SetCell("learning_rate", metric.LearningRate);
        }

        File.WriteAllText(filePath, output.ToString());
    }


    static List<StepMetrics> LoadExistingMetrics(string filePath, int maxStepInclusive)
    {
        List<StepMetrics> metrics = [];
        if (!File.Exists(filePath))
            return metrics;

        string[] lines = File.ReadAllLines(filePath);
        for (int lineIndex = 1; lineIndex < lines.Length; ++lineIndex)
        {
            string line = lines[lineIndex].Trim();
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] cells = line.Split(',');
            if (cells.Length < 7)
                continue;

            int step = int.Parse(cells[0], CultureInfo.InvariantCulture);
            if (step > maxStepInclusive)
                continue;

            metrics.Add(new(
                Step: step,
                WallClockSeconds: float.Parse(cells[1], CultureInfo.InvariantCulture),
                AverageReward: float.Parse(cells[2], CultureInfo.InvariantCulture),
                AverageMoveEntropy: float.Parse(cells[3], CultureInfo.InvariantCulture),
                ValueMseMean: float.Parse(cells[4], CultureInfo.InvariantCulture),
                PolicyLossMean: float.Parse(cells[5], CultureInfo.InvariantCulture),
                CompletedGameCount: int.Parse(cells[6], CultureInfo.InvariantCulture),
                LearningRate: cells.Length > 7 ? float.Parse(cells[7], CultureInfo.InvariantCulture) : 0f,
                ValueReplayCount: 0));
        }

        return metrics;
    }


    static void WriteReadme(string filePath, string commitHash, ExperimentConfig config, string localWeightsDir, ResumeConfig resume)
    {
        string runDate = DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        string resumeLines = string.IsNullOrWhiteSpace(resume.CheckpointPath) ? "" : $"""
24. Resumed from checkpoint: `{resume.CheckpointPath}`
25. Resume step: `{resume.ResumeStep}`
""";
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
7. PPO epsilon: `{config.PpoEpsilon}`
8. Entropy coefficient: `{config.EntropyCoefficient}`
9. Rollout size: `{config.RolloutSize}` positions
10. Parallel rollout games: `{config.RolloutParallelGameCount}`
11. Batch size: `{config.BatchSize}`
12. Training epochs per step: `{config.TrainingEpochsPerStep}`
13. Total PPO steps: `{config.StepCount}`
14. Sampled softmax candidates: `{config.SampledSoftmaxCount}`
15. Policy/value update pattern: `1 policy batch, 1 value batch, then optimizer step`
16. Reward function: `standard game reward`
17. Value target: `average of later position value predictions, with terminal position assigned true reward`
18. Value head: `Linear(512 -> 1)` attached to the trunk residual stream
19. Snapshot frequency: every `{config.SnapshotFrequency}` step(s)
20. Snapshot weights directory: `{localWeightsDir}`
21. Notebook: [analysis.ipynb](analysis.ipynb)
22. Starting hands per round: `{config.InitialHandsPerRound}`
23. Starting discards per round: `{config.InitialDiscardsPerRound}`
24. Deck initializer: `{(config.UseRandomDeckInitializer ? "uniform random 52 cards with replacement" : "default deck")}`
{resumeLines}

# Description
- This experiment runs PPO with the current `512`-wide GELU trunk and `add+GELU` move head architecture.
- Rollouts are sampled on-policy from the full policy distribution, while PPO policy training uses `40`-candidate importance-corrected sampled softmax with q-adjusted old and new candidate distributions.
- The value network trains on the same on-policy minibatches as the policy update, sharing the same trunk forward pass for each sample.
- The CSV tracks cumulative wall clock time, average rollout reward, average move entropy, value-network MSE, policy loss, learning rate, and completed game count for each PPO step.
- The notebook graphs reward, entropy, value MSE, policy loss, wall clock progress, and learning rate over the run.
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
        Dictionary<string, object> notebook = [];
        List<Dictionary<string, object>> cells = [];

        cells.Add(CreateMarkdownCell($"""
# {experimentName}

This notebook visualizes the PPO training metrics stored in `analysis.csv`.
"""));

        cells.Add(CreateCodeCell($"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

csv_path = Path(r"{csvPath}")

def load_rows(path):
    loaded_rows = []
    if str(path) in ("", "."):
        return loaded_rows
    if not path.exists():
        return loaded_rows
    with path.open("r", newline="") as csv_file:
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
            loaded_rows.append(parsed_row)
    return loaded_rows

def rolling_average(values, window_size):
    output = []
    for index in range(len(values)):
        start = max(0, index - window_size + 1)
        window_values = values[start:index + 1]
        output.append(sum(window_values) / len(window_values))
    return output

rows = load_rows(csv_path)

steps = [int(row["step"]) for row in rows]
wall_clock_seconds = [row["wall_clock_seconds"] for row in rows]
average_reward = [row["average_reward"] for row in rows]
average_move_entropy = [row["average_move_entropy"] for row in rows]
value_mse_mean = [row["value_mse_mean"] for row in rows]
policy_loss_mean = [row["policy_loss_mean"] for row in rows]
completed_game_count = [row["completed_game_count"] for row in rows]
learning_rate = [row["learning_rate"] for row in rows]

rolling_window = 5
rolling_reward = rolling_average(average_reward, rolling_window)
rolling_entropy = rolling_average(average_move_entropy, rolling_window)
rolling_value_mse = rolling_average(value_mse_mean, rolling_window)
rolling_learning_rate = rolling_average(learning_rate, rolling_window)
"""));

        cells.Add(CreateCodeCell("""
plt.style.use("seaborn-v0_8-whitegrid")

current_colors = {
    "reward": "#1f8f5f",
    "entropy": "#2a6fdb",
    "value": "#d94b3d",
    "loss": "#7a3db8",
    "wall": "#c97a00",
    "games": "#7a5530",
    "lr": "#1d8fa3",
}
reference_raw_color = "#a0a7b4"
reference_trend_color = "#3b4252"

def style_axis(axis):
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_alpha(0.35)
    axis.spines["bottom"].set_alpha(0.35)

figure, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
figure.patch.set_facecolor("white")

axes[0, 0].plot(steps, average_reward, linewidth=1.15, color=current_colors["reward"], alpha=0.24, label="Current raw")
axes[0, 0].plot(steps, rolling_reward, linewidth=2.8, color=current_colors["reward"], label="Current 5-step trend")
axes[0, 0].set_title("Average Reward vs Step")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Average Reward")
axes[0, 0].set_ylim(1.3, 1.55)
style_axis(axes[0, 0])
axes[0, 0].legend(frameon=False)

axes[0, 1].plot(wall_clock_seconds, average_reward, linewidth=1.15, color=current_colors["reward"], alpha=0.24, label="Current raw")
axes[0, 1].plot(wall_clock_seconds, rolling_reward, linewidth=2.8, color=current_colors["reward"], label="Current 5-step trend")
axes[0, 1].set_title("Average Reward vs Wall Clock")
axes[0, 1].set_xlabel("Seconds")
axes[0, 1].set_ylabel("Average Reward")
axes[0, 1].set_ylim(1.3, 1.55)
style_axis(axes[0, 1])
axes[0, 1].legend(frameon=False)

axes[1, 0].plot(steps, average_move_entropy, linewidth=1.15, color=current_colors["entropy"], alpha=0.24, label="Current raw")
axes[1, 0].plot(steps, rolling_entropy, linewidth=2.8, color=current_colors["entropy"], label="Current 5-step trend")
axes[1, 0].set_title("Average Move Entropy vs Step")
axes[1, 0].set_xlabel("Step")
axes[1, 0].set_ylabel("Entropy")
style_axis(axes[1, 0])
axes[1, 0].legend(frameon=False)

axes[1, 1].plot(steps, value_mse_mean, linewidth=1.15, color=current_colors["value"], alpha=0.24, label="Current raw")
axes[1, 1].plot(steps, rolling_value_mse, linewidth=2.8, color=current_colors["value"], label="Current 5-step trend")
axes[1, 1].set_title("Value MSE vs Step")
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("MSE")
axes[1, 1].set_yscale("log")
style_axis(axes[1, 1])
axes[1, 1].legend(frameon=False)

axes[2, 0].plot(steps, policy_loss_mean, linewidth=2.2, color=current_colors["loss"], label="Current")
axes[2, 0].set_title("Policy Loss vs Step")
axes[2, 0].set_xlabel("Step")
axes[2, 0].set_ylabel("Loss")
style_axis(axes[2, 0])
axes[2, 0].legend(frameon=False)

axes[2, 1].plot(steps, wall_clock_seconds, linewidth=2.2, color=current_colors["wall"], label="Current wall")
axes[2, 1].plot(steps, completed_game_count, linewidth=2.2, color=current_colors["games"], label="Current games")
axes[2, 1].set_title("Wall Clock and Completed Games")
axes[2, 1].set_xlabel("Step")
style_axis(axes[2, 1])
axes[2, 1].legend(frameon=False, ncol=2)

axes[3, 0].plot(steps, learning_rate, linewidth=1.15, color=current_colors["lr"], alpha=0.24, label="Current raw")
axes[3, 0].plot(steps, rolling_learning_rate, linewidth=2.8, color=current_colors["lr"], label="Current 5-step trend")
axes[3, 0].set_title("Learning Rate vs Step")
axes[3, 0].set_xlabel("Step")
axes[3, 0].set_ylabel("Learning Rate")
style_axis(axes[3, 0])
axes[3, 0].legend(frameon=False)

axes[3, 1].plot(wall_clock_seconds, learning_rate, linewidth=1.15, color=current_colors["lr"], alpha=0.24, label="Current raw")
axes[3, 1].plot(wall_clock_seconds, rolling_learning_rate, linewidth=2.8, color=current_colors["lr"], label="Current 5-step trend")
axes[3, 1].set_title("Learning Rate vs Wall Clock")
axes[3, 1].set_xlabel("Seconds")
axes[3, 1].set_ylabel("Learning Rate")
style_axis(axes[3, 1])
axes[3, 1].legend(frameon=False)

figure.suptitle("PPO Training Overview", fontsize=16, y=1.02)
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

public sealed class GameTurnTrace
{
    public int TurnIndex { get; init; }
    public string StateText { get; init; }
    public float Score { get; init; }
    public int RemainingHands { get; init; }
    public int RemainingDiscards { get; init; }
    public int RemainingDeck { get; init; }
    public float ThresholdValue { get; init; }
    public float ElapsedSeconds { get; init; }
    public int TotalTrajectoryCount { get; init; }
    public int ChosenMoveIndex { get; init; }
    public string ChosenMoveText { get; init; }
    public string TopMoveText { get; init; }
    public string ConsideredMovesText { get; init; }
    public List<CandidateMoveTrace> Candidates { get; init; } = [];
}

public sealed class CandidateMoveTrace
{
    public int OriginalPolicyRank { get; init; }
    public float PolicyProbability { get; init; }
    public int MoveIndex { get; init; }
    public string MoveText { get; init; }
    public bool IsTopMove { get; init; }
    public bool IsConsidered { get; set; } = true;
    public RunningStats TrajectoryStats { get; } = new();
    public List<TrajectorySampleTrace> TrajectorySamples { get; } = [];
}

public readonly record struct TrajectorySampleTrace(float Value, string HashPath);

public sealed class RunningStats
{
    float _m2;

    public int Count { get; private set; }

    public float Mean { get; private set; }

    public float SampleVariance => Count > 1 ? _m2 / (Count - 1) : 0f;

    public float SampleStandardDeviation => MathF.Sqrt(MathF.Max(0f, SampleVariance));


    public void Add(float value)
    {
        Count++;
        float delta = value - Mean;
        Mean += delta / Count;
        float delta2 = value - Mean;
        _m2 += delta * delta2;
    }
}

public sealed class TrajectoryPruningAgent : IAgent, IDisposable
{
    static readonly Dictionary<string, int> MoveIndexLookup = BuildMoveIndexLookup();

    readonly PpoPolicyValueModel _model;
    readonly PolicyOnlyAgent _policyAgent;
    readonly FastRandom _random;
    readonly float _entropyLimit;
    readonly int _topMoveTrajectoryCount;
    readonly int _initialOtherMoveTrajectoryCount;
    readonly int _topMoveCount;
    readonly int _maxTrajectoryCount;
    readonly int _additionalTrajectoryCountPerRound;

    public TrajectoryPruningAgent(PpoPolicyValueModel model, float entropyLimit, int randomSeed, int topMoveTrajectoryCount, int initialOtherMoveTrajectoryCount, int topMoveCount, int maxTrajectoryCount, int additionalTrajectoryCountPerRound)
    {
        _model = model;
        _policyAgent = new(model, ownsModel: false);
        _random = new((ulong)randomSeed);
        _entropyLimit = entropyLimit;
        _topMoveTrajectoryCount = topMoveTrajectoryCount;
        _initialOtherMoveTrajectoryCount = initialOtherMoveTrajectoryCount;
        _topMoveCount = topMoveCount;
        _maxTrajectoryCount = maxTrajectoryCount;
        _additionalTrajectoryCountPerRound = additionalTrajectoryCountPerRound;
    }


    public void Dispose()
    {
        _policyAgent.Dispose();
    }


    public void MakeMove(float temp, bool annotatePolicy, params ReadOnlySpan<GameState> gameStates)
    {
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            gameState.AdvanceToNextPlayerChoice();
            if (gameState.GameIsDone)
                continue;

            GameTurnTrace trace = ChooseMoveWithTrace(gameState, turnIndex: stateIndex + 1);
            PolicyOnlyAgent.MoveForIndex(gameState, trace.ChosenMoveIndex).Apply(gameState);
            _ = temp;
            _ = annotatePolicy;
        }
    }


    public float[][] GetPolicy(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        return _policyAgent.GetPolicy(temp, gameStates);
    }


    public bool IsGameDone(GameState gameState)
    {
        return _policyAgent.IsGameDone(gameState);
    }


    public GameTurnTrace ChooseMoveWithTrace(GameState gameState, int turnIndex)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        gameState.AdvanceToNextPlayerChoice();

        RankedMove[] rankedMoves = GetRankedLegalMoves(gameState);
        int candidateCount = Math.Min(_topMoveCount, rankedMoves.Length);
        List<CandidateMoveTrace> candidates = [];
        for (int rankIndex = 0; rankIndex < candidateCount; ++rankIndex)
        {
            RankedMove rankedMove = rankedMoves[rankIndex];
            candidates.Add(new()
            {
                OriginalPolicyRank = rankIndex + 1,
                PolicyProbability = rankedMove.PolicyProbability,
                MoveIndex = rankedMove.MoveIndex,
                MoveText = rankedMove.MoveText,
                IsTopMove = rankIndex == 0,
            });
        }

        byte[] serializedState = SerializeGameState(gameState);

        int totalTrajectoryCount = 0;
        for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
        {
            CandidateMoveTrace candidate = candidates[candidateIndex];
            int sampleCount = candidate.IsTopMove ? _topMoveTrajectoryCount : _initialOtherMoveTrajectoryCount;
            AddTrajectorySamples(gameState.GameData, serializedState, candidate, sampleCount);
            totalTrajectoryCount += sampleCount;
        }

        float thresholdValue = candidates[0].TrajectoryStats.Mean - candidates[0].TrajectoryStats.SampleStandardDeviation;
        PruneCandidates(candidates, thresholdValue);

        while (totalTrajectoryCount < _maxTrajectoryCount && CountConsidered(candidates) > 1)
        {
            for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
            {
                CandidateMoveTrace candidate = candidates[candidateIndex];
                if (!candidate.IsConsidered)
                    continue;

                for (int sampleIndex = 0; sampleIndex < _additionalTrajectoryCountPerRound && totalTrajectoryCount < _maxTrajectoryCount; ++sampleIndex)
                {
                    AddTrajectorySamples(gameState.GameData, serializedState, candidate, 1);
                    totalTrajectoryCount++;
                }
            }

            PruneCandidates(candidates, thresholdValue);
        }

        CandidateMoveTrace chosenMove = candidates
            .Where(candidate => candidate.IsConsidered)
            .OrderByDescending(candidate => candidate.TrajectoryStats.Mean)
            .ThenBy(candidate => candidate.OriginalPolicyRank)
            .First();

        string consideredMovesText = string.Join(
            " | ",
            candidates
                .Where(candidate => candidate.IsConsidered)
                .Select(candidate => $"#{candidate.OriginalPolicyRank} {candidate.MoveText}"));

        return new()
        {
            TurnIndex = turnIndex,
            StateText = Program.DescribeState(gameState),
            Score = (float)gameState.ScoringState.CurrentRoundTotalScore,
            RemainingHands = gameState.HandState.RemainingHands,
            RemainingDiscards = gameState.HandState.RemainingDiscards,
            RemainingDeck = gameState.DeckState.RemainingDeckCardCount,
            ThresholdValue = thresholdValue,
            ElapsedSeconds = (float)stopwatch.Elapsed.TotalSeconds,
            TotalTrajectoryCount = totalTrajectoryCount,
            ChosenMoveIndex = chosenMove.MoveIndex,
            ChosenMoveText = chosenMove.MoveText,
            TopMoveText = candidates[0].MoveText,
            ConsideredMovesText = consideredMovesText,
            Candidates = candidates,
        };
    }


    void AddTrajectorySamples(GameData gameData, byte[] serializedState, CandidateMoveTrace candidate, int sampleCount)
    {
        TrajectorySampleTrace[] rewards = SimulateTrajectories(gameData, serializedState, candidate.MoveIndex, sampleCount);
        for (int rewardIndex = 0; rewardIndex < rewards.Length; ++rewardIndex)
        {
            candidate.TrajectoryStats.Add(rewards[rewardIndex].Value);
            candidate.TrajectorySamples.Add(rewards[rewardIndex]);
        }
    }


    TrajectorySampleTrace[] SimulateTrajectories(GameData gameData, byte[] serializedState, int initialMoveIndex, int sampleCount)
    {
        TrajectorySampleTrace[] rewards = new TrajectorySampleTrace[sampleCount];
        List<TrajectorySimulation> initialSimulations = [];

        for (int simulationIndex = 0; simulationIndex < sampleCount; ++simulationIndex)
        {
            GameState simulationState = CloneGameState(gameData, serializedState);
            PolicyOnlyAgent.MoveForIndex(simulationState, initialMoveIndex).Apply(simulationState);
            simulationState.Reseed();
            simulationState.AdvanceToNextPlayerChoice();

            TrajectorySimulation simulation = new(simulationState, simulationIndex);
            initialSimulations.Add(simulation);
        }

        AddCurrentStateRewards(initialSimulations);

        List<TrajectorySimulation> activeSimulations = [];
        for (int simulationIndex = 0; simulationIndex < initialSimulations.Count; ++simulationIndex)
        {
            TrajectorySimulation simulation = initialSimulations[simulationIndex];
            if (simulation.GameState.GameIsDone)
                rewards[simulation.OutputIndex] = simulation.ToSample();
            else
                activeSimulations.Add(simulation);
        }

        while (activeSimulations.Count > 0)
        {
            GameState[] states = new GameState[activeSimulations.Count];
            for (int stateIndex = 0; stateIndex < activeSimulations.Count; ++stateIndex)
                states[stateIndex] = activeSimulations[stateIndex].GameState;

            float[][] policies = _policyAgent.GetPolicy(temp: 1f, states);

            for (int stateIndex = 0; stateIndex < activeSimulations.Count; ++stateIndex)
            {
                TrajectorySimulation simulation = activeSimulations[stateIndex];
                float[] policy = policies[stateIndex];
                float entropy = CalculateEntropy(policy);
                int sampledMoveIndex = AgentUtilities.SampleIndex(policy, _random);
                PolicyOnlyAgent.MoveForIndex(simulation.GameState, sampledMoveIndex).Apply(simulation.GameState);
                simulation.GameState.AdvanceToNextPlayerChoice();
                simulation.CumulativeEntropy += entropy;
            }

            AddCurrentStateRewards(activeSimulations);

            List<TrajectorySimulation> nextActiveSimulations = [];
            for (int stateIndex = 0; stateIndex < activeSimulations.Count; ++stateIndex)
            {
                TrajectorySimulation simulation = activeSimulations[stateIndex];
                if (simulation.GameState.GameIsDone || simulation.CumulativeEntropy >= _entropyLimit)
                    rewards[simulation.OutputIndex] = simulation.ToSample();
                else
                    nextActiveSimulations.Add(simulation);
            }

            activeSimulations = nextActiveSimulations;
        }

        return rewards;
    }


    void AddCurrentStateRewards(IReadOnlyList<TrajectorySimulation> simulations)
    {
        List<TrajectorySimulation> nonTerminalSimulations = [];
        for (int simulationIndex = 0; simulationIndex < simulations.Count; ++simulationIndex)
        {
            TrajectorySimulation simulation = simulations[simulationIndex];
            if (simulation.GameState.GameIsDone)
            {
                simulation.AddReward(Program.GetStandardReward(simulation.GameState));
                continue;
            }

            nonTerminalSimulations.Add(simulation);
        }

        if (nonTerminalSimulations.Count == 0)
            return;

        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameStateEmbedder gameStateEmbedder = new(nonTerminalSimulations.Count);
        for (int simulationIndex = 0; simulationIndex < nonTerminalSimulations.Count; ++simulationIndex)
            gameStateEmbedder.AddGameState(nonTerminalSimulations[simulationIndex].GameState);

        GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(PpoPolicyValueModel.EvalDevice);
        Tensor values = _model.GetValues(gameStateTensors).to(CPU);
        float[] valueData = values.data<float>().ToArray();
        gameStateTensors.Dispose();

        for (int simulationIndex = 0; simulationIndex < nonTerminalSimulations.Count; ++simulationIndex)
            nonTerminalSimulations[simulationIndex].AddReward(valueData[simulationIndex]);
    }


    RankedMove[] GetRankedLegalMoves(GameState gameState)
    {
        float[][] policies = _policyAgent.GetPolicy(temp: 1f, gameState);
        float[] policy = policies[0];
        Move[] legalMoves = gameState.GetMoveOptions();
        RankedMove[] rankedMoves = new RankedMove[legalMoves.Length];
        for (int moveIndex = 0; moveIndex < legalMoves.Length; ++moveIndex)
        {
            Move move = legalMoves[moveIndex];
            UseHandMove useHandMove = (UseHandMove)move;
            int policyMoveIndex = GetMoveIndex(useHandMove);
            rankedMoves[moveIndex] = new(
                MoveIndex: policyMoveIndex,
                PolicyProbability: policy[policyMoveIndex],
                MoveText: FormatMove(gameState, move));
        }

        Array.Sort(rankedMoves, static (left, right) => right.PolicyProbability.CompareTo(left.PolicyProbability));
        return rankedMoves;
    }


    static void PruneCandidates(List<CandidateMoveTrace> candidates, float thresholdValue)
    {
        for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
        {
            CandidateMoveTrace candidate = candidates[candidateIndex];
            if (candidate.IsTopMove)
            {
                candidate.IsConsidered = true;
                continue;
            }

            if (!candidate.IsConsidered)
                continue;

            float upperConfidence = candidate.TrajectoryStats.Mean + 2f * candidate.TrajectoryStats.SampleStandardDeviation;
            if (upperConfidence < thresholdValue)
                candidate.IsConsidered = false;
        }
    }


    static int CountConsidered(List<CandidateMoveTrace> candidates)
    {
        int count = 0;
        for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
        {
            if (candidates[candidateIndex].IsConsidered)
                count++;
        }

        return count;
    }


    static float CalculateEntropy(ReadOnlySpan<float> probabilities)
    {
        float entropy = 0f;
        for (int probabilityIndex = 0; probabilityIndex < probabilities.Length; ++probabilityIndex)
        {
            float probability = probabilities[probabilityIndex];
            if (probability <= 0f)
                continue;

            entropy -= probability * MathF.Log(MathF.Max(probability, 1e-9f));
        }

        return entropy;
    }


    static string FormatMove(GameState gameState, Move move)
    {
        if (move is not UseHandMove useHandMove)
            return move.ToString();

        Card[] cards = new Card[useHandMove.CardIndices.Length];
        for (int cardIndex = 0; cardIndex < useHandMove.CardIndices.Length; ++cardIndex)
            cards[cardIndex] = gameState.HandState.Hand[useHandMove.CardIndices[cardIndex]];

        string action = useHandMove.IsDiscard ? "Discard" : "Play";
        string cardText = CardParseUtils.SerializeHand(cards);
        string indexText = string.Join(",", useHandMove.CardIndices);
        return $"{action} Hand: {cardText} [idx:{indexText}]";
    }


    static byte[] SerializeGameState(GameState gameState)
    {
        using MemoryStream stream = new();
        gameState.Serialize(stream);
        return stream.ToArray();
    }


    static GameState CloneGameState(GameData gameData, byte[] serializedState)
    {
        GameState clonedState = new(gameData);
        using MemoryStream stream = new(serializedState, writable: false);
        clonedState.Deserialize(stream);
        return clonedState;
    }


    static int GetMoveIndex(UseHandMove move)
    {
        string key = GetMoveKey(move.CardIndices);
        int handIndex = MoveIndexLookup[key];
        return handIndex * 2 + (move.IsDiscard ? 1 : 0);
    }


    static Dictionary<string, int> BuildMoveIndexLookup()
    {
        Dictionary<string, int> lookup = [];
        for (int handIndex = 0; handIndex < PpoPolicyValueModel.HandCombinations.Length; ++handIndex)
            lookup[GetMoveKey(PpoPolicyValueModel.HandCombinations[handIndex])] = handIndex;
        return lookup;
    }


    static string GetMoveKey(ReadOnlySpan<byte> cardIndices)
    {
        if (cardIndices.Length == 0)
            return "";

        string[] parts = new string[cardIndices.Length];
        for (int index = 0; index < cardIndices.Length; ++index)
            parts[index] = cardIndices[index].ToString(CultureInfo.InvariantCulture);
        return string.Join(",", parts);
    }


    static string GetMoveKey(ReadOnlySpan<int> cardIndices)
    {
        if (cardIndices.Length == 0)
            return "";

        string[] parts = new string[cardIndices.Length];
        for (int index = 0; index < cardIndices.Length; ++index)
            parts[index] = cardIndices[index].ToString(CultureInfo.InvariantCulture);
        return string.Join(",", parts);
    }


    readonly record struct RankedMove(int MoveIndex, float PolicyProbability, string MoveText);

    sealed class TrajectorySimulation
    {
        public readonly GameState GameState;
        public readonly int OutputIndex;
        readonly List<int> _rewardStateHashes = [];

        public float CumulativeEntropy;
        float _rewardTotal;
        int _rewardCount;

        public TrajectorySimulation(GameState gameState, int outputIndex)
        {
            GameState = gameState;
            OutputIndex = outputIndex;
        }

        public float MeanReward => _rewardCount == 0 ? 0f : _rewardTotal / _rewardCount;


        public void AddReward(float reward)
        {
            _rewardTotal += reward;
            _rewardCount++;
            _rewardStateHashes.Add(GameState.GetHashCode());
        }


        public TrajectorySampleTrace ToSample()
        {
            string[] hashParts = new string[_rewardStateHashes.Count];
            for (int hashIndex = 0; hashIndex < _rewardStateHashes.Count; ++hashIndex)
                hashParts[hashIndex] = _rewardStateHashes[hashIndex].ToString(CultureInfo.InvariantCulture);

            return new(MeanReward, string.Join("|", hashParts));
        }
    }
}

public readonly record struct ExperimentConfig
(
    string ExperimentName,
    int RolloutSize,
    int RolloutParallelGameCount,
    int BatchSize,
    int TrainingEpochsPerStep,
    int StepCount,
    int SampledSoftmaxCount,
    float LearningRate,
    float AdamBeta1,
    float AdamBeta2,
    float WeightDecay,
    float PpoEpsilon,
    float EntropyCoefficient,
    int ValueReplayBufferCapacity,
    int SnapshotFrequency,
    int RandomSeed,
    int InitialHandsPerRound,
    int InitialDiscardsPerRound,
    bool UseRandomDeckInitializer,
    string ResumeSourceExperimentName,
    string NotebookReferenceExperimentName
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

public readonly record struct StepMetrics
(
    int Step,
    float WallClockSeconds,
    float AverageReward,
    float AverageMoveEntropy,
    float ValueMseMean,
    float PolicyLossMean,
    int CompletedGameCount,
    float LearningRate,
    int ValueReplayCount
);

public readonly record struct TrainingMetrics
(
    float PolicyLossMean,
    float ValueMseMean
);

public sealed class PpoRolloutDataset : IDisposable
{
    readonly byte[,] _fullHand;
    readonly byte[,] _remainingDeck;
    readonly byte[] _remainingHands;
    readonly byte[] _remainingDiscards;
    readonly float[] _score;
    readonly float[,] _useHandScores;
    readonly long[,] _sampledMoveIndices;
    readonly float[,] _oldSampledMoveProbs;
    readonly float[,] _sampledMoveLogQ;
    readonly float[,] _sampledMoveValidMask;
    readonly float[] _valueTargets;

    public PpoRolloutDataset(int capacity, int sampledSoftmaxCount)
    {
        Capacity = capacity;
        _fullHand = new byte[capacity, GameData.HandSize];
        _remainingDeck = new byte[capacity, 52];
        _remainingHands = new byte[capacity];
        _remainingDiscards = new byte[capacity];
        _score = new float[capacity];
        _useHandScores = new float[capacity, PpoPolicyValueModel.UseableHandCount];
        _sampledMoveIndices = new long[capacity, sampledSoftmaxCount];
        _oldSampledMoveProbs = new float[capacity, sampledSoftmaxCount];
        _sampledMoveLogQ = new float[capacity, sampledSoftmaxCount];
        _sampledMoveValidMask = new float[capacity, sampledSoftmaxCount];
        _valueTargets = new float[capacity];
    }

    public int Capacity { get; }

    public int Count { get; private set; }

    public float AverageReward { get; private set; }

    public float AverageMoveEntropy { get; private set; }

    public int CompletedGameCount { get; private set; }


    public void Dispose()
    {
    }


    public void AddSample(TrajectoryPosition position, float valueTarget)
    {
        int sampleIndex = Count;
        for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
            _fullHand[sampleIndex, cardIndex] = (byte)position.FullHand[cardIndex];

        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
            _remainingDeck[sampleIndex, cardIndex] = (byte)position.RemainingDeck[cardIndex];

        _remainingHands[sampleIndex] = (byte)position.RemainingHands;
        _remainingDiscards[sampleIndex] = (byte)position.RemainingDiscards;
        _score[sampleIndex] = position.Score;
        _valueTargets[sampleIndex] = valueTarget;

        for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
            _useHandScores[sampleIndex, handIndex] = position.UseHandScores[handIndex];

        for (int sampleMoveIndex = 0; sampleMoveIndex < position.SampledMoveIndices.Length; ++sampleMoveIndex)
        {
            _sampledMoveIndices[sampleIndex, sampleMoveIndex] = position.SampledMoveIndices[sampleMoveIndex];
            _oldSampledMoveProbs[sampleIndex, sampleMoveIndex] = position.OldSampledMoveProbs[sampleMoveIndex];
            _sampledMoveLogQ[sampleIndex, sampleMoveIndex] = position.SampledMoveLogQ[sampleMoveIndex];
            _sampledMoveValidMask[sampleIndex, sampleMoveIndex] = position.SampledMoveValidMask[sampleMoveIndex];
        }

        Count++;
    }


    public void SetMetrics(float averageReward, float averageMoveEntropy, int completedGameCount)
    {
        AverageReward = averageReward;
        AverageMoveEntropy = averageMoveEntropy;
        CompletedGameCount = completedGameCount;
    }


    public void FillBatch(int[] shuffledIndices, int batchStart, int batchSize, PpoMiniBatchBuffers batchBuffers)
    {
        for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
        {
            int sampleIndex = shuffledIndices[batchStart + batchIndex];

            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                batchBuffers.FullHand[batchIndex, cardIndex] = _fullHand[sampleIndex, cardIndex];

            for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                batchBuffers.RemainingDeck[batchIndex, cardIndex] = _remainingDeck[sampleIndex, cardIndex];

            batchBuffers.RemainingHands[batchIndex] = _remainingHands[sampleIndex];
            batchBuffers.RemainingDiscards[batchIndex] = _remainingDiscards[sampleIndex];
            batchBuffers.Score[batchIndex, 0] = _score[sampleIndex];
            batchBuffers.ValueTargets[batchIndex] = _valueTargets[sampleIndex];

            for (int handIndex = 0; handIndex < PpoPolicyValueModel.UseableHandCount; ++handIndex)
                batchBuffers.UseHandScores[batchIndex, handIndex] = _useHandScores[sampleIndex, handIndex];

            for (int sampleMoveIndex = 0; sampleMoveIndex < batchBuffers.SampledSoftmaxCount; ++sampleMoveIndex)
            {
                batchBuffers.SampledMoveIndices[batchIndex, sampleMoveIndex] = _sampledMoveIndices[sampleIndex, sampleMoveIndex];
                batchBuffers.OldSampledMoveProbs[batchIndex, sampleMoveIndex] = _oldSampledMoveProbs[sampleIndex, sampleMoveIndex];
                batchBuffers.SampledMoveLogQ[batchIndex, sampleMoveIndex] = _sampledMoveLogQ[sampleIndex, sampleMoveIndex];
                batchBuffers.SampledMoveValidMask[batchIndex, sampleMoveIndex] = _sampledMoveValidMask[sampleIndex, sampleMoveIndex];
            }
        }

        batchBuffers.RefreshTensors();
    }


    public void AddToValueReplayBuffer(ValueReplayBuffer replayBuffer)
    {
        for (int sampleIndex = 0; sampleIndex < Count; ++sampleIndex)
        {
            replayBuffer.AddSample(
                fullHand: _fullHand,
                remainingDeck: _remainingDeck,
                sampleIndex: sampleIndex,
                remainingHands: _remainingHands[sampleIndex],
                remainingDiscards: _remainingDiscards[sampleIndex],
                score: _score[sampleIndex],
                valueTarget: _valueTargets[sampleIndex]);
        }
    }
}

public sealed class ValueReplayBuffer
{
    readonly byte[,] _fullHand;
    readonly byte[,] _remainingDeck;
    readonly byte[] _remainingHands;
    readonly byte[] _remainingDiscards;
    readonly float[] _score;
    readonly float[] _valueTargets;
    int _nextInsertIndex;

    public ValueReplayBuffer(int capacity)
    {
        Capacity = capacity;
        _fullHand = new byte[capacity, GameData.HandSize];
        _remainingDeck = new byte[capacity, 52];
        _remainingHands = new byte[capacity];
        _remainingDiscards = new byte[capacity];
        _score = new float[capacity];
        _valueTargets = new float[capacity];
    }

    public int Capacity { get; }

    public int Count { get; private set; }


    public void AddSample(byte[,] fullHand, byte[,] remainingDeck, int sampleIndex, byte remainingHands, byte remainingDiscards, float score, float valueTarget)
    {
        for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
            _fullHand[_nextInsertIndex, cardIndex] = fullHand[sampleIndex, cardIndex];

        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
            _remainingDeck[_nextInsertIndex, cardIndex] = remainingDeck[sampleIndex, cardIndex];

        _remainingHands[_nextInsertIndex] = remainingHands;
        _remainingDiscards[_nextInsertIndex] = remainingDiscards;
        _score[_nextInsertIndex] = score;
        _valueTargets[_nextInsertIndex] = valueTarget;

        _nextInsertIndex = (_nextInsertIndex + 1) % Capacity;
        if (Count < Capacity)
            Count++;
    }


    public void FillBatch(Random random, ValueMiniBatchBuffers batchBuffers)
    {
        for (int batchIndex = 0; batchIndex < batchBuffers.BatchSize; ++batchIndex)
        {
            int sampleIndex = random.Next(Count);

            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                batchBuffers.FullHand[batchIndex, cardIndex] = _fullHand[sampleIndex, cardIndex];

            for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                batchBuffers.RemainingDeck[batchIndex, cardIndex] = _remainingDeck[sampleIndex, cardIndex];

            batchBuffers.RemainingHands[batchIndex] = _remainingHands[sampleIndex];
            batchBuffers.RemainingDiscards[batchIndex] = _remainingDiscards[sampleIndex];
            batchBuffers.Score[batchIndex, 0] = _score[sampleIndex];
            batchBuffers.ValueTargets[batchIndex] = _valueTargets[sampleIndex];
        }

        batchBuffers.RefreshTensors();
    }
}

public sealed class PpoMiniBatchBuffers : IDisposable
{
    readonly int _batchSize;

    public readonly long[,] FullHand;
    public readonly long[,] RemainingDeck;
    public readonly long[] RemainingHands;
    public readonly long[] RemainingDiscards;
    public readonly float[,] Score;
    public readonly float[,] UseHandScores;
    public readonly long[,] SampledMoveIndices;
    public readonly float[,] OldSampledMoveProbs;
    public readonly float[,] SampledMoveLogQ;
    public readonly float[,] SampledMoveValidMask;
    public readonly float[] ValueTargets;

    public PpoMiniBatch Batch;

    public PpoMiniBatchBuffers(int batchSize, int sampledSoftmaxCount)
    {
        _batchSize = batchSize;
        SampledSoftmaxCount = sampledSoftmaxCount;
        FullHand = new long[_batchSize, GameData.HandSize];
        RemainingDeck = new long[_batchSize, 52];
        RemainingHands = new long[_batchSize];
        RemainingDiscards = new long[_batchSize];
        Score = new float[_batchSize, 1];
        UseHandScores = new float[_batchSize, PpoPolicyValueModel.UseableHandCount];
        SampledMoveIndices = new long[_batchSize, sampledSoftmaxCount];
        OldSampledMoveProbs = new float[_batchSize, sampledSoftmaxCount];
        SampledMoveLogQ = new float[_batchSize, sampledSoftmaxCount];
        SampledMoveValidMask = new float[_batchSize, sampledSoftmaxCount];
        ValueTargets = new float[_batchSize];
        Batch = new();
        RefreshTensors();
    }

    public int BatchSize => _batchSize;

    public int SampledSoftmaxCount { get; }


    public void Dispose()
    {
        Batch.Dispose();
    }


    public void RefreshTensors()
    {
        Batch.Dispose();
        Batch = new()
        {
            StateTensors = new()
            {
                FullHand = tensor(FullHand, dtype: ScalarType.Int64),
                RemainingDeck = tensor(RemainingDeck, dtype: ScalarType.Int64),
                RemainingHands = tensor(RemainingHands, dtype: ScalarType.Int64),
                RemainingDiscards = tensor(RemainingDiscards, dtype: ScalarType.Int64),
                Score = tensor(Score, dtype: ScalarType.Float32),
            },
            UseHandTensors = new()
            {
                Score = tensor(UseHandScores, dtype: ScalarType.Float32),
            },
            SampledMoveIndices = tensor(SampledMoveIndices, dtype: ScalarType.Int64),
            OldSampledMoveProbs = tensor(OldSampledMoveProbs, dtype: ScalarType.Float32),
            SampledMoveLogQ = tensor(SampledMoveLogQ, dtype: ScalarType.Float32),
            SampledMoveValidMask = tensor(SampledMoveValidMask, dtype: ScalarType.Float32),
            ValueTargets = tensor(ValueTargets, dtype: ScalarType.Float32),
        };
    }
}

public sealed class PpoMiniBatch : Ramen.AgentTools.ITensorGroup
{
    public GameStateTensors StateTensors;
    public UseHandTensors UseHandTensors;
    public Tensor SampledMoveIndices;
    public Tensor OldSampledMoveProbs;
    public Tensor SampledMoveLogQ;
    public Tensor SampledMoveValidMask;
    public Tensor ValueTargets;
}

public sealed class ValueMiniBatchBuffers : IDisposable
{
    readonly int _batchSize;

    public readonly long[,] FullHand;
    public readonly long[,] RemainingDeck;
    public readonly long[] RemainingHands;
    public readonly long[] RemainingDiscards;
    public readonly float[,] Score;
    public readonly float[] ValueTargets;

    public ValueMiniBatch Batch;

    public ValueMiniBatchBuffers(int batchSize)
    {
        _batchSize = batchSize;
        FullHand = new long[_batchSize, GameData.HandSize];
        RemainingDeck = new long[_batchSize, 52];
        RemainingHands = new long[_batchSize];
        RemainingDiscards = new long[_batchSize];
        Score = new float[_batchSize, 1];
        ValueTargets = new float[_batchSize];
        Batch = new();
        RefreshTensors();
    }

    public int BatchSize => _batchSize;


    public void Dispose()
    {
        Batch.Dispose();
    }


    public void RefreshTensors()
    {
        Batch.Dispose();
        Batch = new()
        {
            StateTensors = new()
            {
                FullHand = tensor(FullHand, dtype: ScalarType.Int64),
                RemainingDeck = tensor(RemainingDeck, dtype: ScalarType.Int64),
                RemainingHands = tensor(RemainingHands, dtype: ScalarType.Int64),
                RemainingDiscards = tensor(RemainingDiscards, dtype: ScalarType.Int64),
                Score = tensor(Score, dtype: ScalarType.Float32),
            },
            ValueTargets = tensor(ValueTargets, dtype: ScalarType.Float32),
        };
    }
}

public sealed class ValueMiniBatch : Ramen.AgentTools.ITensorGroup
{
    public GameStateTensors StateTensors;
    public Tensor ValueTargets;
}

public sealed class TrajectoryPosition
{
    public readonly long[] FullHand = new long[GameData.HandSize];
    public readonly long[] RemainingDeck = new long[52];
    public readonly float[] UseHandScores = new float[PpoPolicyValueModel.UseableHandCount];
    public readonly long[] SampledMoveIndices;
    public readonly float[] OldSampledMoveProbs;
    public readonly float[] SampledMoveLogQ;
    public readonly float[] SampledMoveValidMask;

    public readonly long RemainingHands;
    public readonly long RemainingDiscards;
    public readonly float Score;
    public float PositionValueEstimate;
    public float PolicyEntropy;

    public TrajectoryPosition(GameState gameState)
    {
        for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
            FullHand[cardIndex] = gameState.HandState.Hand[cardIndex].ToIndex();

        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
            RemainingDeck[cardIndex] = cardIndex < deck.Length ? deck[cardIndex].ToIndex() : 0;

        RemainingHands = gameState.HandState.RemainingHands;
        RemainingDiscards = gameState.HandState.RemainingDiscards;
        Score = (float)gameState.ScoringState.CurrentRoundTotalScore / 300f;

        WriteUseHandScores(gameState, UseHandScores);

        SampledMoveIndices = new long[40];
        OldSampledMoveProbs = new float[40];
        SampledMoveLogQ = new float[40];
        SampledMoveValidMask = new float[40];
    }


    static void WriteUseHandScores(GameState gameState, float[] output)
    {
        float scoreBefore = (float)gameState.ScoringState.CurrentRoundTotalScore;
        int handCardCount = gameState.HandState.HandCardCount;

        for (int handIndex = 0; handIndex < PpoPolicyValueModel.HandCombinations.Length; ++handIndex)
        {
            int[] cardIndices = PpoPolicyValueModel.HandCombinations[handIndex];
            if (cardIndices[^1] >= handCardCount)
            {
                output[handIndex] = 0f;
                continue;
            }

            UseHandMove move = new(isDiscard: false, cardIndices);
            move.Apply(gameState);
            float scoreAfter = (float)gameState.ScoringState.CurrentRoundTotalScore;
            output[handIndex] = scoreAfter / 300f;
            move.Revert(gameState);
        }
    }
}

public sealed class PpoPolicyValueModel : Module, IPolicyModel
{
    public static readonly Device EvalDevice = mps_is_available() ? MPS : CPU;
    public static readonly int[][] HandCombinations = Combinatorics.GetCombinations(
        setSize: GameData.HandSize,
        minSubsetSize: 1,
        maxSubsetSize: GameData.MaxPlayedHandSize);
    public static readonly int UseableHandCount = HandCombinations.Length;
    public static readonly int MoveCount = UseableHandCount * 2;

    static readonly long[,] PlayedHandMaskData = BuildHandMaskData(playedCards: true);
    static readonly long[,] RemainingHandMaskData = BuildHandMaskData(playedCards: false);

    readonly MaskedMeanCardSetEmbedding _fullHandEmbedding = new(embeddingWidth: 128, device: EvalDevice);
    readonly MaskedMeanCardSetEmbedding _remainingDeckEmbedding = new(embeddingWidth: 64, device: EvalDevice);
    readonly MaskedMeanCardSetEmbedding _playedHandEmbedding = new(embeddingWidth: 32, device: EvalDevice);
    readonly MaskedMeanCardSetEmbedding _remainingHandEmbedding = new(embeddingWidth: 32, device: EvalDevice);
    readonly BilinearOneHotScoreEmbedder _scoreEmbedding = new();
    readonly Linear _stateProjection = Linear(TrunkStateFeatureWidth, TrunkWidth, device: EvalDevice);
    readonly ModuleList<GeluResidualBlock> _stateResidualBlocks = new();
    readonly Linear _compressedStateProjection = Linear(TrunkWidth, CompactWidth, device: EvalDevice);
    readonly GELU _stateActivation = GELU();
    readonly Linear _moveOnlyProjection = Linear(MoveOnlyFeatureWidth, CompactWidth, device: EvalDevice);
    readonly GELU _moveMergeActivation = GELU();
    readonly GeluResidualBlock _moveResidualBlock = new(width: CompactWidth, hiddenWidth: CompactWidth, device: EvalDevice);
    readonly Linear _moveOutputProjection = Linear(CompactWidth, 1, device: EvalDevice);
    readonly Linear _valueHead = Linear(TrunkWidth, 1, device: EvalDevice);
    readonly Tensor _playedHandMask;
    readonly Tensor _remainingHandMask;
    readonly bool _useTorchScriptCompile;
    readonly bool _useHalfPrecisionLinearWeights;
    readonly torch.jit.CompilationUnit _policyCompilationUnit = null!;

    const int TrunkWidth = 512;
    const int TrunkHiddenWidth = 1024;
    const int CompactWidth = 256;
    const int ScorePaddingWidth = 32;
    const int CountWidth = 20;
    const int CountPaddingWidth = 20;
    const int ScoreEmbeddingWidth = BilinearOneHotScoreEmbedder.BucketCount;
    const int TrunkStateFeatureWidth = 128 + 64 + ScoreEmbeddingWidth + ScorePaddingWidth + CountWidth + CountPaddingWidth;
    const int MoveOnlyFeatureWidth = 32 + 32 + ScoreEmbeddingWidth + CountWidth;
    const int TrunkResidualBlockCount = 4;

    public PpoPolicyValueModel(bool useTorchScriptCompile = false, bool useHalfPrecisionLinearWeights = false) : base(nameof(PpoPolicyValueModel))
    {
        _useTorchScriptCompile = useTorchScriptCompile;
        _useHalfPrecisionLinearWeights = useHalfPrecisionLinearWeights;
        _playedHandMask = tensor(PlayedHandMaskData, dtype: ScalarType.Int64, device: EvalDevice);
        _remainingHandMask = tensor(RemainingHandMaskData, dtype: ScalarType.Int64, device: EvalDevice);
        _playedHandMask.DetachFromScope();
        _remainingHandMask.DetachFromScope();
        TensorManager.PersistForever(_playedHandMask);
        TensorManager.PersistForever(_remainingHandMask);

        for (int blockIndex = 0; blockIndex < TrunkResidualBlockCount; ++blockIndex)
        {
            _stateResidualBlocks.append(new GeluResidualBlock(
                width: TrunkWidth,
                hiddenWidth: TrunkHiddenWidth,
                device: EvalDevice));
        }

        if (_useTorchScriptCompile)
            _policyCompilationUnit = torch.jit.compile(BuildPolicyTorchScriptSource(_useHalfPrecisionLinearWeights));

        RegisterComponents();
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        if (_useTorchScriptCompile)
            return GetPolicyLogitsTorchScript(gameStateTensors, useHandTensors);

        return GetPolicyLogitsAndValues(gameStateTensors, useHandTensors).logits;
    }


    Tensor GetPolicyLogitsTorchScript(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        using var scope = NewDisposeScope();

        Tensor logits = _policyCompilationUnit.invoke<Tensor>(
            "policy_logits",
            gameStateTensors.FullHand.to(EvalDevice),
            gameStateTensors.RemainingDeck.to(EvalDevice),
            gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64),
            gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64),
            gameStateTensors.Score.to(EvalDevice),
            useHandTensors.Score.to(EvalDevice),
            _playedHandMask,
            _remainingHandMask,
            _fullHandEmbedding.CardEmbedding.weight,
            _remainingDeckEmbedding.CardEmbedding.weight,
            _playedHandEmbedding.CardEmbedding.weight,
            _remainingHandEmbedding.CardEmbedding.weight,
            _stateProjection.weight,
            _stateProjection.bias,
            _stateResidualBlocks[0].LayerNorm.weight,
            _stateResidualBlocks[0].LayerNorm.bias,
            _stateResidualBlocks[0].HiddenProjection.weight,
            _stateResidualBlocks[0].HiddenProjection.bias,
            _stateResidualBlocks[0].OutputProjection.weight,
            _stateResidualBlocks[0].OutputProjection.bias,
            _stateResidualBlocks[1].LayerNorm.weight,
            _stateResidualBlocks[1].LayerNorm.bias,
            _stateResidualBlocks[1].HiddenProjection.weight,
            _stateResidualBlocks[1].HiddenProjection.bias,
            _stateResidualBlocks[1].OutputProjection.weight,
            _stateResidualBlocks[1].OutputProjection.bias,
            _stateResidualBlocks[2].LayerNorm.weight,
            _stateResidualBlocks[2].LayerNorm.bias,
            _stateResidualBlocks[2].HiddenProjection.weight,
            _stateResidualBlocks[2].HiddenProjection.bias,
            _stateResidualBlocks[2].OutputProjection.weight,
            _stateResidualBlocks[2].OutputProjection.bias,
            _stateResidualBlocks[3].LayerNorm.weight,
            _stateResidualBlocks[3].LayerNorm.bias,
            _stateResidualBlocks[3].HiddenProjection.weight,
            _stateResidualBlocks[3].HiddenProjection.bias,
            _stateResidualBlocks[3].OutputProjection.weight,
            _stateResidualBlocks[3].OutputProjection.bias,
            _compressedStateProjection.weight,
            _compressedStateProjection.bias,
            _moveOnlyProjection.weight,
            _moveOnlyProjection.bias,
            _moveResidualBlock.LayerNorm.weight,
            _moveResidualBlock.LayerNorm.bias,
            _moveResidualBlock.HiddenProjection.weight,
            _moveResidualBlock.HiddenProjection.bias,
            _moveResidualBlock.OutputProjection.weight,
            _moveResidualBlock.OutputProjection.bias,
            _moveOutputProjection.weight,
            _moveOutputProjection.bias);

        logits.MoveToOuterDisposeScope();
        return logits;
    }


    public (Tensor logits, Tensor values) GetPolicyLogitsAndValues(GameStateTensors gameStateTensors, UseHandTensors useHandTensors)
    {
        using var scope = NewDisposeScope();

        (Tensor compactState, Tensor trunkState) = EncodeState(gameStateTensors);
        Tensor playedHandEmbeddings = BuildHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _playedHandMask,
            embedder: _playedHandEmbedding);
        Tensor remainingHandEmbeddings = BuildHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _remainingHandMask,
            embedder: _remainingHandEmbedding);
        Tensor preScoreEmbedding = _scoreEmbedding
            .forward(gameStateTensors.Score.to(EvalDevice) * 300f)
            .squeeze(1);
        Tensor postPlayScoreEmbedding = _scoreEmbedding.forward(useHandTensors.Score.to(EvalDevice) * 300f);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor playPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState((remainingHands - 1).clamp_min(0), remainingDiscards),
            moveCount: UseableHandCount);
        Tensor discardPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState(remainingHands, (remainingDiscards - 1).clamp_min(0)),
            moveCount: UseableHandCount);
        Tensor compactStateExpanded = ExpandAcrossMoves(compactState, UseableHandCount);
        Tensor preScoreExpanded = ExpandAcrossMoves(preScoreEmbedding, UseableHandCount);

        Tensor playFeatures = cat(
            [
                compactStateExpanded,
                playedHandEmbeddings,
                remainingHandEmbeddings,
                postPlayScoreEmbedding,
                playPostCountEmbedding,
            ],
            dim: -1);
        Tensor discardFeatures = cat(
            [
                compactStateExpanded,
                playedHandEmbeddings,
                remainingHandEmbeddings,
                preScoreExpanded,
                discardPostCountEmbedding,
            ],
            dim: -1);
        Tensor playLogits = ScoreMoveFeatures(playFeatures).squeeze(-1);
        Tensor discardLogits = ScoreMoveFeatures(discardFeatures).squeeze(-1);
        Tensor logits = stack([playLogits, discardLogits], dim: 2).view([playLogits.size(0), MoveCount]);
        Tensor values = _valueHead.forward(trunkState).squeeze(-1);
        logits.MoveToOuterDisposeScope();
        values.MoveToOuterDisposeScope();
        return (logits, values);
    }


    public Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor moveIndices)
    {
        return GetSelectedPolicyLogitsAndValues(
            gameStateTensors: gameStateTensors,
            useHandTensors: useHandTensors,
            moveIndices: moveIndices).logits;
    }


    public (Tensor logits, Tensor values) GetSelectedPolicyLogitsAndValues(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor moveIndices)
    {
        using var scope = NewDisposeScope();

        Tensor selectedMoveIndices = moveIndices.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor selectedHandIndices = selectedMoveIndices.div(2).to_type(ScalarType.Int64);
        Tensor selectedActionIndices = selectedMoveIndices.remainder(2).to_type(ScalarType.Int64);
        int selectedMoveCount = (int)selectedMoveIndices.size(1);

        (Tensor compactState, Tensor trunkState) = EncodeState(gameStateTensors);
        Tensor selectedPlayedHandEmbeddings = BuildSelectedHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _playedHandMask,
            embedder: _playedHandEmbedding,
            selectedHandIndices: selectedHandIndices);
        Tensor selectedRemainingHandEmbeddings = BuildSelectedHandEmbeddings(
            fullHand: gameStateTensors.FullHand.to(EvalDevice),
            handMask: _remainingHandMask,
            embedder: _remainingHandEmbedding,
            selectedHandIndices: selectedHandIndices);
        Tensor preScoreEmbedding = _scoreEmbedding
            .forward(gameStateTensors.Score.to(EvalDevice) * 300f)
            .squeeze(1);
        Tensor selectedPostPlayScoreEmbedding = _scoreEmbedding
            .forward(useHandTensors.Score.to(EvalDevice).gather(dim: 1, index: selectedHandIndices) * 300f);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64);
        Tensor playPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState((remainingHands - 1).clamp_min(0), remainingDiscards),
            moveCount: selectedMoveCount);
        Tensor discardPostCountEmbedding = ExpandAcrossMoves(
            EncodeCountState(remainingHands, (remainingDiscards - 1).clamp_min(0)),
            moveCount: selectedMoveCount);
        Tensor compactStateExpanded = ExpandAcrossMoves(compactState, selectedMoveCount);
        Tensor preScoreExpanded = ExpandAcrossMoves(preScoreEmbedding, selectedMoveCount);

        Tensor playFeatures = cat(
            [
                compactStateExpanded,
                selectedPlayedHandEmbeddings,
                selectedRemainingHandEmbeddings,
                selectedPostPlayScoreEmbedding,
                playPostCountEmbedding,
            ],
            dim: -1);
        Tensor discardFeatures = cat(
            [
                compactStateExpanded,
                selectedPlayedHandEmbeddings,
                selectedRemainingHandEmbeddings,
                preScoreExpanded,
                discardPostCountEmbedding,
            ],
            dim: -1);
        Tensor discardActionMask = selectedActionIndices.to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor mixedFeatures = playFeatures * (1f - discardActionMask) + discardFeatures * discardActionMask;
        Tensor selectedLogits = ScoreMoveFeatures(mixedFeatures).squeeze(-1);
        Tensor values = _valueHead.forward(trunkState).squeeze(-1);

        selectedLogits.MoveToOuterDisposeScope();
        values.MoveToOuterDisposeScope();
        return (selectedLogits, values);
    }


    public Tensor GetValues(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        (_, Tensor trunkState) = EncodeState(gameStateTensors);
        Tensor values = _valueHead.forward(trunkState).squeeze(-1);
        values.MoveToOuterDisposeScope();
        return values;
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }


    Tensor ScoreMoveFeatures(Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor compactStream = input[.., .., ..CompactWidth];
        Tensor moveOnlyFeatures = input[.., .., CompactWidth..];
        Tensor residualStream = compactStream + _moveOnlyProjection.forward(moveOnlyFeatures);
        residualStream = _moveMergeActivation.forward(residualStream);
        residualStream = _moveResidualBlock.forward(residualStream);
        Tensor output = _moveOutputProjection.forward(residualStream);
        output.MoveToOuterDisposeScope();
        return output;
    }


    (Tensor compactState, Tensor trunkState) EncodeState(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor fullHandEmbedding = _fullHandEmbedding.forward(gameStateTensors.FullHand.to(EvalDevice));
        Tensor remainingDeckEmbedding = _remainingDeckEmbedding.forward(gameStateTensors.RemainingDeck.to(EvalDevice));
        Tensor scoreEmbedding = _scoreEmbedding.forward(gameStateTensors.Score.to(EvalDevice) * 300f).squeeze(1);
        Tensor countEmbedding = EncodeCountState(
            remainingHands: gameStateTensors.RemainingHands.to(EvalDevice).to_type(ScalarType.Int64),
            remainingDiscards: gameStateTensors.RemainingDiscards.to(EvalDevice).to_type(ScalarType.Int64));
        Tensor paddedScoreEmbedding = PadLastDimWithZeros(scoreEmbedding, ScorePaddingWidth);
        Tensor paddedCountEmbedding = PadLastDimWithZeros(countEmbedding, CountPaddingWidth);

        Tensor stateFeatures = cat(
            [
                fullHandEmbedding,
                remainingDeckEmbedding,
                paddedScoreEmbedding,
                paddedCountEmbedding,
            ],
            dim: -1);
        Tensor trunkState = _stateProjection.forward(stateFeatures);
        for (int blockIndex = 0; blockIndex < _stateResidualBlocks.Count; ++blockIndex)
            trunkState = _stateResidualBlocks[blockIndex].forward(trunkState);

        Tensor compactState = _stateActivation.forward(_compressedStateProjection.forward(trunkState));
        compactState.MoveToOuterDisposeScope();
        trunkState.MoveToOuterDisposeScope();
        return (compactState, trunkState);
    }


    Tensor PadLastDimWithZeros(Tensor input, int paddingWidth)
    {
        using var scope = NewDisposeScope();

        if (paddingWidth <= 0)
        {
            input.MoveToOuterDisposeScope();
            return input;
        }

        Tensor padding = zeros([input.size(0), paddingWidth], dtype: input.dtype, device: input.device);
        Tensor padded = cat([input, padding], dim: -1);
        padded.MoveToOuterDisposeScope();
        return padded;
    }


    Tensor BuildHandEmbeddings(Tensor fullHand, Tensor handMask, MaskedMeanCardSetEmbedding embedder)
    {
        using var scope = NewDisposeScope();

        int batchSize = (int)fullHand.size(0);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor expandedMask = handMask.unsqueeze(0).expand(batchSize, UseableHandCount, GameData.HandSize);
        Tensor maskedHands = expandedFullHand * expandedMask;
        Tensor embeddings = embedder.forward(maskedHands);
        embeddings.MoveToOuterDisposeScope();
        return embeddings;
    }


    Tensor BuildSelectedHandEmbeddings(Tensor fullHand, Tensor handMask, MaskedMeanCardSetEmbedding embedder, Tensor selectedHandIndices)
    {
        using var scope = NewDisposeScope();

        int batchSize = (int)fullHand.size(0);
        int selectedMoveCount = (int)selectedHandIndices.size(1);
        Tensor expandedFullHand = fullHand.unsqueeze(1).expand(batchSize, selectedMoveCount, GameData.HandSize);
        Tensor selectedMasks = handMask
            .index_select(dim: 0, index: selectedHandIndices.view(-1))
            .view(batchSize, selectedMoveCount, GameData.HandSize);
        Tensor maskedHands = expandedFullHand * selectedMasks;
        Tensor embeddings = embedder.forward(maskedHands);
        embeddings.MoveToOuterDisposeScope();
        return embeddings;
    }


    Tensor EncodeCountState(Tensor remainingHands, Tensor remainingDiscards)
    {
        using var scope = NewDisposeScope();

        Tensor combinedIndex = remainingHands.mul(4).add(remainingDiscards).to_type(ScalarType.Int64);
        Tensor encoded = functional.one_hot(combinedIndex, CountWidth).to_type(ScalarType.Float32);
        encoded.MoveToOuterDisposeScope();
        return encoded;
    }


    static Tensor ExpandAcrossMoves(Tensor tensorToExpand, int moveCount)
    {
        using var scope = NewDisposeScope();

        Tensor expanded = tensorToExpand
            .unsqueeze(1)
            .expand(tensorToExpand.size(0), moveCount, tensorToExpand.size(1));
        expanded.MoveToOuterDisposeScope();
        return expanded;
    }


    static string BuildPolicyTorchScriptSource(bool useHalfPrecisionLinearWeights)
    {
        string source = """
def linear(input, weight, bias):
    return input.matmul(weight.t()) + bias

def layer_norm(input, weight, bias):
    mean = input.mean(dim=-1, keepdim=True)
    centered = input - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normalized = centered / torch.sqrt(variance + 1e-5)
    return normalized * weight + bias

def gelu(input):
    return 0.5 * input * (1.0 + torch.tanh(0.7978845608 * (input + 0.044715 * input * input * input)))

def masked_mean_card_set_embedding(card_set, embedding_weight):
    card_indices = card_set.long()
    valid_mask = card_indices.gt(0).float().unsqueeze(-1)
    flat_indices = card_indices.reshape(-1)
    if card_indices.dim() == 2:
        embedded_flat = embedding_weight.index_select(0, flat_indices)
        embedded_cards = embedded_flat.view(card_indices.size(0), card_indices.size(1), embedding_weight.size(1))
        summed = (embedded_cards * valid_mask).sum(dim=1)
        counts = valid_mask.sum(dim=1).clamp_min(1.0)
        return summed / counts
    embedded_flat = embedding_weight.index_select(0, flat_indices)
    embedded_cards = embedded_flat.view(card_indices.size(0), card_indices.size(1), card_indices.size(2), embedding_weight.size(1))
    summed = (embedded_cards * valid_mask).sum(dim=2)
    counts = valid_mask.sum(dim=2).clamp_min(1.0)
    return summed / counts

def build_hand_embeddings(full_hand, hand_mask, embedding_weight):
    batch_size = full_hand.size(0)
    move_count = hand_mask.size(0)
    expanded_full_hand = full_hand.unsqueeze(1).expand(batch_size, move_count, full_hand.size(1))
    expanded_mask = hand_mask.unsqueeze(0).expand(batch_size, move_count, hand_mask.size(1))
    masked_hands = expanded_full_hand * expanded_mask
    return masked_mean_card_set_embedding(masked_hands, embedding_weight)

def score_embedding(score):
    bucket_count = 30
    bucket_width = 10.0
    bucket_position = torch.clamp(score.float() / bucket_width, 0.0, bucket_count - 1.0)
    lower_index = torch.floor(bucket_position).long()
    upper_index = torch.clamp(lower_index + 1, max=bucket_count - 1)
    upper_weight = bucket_position - lower_index.float()
    lower_weight = 1.0 - upper_weight
    flat_lower_index = lower_index.reshape(-1)
    flat_upper_index = upper_index.reshape(-1)
    eye = torch.eye(bucket_count).to(score.device)
    lower_one_hot = eye.index_select(0, flat_lower_index)
    upper_one_hot = eye.index_select(0, flat_upper_index)
    result = lower_one_hot * lower_weight.reshape(-1).unsqueeze(-1) + upper_one_hot * upper_weight.reshape(-1).unsqueeze(-1)
    if score.dim() == 1:
        return result.view(score.size(0), bucket_count)
    return result.view(score.size(0), score.size(1), bucket_count)

def encode_count_state(remaining_hands, remaining_discards):
    count_width = 20
    combined_index = (remaining_hands * 4 + remaining_discards).long()
    flat_index = combined_index.reshape(-1)
    eye = torch.eye(count_width).to(remaining_hands.device)
    result = eye.index_select(0, flat_index)
    if remaining_hands.dim() == 1:
        return result.view(remaining_hands.size(0), count_width)
    return result.view(remaining_hands.size(0), remaining_hands.size(1), count_width)

def pad_last_dim_with_zeros(input, padding_width: int):
    if padding_width <= 0:
        return input
    padding = (input[:, :1] * 0).expand(input.size(0), padding_width)
    return torch.cat([input, padding], dim=-1)

def gelu_residual_block(
    input,
    layer_norm_weight,
    layer_norm_bias,
    hidden_weight,
    hidden_bias,
    output_weight,
    output_bias,
):
    normalized = layer_norm(input, layer_norm_weight, layer_norm_bias)
    hidden = linear(normalized, hidden_weight, hidden_bias)
    activated = gelu(hidden)
    residual = linear(activated, output_weight, output_bias)
    return input + residual

def score_move_features(
    input,
    move_only_projection_weight,
    move_only_projection_bias,
    move_residual_layer_norm_weight,
    move_residual_layer_norm_bias,
    move_residual_hidden_weight,
    move_residual_hidden_bias,
    move_residual_output_weight,
    move_residual_output_bias,
    move_output_projection_weight,
    move_output_projection_bias,
):
    compact_stream = input[..., :256]
    move_only_features = input[..., 256:]
    residual_stream = compact_stream + linear(move_only_features, move_only_projection_weight, move_only_projection_bias)
    residual_stream = gelu(residual_stream)
    residual_stream = gelu_residual_block(
        residual_stream,
        move_residual_layer_norm_weight,
        move_residual_layer_norm_bias,
        move_residual_hidden_weight,
        move_residual_hidden_bias,
        move_residual_output_weight,
        move_residual_output_bias,
    )
    return linear(residual_stream, move_output_projection_weight, move_output_projection_bias)

def policy_logits(
    full_hand,
    remaining_deck,
    remaining_hands,
    remaining_discards,
    score,
    use_hand_scores,
    played_hand_mask,
    remaining_hand_mask,
    full_hand_embedding_weight,
    remaining_deck_embedding_weight,
    played_hand_embedding_weight,
    remaining_hand_embedding_weight,
    state_projection_weight,
    state_projection_bias,
    state_residual_0_layer_norm_weight,
    state_residual_0_layer_norm_bias,
    state_residual_0_hidden_weight,
    state_residual_0_hidden_bias,
    state_residual_0_output_weight,
    state_residual_0_output_bias,
    state_residual_1_layer_norm_weight,
    state_residual_1_layer_norm_bias,
    state_residual_1_hidden_weight,
    state_residual_1_hidden_bias,
    state_residual_1_output_weight,
    state_residual_1_output_bias,
    state_residual_2_layer_norm_weight,
    state_residual_2_layer_norm_bias,
    state_residual_2_hidden_weight,
    state_residual_2_hidden_bias,
    state_residual_2_output_weight,
    state_residual_2_output_bias,
    state_residual_3_layer_norm_weight,
    state_residual_3_layer_norm_bias,
    state_residual_3_hidden_weight,
    state_residual_3_hidden_bias,
    state_residual_3_output_weight,
    state_residual_3_output_bias,
    compressed_state_projection_weight,
    compressed_state_projection_bias,
    move_only_projection_weight,
    move_only_projection_bias,
    move_residual_layer_norm_weight,
    move_residual_layer_norm_bias,
    move_residual_hidden_weight,
    move_residual_hidden_bias,
    move_residual_output_weight,
    move_residual_output_bias,
    move_output_projection_weight,
    move_output_projection_bias,
):
    full_hand_embedding = masked_mean_card_set_embedding(full_hand, full_hand_embedding_weight)
    remaining_deck_embedding = masked_mean_card_set_embedding(remaining_deck, remaining_deck_embedding_weight)
    played_hand_embeddings = build_hand_embeddings(full_hand, played_hand_mask, played_hand_embedding_weight)
    remaining_hand_embeddings = build_hand_embeddings(full_hand, remaining_hand_mask, remaining_hand_embedding_weight)
    score_embedding_before = score_embedding(score * 300.0).squeeze(1)
    score_embedding_after = score_embedding(use_hand_scores * 300.0)
    remaining_hands_i64 = remaining_hands.long()
    remaining_discards_i64 = remaining_discards.long()
    play_post_count_embedding = encode_count_state((remaining_hands_i64 - 1).clamp_min(0), remaining_discards_i64)
    discard_post_count_embedding = encode_count_state(remaining_hands_i64, (remaining_discards_i64 - 1).clamp_min(0))
    padded_score_embedding = pad_last_dim_with_zeros(score_embedding_before, 32)
    count_embedding = encode_count_state(remaining_hands_i64, remaining_discards_i64)
    padded_count_embedding = pad_last_dim_with_zeros(count_embedding, 20)
    state_features = torch.cat(
        [full_hand_embedding, remaining_deck_embedding, padded_score_embedding, padded_count_embedding],
        dim=-1,
    )
    trunk_state = linear(state_features, state_projection_weight, state_projection_bias)
    trunk_state = gelu_residual_block(
        trunk_state,
        state_residual_0_layer_norm_weight,
        state_residual_0_layer_norm_bias,
        state_residual_0_hidden_weight,
        state_residual_0_hidden_bias,
        state_residual_0_output_weight,
        state_residual_0_output_bias,
    )
    trunk_state = gelu_residual_block(
        trunk_state,
        state_residual_1_layer_norm_weight,
        state_residual_1_layer_norm_bias,
        state_residual_1_hidden_weight,
        state_residual_1_hidden_bias,
        state_residual_1_output_weight,
        state_residual_1_output_bias,
    )
    trunk_state = gelu_residual_block(
        trunk_state,
        state_residual_2_layer_norm_weight,
        state_residual_2_layer_norm_bias,
        state_residual_2_hidden_weight,
        state_residual_2_hidden_bias,
        state_residual_2_output_weight,
        state_residual_2_output_bias,
    )
    trunk_state = gelu_residual_block(
        trunk_state,
        state_residual_3_layer_norm_weight,
        state_residual_3_layer_norm_bias,
        state_residual_3_hidden_weight,
        state_residual_3_hidden_bias,
        state_residual_3_output_weight,
        state_residual_3_output_bias,
    )
    compact_state = gelu(linear(trunk_state, compressed_state_projection_weight, compressed_state_projection_bias))
    move_count = use_hand_scores.size(1)
    compact_state_expanded = compact_state.unsqueeze(1).expand(compact_state.size(0), move_count, compact_state.size(1))
    pre_score_expanded = score_embedding_before.unsqueeze(1).expand(score_embedding_before.size(0), move_count, score_embedding_before.size(1))
    play_features = torch.cat(
        [
            compact_state_expanded,
            played_hand_embeddings,
            remaining_hand_embeddings,
            score_embedding_after,
            play_post_count_embedding.unsqueeze(1).expand(play_post_count_embedding.size(0), move_count, play_post_count_embedding.size(1)),
        ],
        dim=-1,
    )
    discard_features = torch.cat(
        [
            compact_state_expanded,
            played_hand_embeddings,
            remaining_hand_embeddings,
            pre_score_expanded,
            discard_post_count_embedding.unsqueeze(1).expand(discard_post_count_embedding.size(0), move_count, discard_post_count_embedding.size(1)),
        ],
        dim=-1,
    )
    play_logits = score_move_features(
        play_features,
        move_only_projection_weight,
        move_only_projection_bias,
        move_residual_layer_norm_weight,
        move_residual_layer_norm_bias,
        move_residual_hidden_weight,
        move_residual_hidden_bias,
        move_residual_output_weight,
        move_residual_output_bias,
        move_output_projection_weight,
        move_output_projection_bias,
    ).squeeze(-1)
    discard_logits = score_move_features(
        discard_features,
        move_only_projection_weight,
        move_only_projection_bias,
        move_residual_layer_norm_weight,
        move_residual_layer_norm_bias,
        move_residual_hidden_weight,
        move_residual_hidden_bias,
        move_residual_output_weight,
        move_residual_output_bias,
        move_output_projection_weight,
        move_output_projection_bias,
    ).squeeze(-1)
    return torch.stack([play_logits, discard_logits], dim=2).view(play_logits.size(0), -1)

""";
        if (!useHalfPrecisionLinearWeights)
            return source;

        return source.Replace(
            """
def linear(input, weight, bias):
    return input.matmul(weight.t()) + bias
""",
            """
def linear(input, weight, bias):
    input_half = input.half()
    return (input_half.matmul(weight.t()) + bias).float()
""",
            StringComparison.Ordinal);
    }


    static long[,] BuildHandMaskData(bool playedCards)
    {
        long[,] handMask = new long[UseableHandCount, GameData.HandSize];
        for (int handIndex = 0; handIndex < HandCombinations.Length; ++handIndex)
        {
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                handMask[handIndex, cardIndex] = playedCards ? 0 : 1;

            int[] combination = HandCombinations[handIndex];
            for (int cardIndex = 0; cardIndex < combination.Length; ++cardIndex)
                handMask[handIndex, combination[cardIndex]] = playedCards ? 1 : 0;
        }

        return handMask;
    }
}

public sealed class MaskedMeanCardSetEmbedding : Module<Tensor, Tensor>
{
    readonly Embedding _cardEmbedding;

    public MaskedMeanCardSetEmbedding(int embeddingWidth, Device device = null) : base(nameof(MaskedMeanCardSetEmbedding))
    {
        Device targetDevice = device ?? CPU;
        _cardEmbedding = Embedding(Card.RankCount * Card.SuitCount + 1, embeddingWidth, device: targetDevice);

        using var noGrad = no_grad();
        _cardEmbedding.weight[0].fill_(0f);
        RegisterComponents();
    }

    public Embedding CardEmbedding => _cardEmbedding;

    public override Tensor forward(Tensor cardSet)
    {
        using var scope = NewDisposeScope();

        Tensor cardIndices = cardSet.to_type(ScalarType.Int64);
        Tensor validMask = cardIndices.gt(0).to_type(ScalarType.Float32).unsqueeze(-1);
        Tensor embeddedCards = _cardEmbedding.forward(cardIndices);
        Tensor summed = (embeddedCards * validMask).sum(dim: embeddedCards.Dimensions - 2);
        Tensor counts = validMask.sum(dim: validMask.Dimensions - 2).clamp_min(1f);
        Tensor pooled = summed / counts;
        pooled.MoveToOuterDisposeScope();
        return pooled;
    }
}

public sealed class GeluResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _hiddenProjection;
    readonly GELU _activation = GELU();
    readonly Linear _outputProjection;

    public GeluResidualBlock(int width, int hiddenWidth, Device device = null) : base(nameof(GeluResidualBlock))
    {
        Device targetDevice = device ?? CPU;
        _layerNorm = LayerNorm(width, device: targetDevice);
        _hiddenProjection = Linear(width, hiddenWidth, device: targetDevice);
        _outputProjection = Linear(hiddenWidth, width, device: targetDevice);
        RegisterComponents();
    }

    public LayerNorm LayerNorm => _layerNorm;

    public Linear HiddenProjection => _hiddenProjection;

    public Linear OutputProjection => _outputProjection;

    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor normalized = _layerNorm.forward(input);
        Tensor hidden = _hiddenProjection.forward(normalized);
        Tensor activated = _activation.forward(hidden);
        Tensor residual = _outputProjection.forward(activated);
        Tensor output = input + residual;
        output.MoveToOuterDisposeScope();
        return output;
    }
}
