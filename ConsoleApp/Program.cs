using Ramen.AI;
using Ramen.ConsoleApp;
using Ramen.Game;
using Ramen.Training;
using TorchSharp;

torch.set_default_device(torch.MPS);

TensorManager.Init();

Console.WriteLine("=== WELCOME! ===");
Console.WriteLine("MPS available: " + torch.mps_is_available());

CancellationTokenSource cancel = new();

while (true)
{
    try
    {
        string command = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(command))
            continue;

        ExecuteCommand(command);
    }
    catch (Exception ex)
    {
        Console.WriteLine(ex.ToString());
    }
}

void ExecuteCommand(string command)
{
    ConsoleCommandContext context = new(command);

    switch (context.Name)
    {
        case "run":
            RunGRPOTrainingRun(context);
            break;
        case "analyze":
            AnalyzeTrainingRun(context);
            break;
        default:
            Console.WriteLine($"Unknown command '{context.Name}'. Supported commands: run, analyze.");
            break;
    }
}

void RunGRPOTrainingRun(ConsoleCommandContext context)
{
    // Parse optional run settings.
    int steps = context.GetIntArg("steps", 1);
    int samplingFrequency = context.GetIntArg("sample", 0);
    string runName = context.GetTextArg("name", "grpo");
    bool resume = context.GetBoolArg("resume", false);

    // Configure the GRPO training run settings.
    BasicGRPOTrainingRun trainingRun = new()
    {
        RolloutSize = context.GetIntArg("rollout", 5000),
        Epochs = context.GetIntArg("epochs", 4),
        LearningRate = context.GetFloatArg("lr", 3e-4f),
        Entropy = context.GetFloatArg("ent", 0f),
    };

    // Execute the training steps and snapshot as requested.
    if (resume)
        _ = RunTrainingRunWithResume(trainingRun, runName, steps, samplingFrequency, cancel);
    else
        _ = ITrainingRun.Run(trainingRun, runName, steps, samplingFrequency, cancel);
}

Task RunTrainingRunWithResume(ITrainingRun trainingRun, string runName, int steps, int samplingFrequency, CancellationTokenSource cancellationTokenSource)
{
    return Task.Run(() =>
    {
        // Resolve run directory and model checkpoint.
        string baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Ramen", "Weights", runName);
        Directory.CreateDirectory(baseDir);
        IPolicyModel model = new PolicyModel();

        int startingStep = 0;
        if (TryGetLatestSnapshot(baseDir, out int lastStep, out string lastFilePath))
        {
            model.Load(lastFilePath);
            startingStep = lastStep;
            Console.WriteLine($"Resuming run '{runName}' from step {startingStep}.");
        }
        else
            Console.WriteLine($"No snapshots found for run '{runName}'. Starting from scratch.");

        // Execute training steps, saving snapshots when requested.
        int finalStep = startingStep + steps;
        for (int step = startingStep + 1; step <= finalStep; ++step)
        {
            if (cancellationTokenSource.IsCancellationRequested)
                break;

            trainingRun.Step(model);

            if (samplingFrequency > 0 && step % samplingFrequency == 0)
            {
                string filePath = Path.Combine(baseDir, $"{step}.bin");
                model.Save(filePath);
            }
        }
    }, cancellationTokenSource.Token);
}

bool TryGetLatestSnapshot(string baseDir, out int step, out string filePath)
{
    step = 0;
    filePath = "";
    if (!Directory.Exists(baseDir))
        return false;

    string[] files = Directory.GetFiles(baseDir, "*.bin");
    bool found = false;
    for (int i = 0; i < files.Length; i++)
    {
        string candidatePath = files[i];
        string candidateName = Path.GetFileNameWithoutExtension(candidatePath);
        if (!int.TryParse(candidateName, out int candidateStep))
            continue;

        if (!found || candidateStep > step)
        {
            step = candidateStep;
            filePath = candidatePath;
            found = true;
        }
    }

    return found;
}

void AnalyzeTrainingRun(ConsoleCommandContext context)
{
    string runName = context.GetTextArg(0, "run name");
    int sampleSize = context.GetIntArg("samples", 256);
    string baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Ramen", "Weights", runName);
    if (!Directory.Exists(baseDir))
    {
        Console.WriteLine($"Training run '{runName}' does not exist.");
        return;
    }

    bool IsFirstPlayerMove(GameState game, Move move)
    {
        List<Move> moveHistory = game.MoveState.MoveHistory;
        for (int i = 0; i < moveHistory.Count; i++)
        {
            if (moveHistory[i] is UseHandMove)
                return false;
        }

        return true;
    }

    CSVBuilder output = TrainingRunAnalysis.Analyze(
        runName,
        sampleSize,
        new RewardStatsTrainingRunAnalyzer(),
        new HandTypePresenceTrainingRunAnalyzer(),
        new EndStateHandCountTrainingRunAnalyzer(),
        new PolicyEntropyTrainingRunAnalyzer(
            ("policy_entropy_mean", (_, _) => true),
            ("policy_entropy_first_move_mean", IsFirstPlayerMove)));
    string analysisPath = Path.Combine(baseDir, "analysis.csv");
    File.WriteAllText(analysisPath, output.ToString());
    Console.WriteLine($"Saved analysis to '{analysisPath}'.");
}
