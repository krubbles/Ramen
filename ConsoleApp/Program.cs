using Ramen.AI;
using Ramen.Game;
using Ramen.ConsoleApp;
using System.Runtime.InteropServices;
using System.Collections.Concurrent;

// needed for cursor
// ExternalConsole.Initialize();
System.Diagnostics.Debug.WriteLine("=== WELCOME! ===");


PolicyModel model = new();

CancellationTokenSource cancel = new();
cancel.TryReset();
ConcurrentQueue<Task> work = new();
TrainingParams trainingParams = new(5);
List<string> CommandHistory = new();
const int MaxCommandHistory = 100;

while (true)
{
    try
    {
        string command = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(command))
            continue;
        ExecuteCommand(command, isFromRepeat: false);
    }
    catch (Exception ex)
    {
        Console.WriteLine(ex.ToString());
    }
}

void ExecuteCommand(string command, bool isFromRepeat)
{
    ConsoleCommandContext context = new(command);
    if (!isFromRepeat && context.Name != "repeat")
        AddCommandHistory(command);

    switch (context.Name)
    {
        case "traingrpo":
            EnqueueWork(() => TrainGRPO(context));
            break;
        case "train":
            EnqueueWork(() => TrainSupervised(context));
            break;
        case "play":
            EnqueueWork(() => Play(context));
            break;
        case "set":
            Set(context);
            break;
        case "cancel":
            Cancel();
            break;
        case "test":
            EnqueueWork(() => Test(context));
            break;
        case "generate":
            EnqueueWork(() => GenerateGames(context));
            break;
        case "stats":
            EnqueueWork(() => Stats(context));
            break;
        case "todata":
            EnqueueWork(() => ToData(context));
            break;
        case "augment":
            EnqueueWork(() => Augment(context));
            break;
        case "clear":
            EnqueueWork(() => Clear());
            break;
        case "repeat":
            Repeat(context);
            break;
        default:
            Console.WriteLine($"Unknown command '{context.Name}'.");
            break;
    }
}

void AddCommandHistory(string command)
{
    CommandHistory.Add(command);
    if (CommandHistory.Count > MaxCommandHistory)
        CommandHistory.RemoveAt(0);
}

void Repeat(ConsoleCommandContext context)
{
    int commandCount = context.GetIntArg(0, "command count");
    int repeatCount = context.GetIntArg(1, "repeat count");
    if (commandCount <= 0 || repeatCount <= 0)
    {
        Console.WriteLine("Repeat counts must be greater than 0.");
        return;
    }

    if (CommandHistory.Count < commandCount)
    {
        Console.WriteLine($"Only {CommandHistory.Count} commands in history.");
        return;
    }

    List<string> commands = CommandHistory.GetRange(CommandHistory.Count - commandCount, commandCount);
    for (int i = 0; i < repeatCount; i++)
    {
        foreach (string command in commands)
            ExecuteCommand(command, isFromRepeat: true);
    }
}

void Play(ConsoleCommandContext context)
{
    int samples = context.GetIntArg(0, "samples");
    float temp = context.GetFloatArg("temp", 1f);
    bool log = context.GetBoolArg("log", false);
    TrainingDataStats stats = TrainingData.GenerateGRPO(model, samples);
    Console.WriteLine("Done playing games.");
    Console.WriteLine($"Average reward: {stats.TotalReward / stats.GamesCount:F4}");
    Console.WriteLine($"Average round0 nlprob: {stats.AverageNLProb(0):F4}");
    Console.WriteLine($"Average round1 nlprob: {stats.AverageNLProb(1):F4}");
    Console.WriteLine($"Average round2 nlprob: {stats.AverageNLProb(2):F4}");
}

void Test(ConsoleCommandContext context)
{
    int samples = context.GetIntArg(0, "samples");
    float temp = context.GetFloatArg("temp", 0.0001f);
    (double mean, double ciLower, double ciUpper, double stdError) = Testing.GetScoreStatistics(model, samples, temp);
    Console.WriteLine($"Average Score: {mean:F4}");
    Console.WriteLine($"95% CI: [{ciLower:F4}, {ciUpper:F4}]");
    Console.WriteLine($"Standard Error: {stdError:F4}");
}

void EnqueueWork(Action action)
{
    Task task = new(action);
    lock (work)
    {
        work.Enqueue(task);
        if (work.Count == 1)
            task.Start();
        task.ContinueWith((task) =>
        {
            lock (work)
            {
                bool dequeued = work.TryDequeue(out Task finished);
                if (dequeued && finished.Exception != null)
                {
                    Console.WriteLine(finished.Exception.Message);
                    Console.WriteLine(finished.Exception.InnerException.StackTrace);
                }
                if (work.TryPeek(out Task nextTask))
                {
                    nextTask.Start();
                }
            }
        });
    }
}


void TrainGRPO(ConsoleCommandContext context)
{
    int epochs = context.GetIntArg(0, "epochs");
    bool useCispo = context.GetBoolArg("cispo", false);
    trainingParams.epochs = epochs;
    Training.TrainPolicyModelGRPO(model, trainingParams, cancel.Token, useCispo);

}

void TrainSupervised(ConsoleCommandContext context)
{
    int epochs = context.GetIntArg(0, "epochs");
    trainingParams.epochs = epochs;
    Training.TrainPolicyModelSupervised(model, trainingParams, cancel.Token);

}

void Set(ConsoleCommandContext context)
{
    string trainingParam = context.GetTextArg(0, "param name");
    switch (trainingParam)
    {
        case "kld":
            trainingParams.kldCoeff = context.GetFloatArg(1, "kld coeff");
            break;
        case "ent":
            trainingParams.entropyCoeff = context.GetFloatArg(1, "entropy coeff");
            break;
        case "lr":
            trainingParams.learningRate = context.GetFloatArg(1, "learning rate");
            break;
        case "bs":
            trainingParams.batchSize = context.GetIntArg(1, "batch size");
            break;
        default:
            Console.WriteLine($"Unrecognized training param '{trainingParam}'. Valid params are kld, ent, lr, and bs.");
            break;
    }
}

void GenerateGames(ConsoleCommandContext context)
{
    string dbName = context.GetTextArg(0, "db");
    int games = context.GetIntArg(1, "games");
    int branches = context.GetIntArg("br", 2);
    int samples = context.GetIntArg("samples", 5);
    bool log = context.GetBoolArg("log", false);
    float temp = context.GetFloatArg("temp", 0.1f);
    bool mc = context.GetBoolArg("mc", true);
    GameDatabase database = new(dbName);

    Console.WriteLine($"Generating {games} games in database '{dbName}'...");

    float totalReward = 0f;

    for (int i = 0; i < games; i++)
    {
        if (cancel.IsCancellationRequested)
            return;

        GameState gameState = mc ?
            TrainingData.PlayGameMonteCarlo(model, branches, samples, log, temp, cancel.Token) :
            TrainingData.PlayGame(model, temp, cancel.Token);
        database.AddGame(gameState);

        RamenAgent agent = new(gameState, model);
        float reward = agent.GetCurrentReward();
        totalReward += reward;
        float averageReward = totalReward / (i + 1);

        Console.Write($"\rGenerated {i + 1}/{games} games, average reward: {averageReward:F4}");
    }

    Console.WriteLine();
    Console.WriteLine($"Successfully generated {games} games in database '{dbName}'");
}

void Stats(ConsoleCommandContext context)
{
    string dbName = context.GetTextArg(0, "db");
    Testing.GameDatabaseStatistics stats = Testing.GetGameDatabaseStatistics(dbName);

    Console.WriteLine($"Database '{dbName}' games: {stats.TotalGames}");
    Console.WriteLine($"Played 2 pair: {stats.PlayedTwoPairPercent:P2}");
    Console.WriteLine($"Played straight: {stats.PlayedStraightPercent:P2}");
    Console.WriteLine($"Played flush: {stats.PlayedFlushPercent:P2}");
    Console.WriteLine($"Played full house: {stats.PlayedFullHousePercent:P2}");
    Console.WriteLine($"Discard same suit: {stats.DiscardSameSuitPercent:P2}");
    Console.WriteLine($"Discard rank range <= 4: {stats.DiscardRankRangePercent:P2}");
}

void Cancel()
{
    cancel.Cancel();
    if (work.TryPeek(out var task))
        task.Wait();
    cancel = new();
}

void ToData(ConsoleCommandContext context)
{
    string dbName = context.GetTextArg(0, "db");
    GameDatabase database = new(dbName, load: true);
    Console.WriteLine($"Loading {dbName}...");
    int countBefore = TrainingData.PolicyData.Count;
    TrainingData.GenerateTrainingDataFromGames(model, database);
    int countAfter = TrainingData.PolicyData.Count;
    Console.WriteLine($"Added {countAfter - countBefore} training samples from '{dbName}'");
}

void Augment(ConsoleCommandContext context)
{
    int countBefore = TrainingData.PolicyData.Count;
    DataAugmentation.AugmentEvaluationTrainingDataBySuitRemap();
    int countAfter = TrainingData.PolicyData.Count;
    Console.WriteLine($"Augmented training data. Added {countAfter - countBefore} samples");
}

void Clear()
{
    TrainingData.PolicyData.Clear();
    Console.WriteLine("Cleared all training data");
}

static class ExternalConsole
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool FreeConsole();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AllocConsole();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetStdHandle(int nStdHandle);

    private const int STD_OUTPUT_HANDLE = -11;
    private const int STD_INPUT_HANDLE = -10;

    public static void Initialize()
    {
        FreeConsole();
        if (AllocConsole())
        {
            var stdOutPtr = GetStdHandle(STD_OUTPUT_HANDLE);
            var safeOutHandle = new Microsoft.Win32.SafeHandles.SafeFileHandle(stdOutPtr, true);
            var sw = new StreamWriter(new FileStream(safeOutHandle, FileAccess.Write)) { AutoFlush = true };
            Console.SetOut(sw);

            var stdInPtr = GetStdHandle(STD_INPUT_HANDLE);
            var safeInHandle = new Microsoft.Win32.SafeHandles.SafeFileHandle(stdInPtr, true);
            var sr = new StreamReader(new FileStream(safeInHandle, FileAccess.Read));
            Console.SetIn(sr);

            Console.WriteLine("New Console Window Allocated!");
        }
    }
}
