using Ramen.AI;
using Ramen.Game;
using Ramen.ConsoleApp;
using System.Runtime.InteropServices;
using System.Security.Principal;

PolicyModel model = new PolicyModel();

CancellationTokenSource cancel = new();
cancel.TryReset();
Queue<Task> work = new();
TrainingParams trainingParams = new(5);

Console.WriteLine("=== PROGRAM STARTED ===");
System.Diagnostics.Debug.WriteLine("=== DEBUGGER OUTPUT ===");

ExternalConsole.Initialize();

while (true)
{
    try
    {
        string command = Console.ReadLine();
        ConsoleCommandContext context = new(command);
        switch (context.Name.ToLower())
        {
            case "train":
                Train(context);
                break;
            case "set":
                Set(context);
                break;
            case "cancel":
                Cancel();
                break;
            case "play":
                Play(context);
                break;
            case "test":
                Test(context);
                break;
            case "generate":
                GenerateGames(context);
                break;
            case "todata":
                ToData(context);
                break;
            case "clear":
                Clear();
                break;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine(ex.ToString()); 
    }
}

void Test(ConsoleCommandContext context)
{
    int samples = context.GetIntArg(0, "samples");
    float temp = context.GetFloatArg("temp", 0.0001f);
    EnqueueWork(() =>
    {
        float score = Testing.GetAverageScore(model, samples, temp);
        Console.WriteLine("Average Score: " + score);
    });
}

void EnqueueWork(Action action)
{
    Task task = new(action);
    work.Enqueue(task);
    if (work.Count == 1)
        task.Start();
    task.ContinueWith((task) =>
    {
        work.Dequeue();
        if (work.Count > 0)
            work.Peek().Start();
    });
}

void Play(ConsoleCommandContext context)
{
    int samples = context.GetIntArg(0, "samples");
    EnqueueWork(() => TrainingData.GenerateEvaluationTrainingData(model, samples, 1f));
}

void Train(ConsoleCommandContext context)
{
    if (TrainingData.EvaluationTrainingData.Count == 0)
    {
        Console.WriteLine("Cannot train, no evaluation training data");
        return;
    }

    int epochs = context.GetIntArg(0, "epochs");
    trainingParams.epochs = epochs;
    EnqueueWork(() => Training.TrainEvaluationModel(model, trainingParams, cancel.Token));
        
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
    int branches = context.GetIntArg(2, "branches");
    int samples = context.GetIntArg(3, "samples");
    bool log = context.GetBoolArg("log", false);
    float temp = context.GetFloatArg("temp", 0.1f);

    EnqueueWork(() =>
    {
        GameDatabase database = new(dbName);

        Console.WriteLine($"Generating {games} games in database '{dbName}'...");

        for (int i = 0; i < games; i++)
        {
            GameState gameState = TrainingData.PlayGameMonteCarlo(model, branches, samples, log, temp);
            database.AddGame(gameState);

            Console.Write($"\rGenerated {i + 1}/{games} games");
        }

        Console.WriteLine();
        Console.WriteLine($"Successfully generated {games} games in database '{dbName}'");
    });
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

    EnqueueWork(() =>
    {
        GameDatabase database = new(dbName, load: true);
        Console.WriteLine($"Loading {dbName}...");
        int countBefore = TrainingData.EvaluationTrainingData.Count;
        TrainingData.GenerateLastMoveTrainingData(model, database);
        int countAfter = TrainingData.EvaluationTrainingData.Count;
        Console.WriteLine($"Added {countAfter - countBefore} training samples from '{dbName}'");
    });
}

void Clear()
{
    TrainingData.EvaluationTrainingData.Clear();
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