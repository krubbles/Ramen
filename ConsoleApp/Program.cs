using Ramen.AI;
using Ramen.Game;
using Ramen.ConsoleApp;

GameEvalModel model = new GameEvalModel();

CancellationTokenSource cancel = new();
cancel.TryReset();
Queue<Task> work = new();
TrainingParams trainingParams = new(5);


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
    EnqueueWork(() =>
    {
        float score = Testing.GetAverageScore(model, samples);
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
    string dbName = context.GetTextArg(0, "database name");
    int numGames = context.GetIntArg(1, "number of games");
    
    EnqueueWork(() =>
    {
        GameDatabase database = new(dbName);
        
        Console.WriteLine($"Generating {numGames} games in database '{dbName}'...");
        
        for (int i = 0; i < numGames; i++)
        {
            GameState gameState = new(GameData.Default);
            RamenAgent agent = new(gameState, model);
            
            while (gameState.Stage != StageOfGame.None)
            {
                if (!agent.MakeMoveStochastic(1.0f))
                    break;
            }
            database.AddGame(gameState);
            
            if ((i + 1) % 10 == 0)
                Console.WriteLine($"Generated {i + 1}/{numGames} games");
        }
        
        Console.WriteLine($"Successfully generated {numGames} games in database '{dbName}'");
    });
}

void Cancel()
{
    cancel.Cancel();
    if (work.TryPeek(out var task))
        task.Wait();
    cancel = new();
}