using Ramen.AI;
using Ramen.Game;
using Ramen.ConsoleApp;
using System.Linq;

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
            case "todata":
                ToData(context);
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
    string dbName = context.GetTextArg(0, "db");
    int games = context.GetIntArg(1, "games");
    int branches = context.GetIntArg(2, "branches");
    int samples = context.GetIntArg(3, "samples");
    bool log = context.GetBoolArg("log", false);
    
    EnqueueWork(() =>
    {
        GameDatabase database = new(dbName);
        
        Console.WriteLine($"Generating {games} games in database '{dbName}'...");
        
        for (int i = 0; i < games; i++)
        {
            GameState gameState = new(GameData.Default);
            RamenAgent agent = new(gameState, model);
            gameState.AdvanceToNextPlayerChoice();

            List<int> playerChoiceMoveSteps = new();
            
            while (!agent.GameIsDone())
            {
                gameState.AdvanceToNextPlayerChoice();
                playerChoiceMoveSteps.Add(gameState.MoveState.MoveStep);
                agent.MakeMove(1.0f);
            }
            
            int rollbackIndex = agent.Random.Next(playerChoiceMoveSteps.Count);
            int rollbackStep = playerChoiceMoveSteps[rollbackIndex];
            gameState.MoveState.RevertToStep(rollbackStep);
            
            var candidateMoves = agent.SampleMoves(1.0f, branches);
            
            float bestAvgReward = float.MinValue;
            Move bestMove = null;
            List<(Move move, float avgReward)> moveEvalutions = new();
            
            foreach (var (candidateMove, _) in candidateMoves)
            {
                float totalReward = 0f;
                
                candidateMove.Apply(gameState);
                int afterCandidateStep = gameState.MoveState.MoveStep;
                
                for (int c = 0; c < samples; c++)
                {
                    gameState.Reseed();
                    
                    while (!agent.GameIsDone())
                    {
                        gameState.AdvanceToNextPlayerChoice();
                        agent.MakeMove(1.0f);
                    }
                    
                    totalReward += agent.GetCurrentReward();
                    
                    gameState.MoveState.RevertToStep(afterCandidateStep);
                }
                
                float avgReward = totalReward / samples;
                moveEvalutions.Add((candidateMove, avgReward));
                
                if (avgReward > bestAvgReward)
                {
                    bestAvgReward = avgReward;
                    bestMove = candidateMove;
                }

                gameState.MoveState.RevertToStep(rollbackStep);
            }

            if (log)
            {
                Console.WriteLine($"GameState: {gameState}");
                var sortedMoves = moveEvalutions.OrderByDescending(x => x.avgReward).ToList();
                foreach (var (move, avgReward) in sortedMoves)
                {
                    Console.WriteLine($"Move: {move} - Avg Reward: {avgReward:F4}");
                }
            }
            
            bestMove.Apply(gameState);
            database.AddGame(gameState);
            
            if ((i + 1) % 50 == 0)
                Console.WriteLine($"Generated {i + 1}/{games} games");
        }
        
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