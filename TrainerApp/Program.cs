using Ramen.AI;
using Ramen.Game;
using Ramen.ConsoleApp;

GameEvalModel model = new GameEvalModel();

CancellationTokenSource trainCancel = new();
Task trainer = null;
TrainingParams trainingParams = new();

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
            case "settp":
                SetTP(context);
                break;
            case "cancel":
                CancelTraining(context);
                break;
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine(ex.ToString()); 
    }
}

void Train(ConsoleCommandContext context)
{
    int epochs = context.GetIntArg(0, "epochs");
    trainingParams.epochs = epochs;
    trainer = Task.Run(() => Training.TrainEvaluationModel(model, trainingParams, trainCancel.Token));
}

void SetTP(ConsoleCommandContext context)
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

void CancelTraining(ConsoleCommandContext context)
{
    trainCancel.Cancel();
    trainer?.Wait();
}