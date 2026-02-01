namespace Ramen.Training;

using System;
using System.Collections.Generic;
using System.IO;
using Ramen.AI;
using Ramen.Game;

public interface ITrainingRunAnalyzer
{
    public void Analyze(PolicyModel model, IEnumerable<GameState> games, CSVBuilder output);
}

public static class TrainingRunAnalysis
{
    public static CSVBuilder Analyze(string runName, params ITrainingRunAnalyzer[] analyzers)
    {
        CSVBuilder output = new();
        string baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Ramen", "Weights", runName);
        if (!Directory.Exists(baseDir))
            return output;

        string[] files = Directory.GetFiles(baseDir, "*.bin");
        List<(int step, string filePath)> stepFiles = [];
        for (int i = 0; i < files.Length; i++)
        {
            string filePath = files[i];
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            if (int.TryParse(fileName, out int step))
                stepFiles.Add((step, filePath));
        }

        stepFiles.Sort((left, right) => left.step.CompareTo(right.step));

        for (int i = 0; i < stepFiles.Count; i++)
        {
            (int step, string filePath) = stepFiles[i];
            PolicyModel model = new();
            model.load(filePath);
            IReadOnlyList<GameState> games = PlayGameBatch(model);

            output.NextRow().SetCell("step", step);
            for (int analyzerIndex = 0; analyzerIndex < analyzers.Length; analyzerIndex++)
            {
                ITrainingRunAnalyzer analyzer = analyzers[analyzerIndex];
                analyzer.Analyze(model, games, output);
            }
        }

        return output;
    }

    static IReadOnlyList<GameState> PlayGameBatch(PolicyModel model)
    {
        int batchSize = TrainingConfig.BatchSize;
        float temp = TrainingConfig.GoodPlayTemp;
        List<GameState> games = new();
        for (int i = 0; i < batchSize; i++)
        {
            GameState gameState = new(GameData.Default);
            RamenAgent agent = new(gameState, model);
            while (!agent.GameIsDone())
            {
                gameState.AdvanceToNextPlayerChoice();
                agent.MakeMove(temp);
            }
            games.Add(gameState);
        }
        return games;
    }
}
