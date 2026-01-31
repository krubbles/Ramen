namespace Ramen.Training;

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Ramen.AI;

public interface ITrainingRun
{
    public void Step(PolicyModel model);

    public static Task Run(ITrainingRun trainingRun, string runName, int steps, int samplingFrequency, CancellationTokenSource cancellationTokenSource)
    {
        return Task.Run(() =>
        {
            PolicyModel model = new();
            string baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Ramen", "Weights", runName);
            Directory.CreateDirectory(baseDir);

            for (int step = 1; step <= steps; ++step)
            {
                if (cancellationTokenSource.IsCancellationRequested)
                    break;

                trainingRun.Step(model);

                if (samplingFrequency > 0 && step % samplingFrequency == 0)
                {
                    string filePath = Path.Combine(baseDir, $"{step}.bin");
                    model.save(filePath);
                }
            }
        }, cancellationTokenSource.Token);
    }
}
