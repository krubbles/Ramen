namespace Ramen.Training;

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Ramen.AI;
using Ramen.AgentTools;

/// <summary>
/// Defines a training run that can advance a policy model step-by-step.
/// </summary>
public interface ITrainingRun
{
    /// <summary>
    /// Performs a single training step on the provided model.
    /// </summary>
    public void Step(IPolicyNetwork model);

    /// <summary>
    /// Repeatedly calls <see cref="Step"> <paramref name="steps"> times. Saves model weight snapshots into [EnvironmentAppData]/Ramen/Weights/[<paramref name="runName">]/[step].bin every <paramref name="samplingFrequency"> steps.
    /// </summary>
    public static Task Run(ITrainingRun trainingRun, string runName, int steps, int samplingFrequency, CancellationTokenSource cancellationTokenSource, int startingStep = 0)
    {
        return Task.Run(() =>
        {
            // Create the model and ensure the output directory exists.
            using PolicyModel model = new();
            string baseDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "Ramen", "Weights", runName);
            Directory.CreateDirectory(baseDir);

            // Execute training steps, saving snapshots when requested.
            for (int step = 1; step <= steps; ++step)
            {
                if (cancellationTokenSource.IsCancellationRequested)
                    break;

                trainingRun.Step(model);

                if (samplingFrequency > 0 && step % samplingFrequency == 0)
                {
                    string filePath = Path.Combine(baseDir, $"{startingStep + step}.bin");
                    model.Save(filePath);
                }

                TensorManager.DisposeAll();
            }
        }, cancellationTokenSource.Token);
    }
}
