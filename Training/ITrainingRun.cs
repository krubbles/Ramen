namespace Ramen.Training;

using Ramen.AI;

public interface ITrainingRun
{
    public void Step(PolicyModel model);
}
