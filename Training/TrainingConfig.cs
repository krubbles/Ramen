namespace Ramen.Training;

public static class TrainingConfig
{
    public const int BatchSize = 64;
    public const float LearningRate = 0.0002f;

    public const float GoodPlayTemp = 1.2f;
    public const float ExploratoryPlayTemp = 2f;
}
