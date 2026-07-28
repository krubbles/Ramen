namespace Ramen.AI;

using TorchSharp.Modules;

/// <summary>
/// Drives the learning rate so a run's entropy follows a reference run's entropy.
/// <para>
/// Architectures differ in how much a given learning rate moves the policy, so a shared
/// learning rate is not a fair basis for comparing them: one architecture will be past its
/// best rate while another is short of it. Forcing both runs down the same entropy
/// trajectory removes that degree of freedom, and reward at equal rollouts becomes
/// comparable. Entropy is a good handle because it measures how far the policy has
/// committed, independent of how well it is doing.
/// </para>
/// </summary>
public sealed class EntropyTrackingLearningRate
{
    readonly float[] _referenceEntropy;
    readonly double _baseLearningRate;
    readonly double _gain;
    readonly double _minMultiplier;
    readonly double _maxMultiplier;
    readonly double _maxErrorPerStep;

    double _multiplier = 1.0;

    public double Multiplier => _multiplier;
    public double LearningRate => _baseLearningRate * _multiplier;
    public int ReferenceLength => _referenceEntropy.Length;

    /// <param name="referenceEntropy">Reference entropy by rollout index, from a previous run.</param>
    /// <param name="gain">
    /// How hard one rollout's entropy error pushes the multiplier. The multiplier moves by
    /// exp(gain * error), so at gain 0.3 an error of 0.1 nats moves it about 3%.
    /// </param>
    /// <param name="maxErrorPerStep">
    /// Entropy error is clamped to this before it is applied, so a single wild rollout
    /// cannot jump the learning rate.
    /// </param>
    public EntropyTrackingLearningRate(
        float[] referenceEntropy,
        double baseLearningRate,
        double gain = 0.3,
        double minMultiplier = 0.05,
        double maxMultiplier = 20.0,
        double maxErrorPerStep = 0.5)
    {
        _referenceEntropy = referenceEntropy;
        _baseLearningRate = baseLearningRate;
        _gain = gain;
        _minMultiplier = minMultiplier;
        _maxMultiplier = maxMultiplier;
        _maxErrorPerStep = maxErrorPerStep;
    }

    /// <summary>
    /// Reference entropy for a rollout, or null past the end of the reference run.
    /// </summary>
    public float? ReferenceFor(int rollout)
    {
        int index = rollout - 1;
        if (index < 0 || index >= _referenceEntropy.Length)
            return null;
        return _referenceEntropy[index];
    }

    /// <summary>
    /// Folds this rollout's entropy into the multiplier and returns the rate for the next
    /// rollout. Past the end of the reference the multiplier is held, so the run continues
    /// at whatever rate it had settled on rather than snapping back.
    /// </summary>
    public double Update(int rollout, float entropy)
    {
        if (ReferenceFor(rollout) is not float reference)
            return LearningRate;

        // Entropy above the reference means the policy has not committed as far, so it needs
        // a larger rate; below means it is sharpening too fast and needs a smaller one.
        double error = entropy - reference;
        error = Math.Clamp(error, -_maxErrorPerStep, _maxErrorPerStep);

        _multiplier = Math.Clamp(_multiplier * Math.Exp(_gain * error), _minMultiplier, _maxMultiplier);
        return LearningRate;
    }

    /// <summary>Applies the current rate to every parameter group.</summary>
    public void Apply(AdamW optimizer)
    {
        foreach (AdamW.ParamGroup group in optimizer.ParamGroups)
            group.LearningRate = LearningRate;
    }

    /// <summary>
    /// Reads the entropy column of a previous run's data.csv.
    /// </summary>
    public static float[] LoadReferenceEntropy(string dataPath, string columnName = "average_entropy")
    {
        string[] lines = File.ReadAllLines(dataPath);
        if (lines.Length < 2)
            throw new InvalidOperationException($"Reference run at {dataPath} has no rows.");

        string[] header = lines[0].Split(',');
        int column = Array.IndexOf(header, columnName);
        if (column < 0)
            throw new InvalidOperationException($"Reference run at {dataPath} has no '{columnName}' column.");

        List<float> values = [];
        for (int lineIndex = 1; lineIndex < lines.Length; ++lineIndex)
        {
            if (string.IsNullOrWhiteSpace(lines[lineIndex]))
                continue;
            values.Add(float.Parse(
                lines[lineIndex].Split(',')[column],
                System.Globalization.CultureInfo.InvariantCulture));
        }
        return [.. values];
    }
}
