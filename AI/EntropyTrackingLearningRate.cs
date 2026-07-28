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
    readonly double _maxAdjustmentPerRollout;

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
    /// <param name="maxAdjustmentPerRollout">
    /// Hard cap on how far the multiplier can move in one rollout, as a ratio. This is the
    /// binding constraint on the controller's speed: entropy responds to a rate change over
    /// several rollouts, so a controller that can move faster than that overshoots and then
    /// has to unwind, which is far worse than tracking loosely.
    /// </param>
    public EntropyTrackingLearningRate(
        float[] referenceEntropy,
        double baseLearningRate,
        double gain = 0.3,
        double minMultiplier = 1.0 / 15.0,
        double maxMultiplier = 15.0,
        double maxErrorPerStep = 0.5,
        double maxAdjustmentPerRollout = 1.2)
    {
        _referenceEntropy = referenceEntropy;
        _baseLearningRate = baseLearningRate;
        _gain = gain;
        _minMultiplier = minMultiplier;
        _maxMultiplier = maxMultiplier;
        _maxErrorPerStep = maxErrorPerStep;
        _maxAdjustmentPerRollout = maxAdjustmentPerRollout;
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
    /// <param name="policyIsResponding">
    /// False when the policy barely moved this rollout. Entropy is then not being held up by
    /// the learning rate, so cutting further would only freeze the run while the error
    /// persists and the multiplier keeps winding down. Increases are still allowed.
    /// </param>
    public double Update(int rollout, float entropy, bool policyIsResponding = true)
    {
        if (ReferenceFor(rollout) is not float reference)
            return LearningRate;

        // Entropy above the reference means the policy has not committed as far, so it needs
        // a larger rate; below means it is sharpening too fast and needs a smaller one.
        double error = entropy - reference;
        error = Math.Clamp(error, -_maxErrorPerStep, _maxErrorPerStep);

        double adjustment = Math.Clamp(
            Math.Exp(_gain * error),
            1.0 / _maxAdjustmentPerRollout,
            _maxAdjustmentPerRollout);

        if (adjustment < 1.0 && !policyIsResponding)
            adjustment = 1.0;

        _multiplier = Math.Clamp(_multiplier * adjustment, _minMultiplier, _maxMultiplier);
        return LearningRate;
    }

    /// <summary>Whether the multiplier has run into a bound, which means tracking has failed
    /// and the run should not be read as a comparison.</summary>
    public bool IsSaturated =>
        _multiplier <= _minMultiplier * 1.001 || _multiplier >= _maxMultiplier * 0.999;

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
