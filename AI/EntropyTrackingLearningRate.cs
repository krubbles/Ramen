namespace Ramen.AI;

using TorchSharp.Modules;

/// <summary>
/// Drives the learning rate so a run's entropy follows a reference run's entropy.
/// <para>
/// Architectures differ in how far a given learning rate moves the policy, so a shared rate
/// is not a fair basis for comparing them: one run sits past its best rate while another is
/// short of it, and a higher rate usually helps early learning while hurting the converged
/// result. Matching the entropy trajectory removes that freedom, leaving reward at equal
/// rollouts comparable and the settled multiplier as a readout of the architecture's rate
/// requirement.
/// </para>
/// <para>
/// The controller works on the entropy <em>slope</em>, not its level. Controlling the level
/// directly cannot be stable here: entropy is near-monotonic, so once a run overshoots below
/// the reference the only correction available is to stop learning and wait, which drives the
/// rate to zero and then back up. Targeting a slope ratio bounded in [0.5, 2] instead means
/// even a large level error only ever asks the run to descend twice as fast as the reference,
/// which is an achievable rate rather than an unbounded demand.
/// </para>
/// </summary>
public sealed class EntropyTrackingLearningRate
{
    readonly float[] _referenceEntropy;
    readonly double _baseLearningRate;
    readonly double _minMultiplier;
    readonly double _maxMultiplier;
    readonly double _maxAdjustmentPerRollout;
    readonly double _levelErrorScale;
    readonly double _maxSlopeRatioTarget;
    readonly double _slopeSmoothing;
    readonly double _slopeFloor;
    readonly double _maxObservedRatio;

    double _multiplier = 1.0;
    double _selfSlope;          // smoothed entropy decrease per rollout, this run
    double _referenceSlope;     // same for the reference
    float _previousEntropy = float.NaN;
    int _samples;

    public double Multiplier => _multiplier;
    public double LearningRate => _baseLearningRate * _multiplier;
    public int ReferenceLength => _referenceEntropy.Length;

    /// <summary>Most recent smoothed slope ratio, for logging. NaN before it is measurable.</summary>
    public double ObservedSlopeRatio =>
        _referenceSlope > _slopeFloor ? _selfSlope / _referenceSlope : double.NaN;

    /// <summary>Slope ratio the controller is currently aiming for.</summary>
    public double TargetSlopeRatio { get; private set; } = 1.0;

    public bool IsSaturated =>
        _multiplier <= _minMultiplier * 1.001 || _multiplier >= _maxMultiplier * 0.999;

    /// <param name="minMultiplier">
    /// Deliberately looser than the upper bound. Cutting the rate is recoverable -- a stalled
    /// run falls behind, which the controller then corrects by pushing the rate back up --
    /// whereas raising it damages MoE routing. Matching a reference that has nearly levelled
    /// off legitimately needs a very small rate, so the floor must not fight that.
    /// </param>
    /// <param name="levelErrorScale">
    /// Entropy error, in nats, at which the target slope ratio reaches its bound. Smaller
    /// values chase the reference harder.
    /// </param>
    /// <param name="maxSlopeRatioTarget">
    /// Bound on the target ratio, applied both ways: at most this much faster than the
    /// reference when behind, and its reciprocal when ahead. This is what keeps the demand on
    /// the learning rate finite no matter how large the level error grows.
    /// </param>
    /// <param name="slopeSmoothing">
    /// Weight on the newest rollout in the slope estimate. A single rollout's entropy change
    /// is mostly noise, so the slopes are exponentially smoothed before being divided.
    /// </param>
    public EntropyTrackingLearningRate(
        float[] referenceEntropy,
        double baseLearningRate,
        double minMultiplier = 1.0 / 50.0,
        double maxMultiplier = 15.0,
        double maxAdjustmentPerRollout = 2.5,
        double levelErrorScale = 0.5,
        double maxSlopeRatioTarget = 2.0,
        double slopeSmoothing = 0.25,
        double slopeFloorFraction = 0.1,
        double maxObservedRatio = 5.0)
    {
        _referenceEntropy = referenceEntropy;
        _baseLearningRate = baseLearningRate;
        _minMultiplier = minMultiplier;
        _maxMultiplier = maxMultiplier;
        _maxAdjustmentPerRollout = maxAdjustmentPerRollout;
        _levelErrorScale = levelErrorScale;
        _maxSlopeRatioTarget = maxSlopeRatioTarget;
        _slopeSmoothing = slopeSmoothing;
        _maxObservedRatio = maxObservedRatio;

        // The floor is a fraction of the reference's own typical slope rather than an absolute
        // constant. Late in a run the reference creeps along at a small but non-zero rate, and
        // dividing by that produces enormous ratios that drive the multiplier into its floor
        // and strand it there -- precisely over the converged stretch the comparison is for.
        double[] slopes = new double[Math.Max(1, referenceEntropy.Length - 1)];
        for (int index = 1; index < referenceEntropy.Length; ++index)
            slopes[index - 1] = Math.Max(0, referenceEntropy[index - 1] - referenceEntropy[index]);
        Array.Sort(slopes);
        double medianSlope = slopes[slopes.Length / 2];
        _slopeFloor = Math.Max(1e-5, medianSlope * slopeFloorFraction);
    }

    /// <summary>Slope below which the reference counts as flat and tracking is suspended.</summary>
    public double SlopeFloor => _slopeFloor;

    /// <summary>Reference entropy for a rollout, or null past the end of the reference.</summary>
    public float? ReferenceFor(int rollout)
    {
        int index = rollout - 1;
        if (index < 0 || index >= _referenceEntropy.Length)
            return null;
        return _referenceEntropy[index];
    }

    /// <summary>
    /// Folds this rollout's entropy into the multiplier and returns the rate for the next
    /// rollout. Past the end of the reference the multiplier is held.
    /// </summary>
    /// <param name="policyIsResponding">
    /// False when the policy barely moved. Cutting the rate cannot slow a run that has already
    /// stopped, so decreases are suppressed while this is false.
    /// </param>
    public double Update(int rollout, float entropy, bool policyIsResponding = true)
    {
        if (ReferenceFor(rollout) is not float reference)
            return LearningRate;

        float previousEntropy = _previousEntropy;
        _previousEntropy = entropy;
        float? previousReference = ReferenceFor(rollout - 1);
        if (float.IsNaN(previousEntropy) || previousReference is not float earlierReference)
            return LearningRate;

        // Slopes are entropy *decrease* per rollout, so a healthy run has a positive slope.
        // One rollout's change is mostly noise, hence the smoothing before they are divided.
        _selfSlope = Smooth(_selfSlope, previousEntropy - entropy);
        _referenceSlope = Smooth(_referenceSlope, earlierReference - reference);
        _samples++;

        // Behind the reference, aim to descend faster; ahead of it, slower. Bounded either
        // way, so the demand on the learning rate stays finite however large the gap is.
        double levelError = entropy - reference;
        double exponent = Math.Clamp(levelError / _levelErrorScale, -1.0, 1.0);
        TargetSlopeRatio = Math.Pow(_maxSlopeRatioTarget, exponent);

        // Two rollouts of history is not enough to divide two smoothed slopes.
        if (_samples < 2)
            return LearningRate;

        double adjustment;
        if (_referenceSlope <= _slopeFloor)
        {
            // The reference has levelled off, so there is no slope to match.
            adjustment = 1.0;
        }
        else if (_selfSlope <= _slopeFloor)
        {
            // This run has stopped descending, so the slope ratio is not invertible. Being
            // behind means it needs a push; being ahead means waiting is the correct action
            // and cutting the rate further would achieve nothing.
            adjustment = levelError > 0 ? _maxAdjustmentPerRollout : 1.0;
        }
        else
        {
            // Assume slope responds roughly in proportion to the rate near the current
            // operating point, and invert. The assumption need not hold exactly: this is a
            // fixed point iteration, and the per-rollout clamp keeps a bad estimate from
            // overshooting.
            // Clamped before inverting: a noisy rollout can otherwise ask for an adjustment
            // far larger than the evidence supports.
            double observedRatio = Math.Clamp(
                _selfSlope / _referenceSlope,
                1.0 / _maxObservedRatio,
                _maxObservedRatio);
            adjustment = TargetSlopeRatio / observedRatio;
        }

        adjustment = Math.Clamp(adjustment, 1.0 / _maxAdjustmentPerRollout, _maxAdjustmentPerRollout);
        if (adjustment < 1.0 && !policyIsResponding)
            adjustment = 1.0;

        _multiplier = Math.Clamp(_multiplier * adjustment, _minMultiplier, _maxMultiplier);
        return LearningRate;
    }

    double Smooth(double current, double sample) =>
        _samples == 0 ? sample : current + _slopeSmoothing * (sample - current);

    /// <summary>Applies the current rate to every parameter group.</summary>
    public void Apply(AdamW optimizer)
    {
        foreach (AdamW.ParamGroup group in optimizer.ParamGroups)
            group.LearningRate = LearningRate;
    }

    /// <summary>Reads the entropy column of a previous run's data.csv.</summary>
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
