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
/// rate to zero and then back up. Asking for a slope instead is an achievable demand rather
/// than an unbounded one.
/// </para>
/// <para>
/// Slopes are measured as a level difference across a window rather than from adjacent
/// rollouts. A single rollout's entropy change is almost entirely noise -- measured against a
/// real run it carries a signal-to-noise ratio of 0.45 in mid-training and 0.12 once converged
/// -- and the ratio of two such estimates is meaningless, which sends the multiplier on a
/// random walk between its bounds however carefully the target is chosen. Differencing levels
/// across <c>W</c> rollouts grows the signal by <c>W</c> while the noise stays put, so it beats
/// smoothing the per-rollout slopes, which would only gain <c>sqrt(W)</c>.
/// </para>
/// <para>
/// The target slope is whatever closes the current level error over a fixed horizon, treating
/// both trajectories as locally linear: descend <c>error / horizon</c> per rollout faster than
/// the reference and the gap is gone in <c>horizon</c> rollouts. A target fixed at some
/// multiple of the reference slope instead — the previous design — keeps demanding maximum
/// speed until the error changes sign, so it always arrives at the trajectory still moving and
/// has to unwind, which is what made the learning rate swing between its bounds.
/// </para>
/// </summary>
public sealed class EntropyTrackingLearningRate
{
    readonly float[] _referenceEntropy;
    readonly double _baseLearningRate;
    readonly double _minMultiplier;
    readonly double _maxMultiplier;
    readonly double _maxAdjustmentPerRollout;
    readonly double _convergenceHorizon;
    readonly double _maxSlopeRatioTarget;
    readonly double _responseExponent;
    readonly int _slopeWindow;
    readonly double _slopeFloor;
    readonly double _maxObservedRatio;
    readonly Queue<float> _entropyHistory = new();

    double _multiplier = 1.0;
    double _selfSlope;          // entropy decrease per rollout over the window, this run
    double _referenceSlope;     // same for the reference
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
    /// <param name="convergenceHorizon">
    /// Rollouts over which the target slope aims to close the current level error. Small
    /// values chase the reference harder and overshoot more; large values track loosely. This
    /// is the knob that decides how aggressive the controller is.
    /// </param>
    /// <param name="maxSlopeRatioTarget">
    /// Safety bound on the target ratio, applied both ways. The horizon normally keeps the
    /// target near 1; this only binds when the level error is large relative to the reference's
    /// own slope, and stops the controller demanding a slope no learning rate can produce.
    /// </param>
    /// <param name="responseExponent">
    /// Damping on the inversion. Converting a slope ratio into a rate change assumes slope is
    /// proportional to rate; in practice the response is weaker and delayed, so a full
    /// correction each rollout overshoots and rings. Below 1 the controller approaches its
    /// operating point geometrically instead.
    /// </param>
    /// <param name="slopeWindow">
    /// Rollouts spanned by the slope estimate. Longer is less noisy but lags further behind a
    /// change in the learning rate, and lag is what makes a controller ring.
    /// </param>
    public EntropyTrackingLearningRate(
        float[] referenceEntropy,
        double baseLearningRate,
        double minMultiplier = 1.0 / 50.0,
        double maxMultiplier = 15.0,
        double maxAdjustmentPerRollout = 2.5,
        double convergenceHorizon = 10.0,
        double maxSlopeRatioTarget = 4.0,
        double responseExponent = 0.35,
        int slopeWindow = 4,
        double slopeFloorFraction = 0.8,
        double maxObservedRatio = 5.0)
    {
        _referenceEntropy = referenceEntropy;
        _baseLearningRate = baseLearningRate;
        _minMultiplier = minMultiplier;
        _maxMultiplier = maxMultiplier;
        _maxAdjustmentPerRollout = maxAdjustmentPerRollout;
        _convergenceHorizon = convergenceHorizon;
        _maxSlopeRatioTarget = maxSlopeRatioTarget;
        _responseExponent = responseExponent;
        _slopeWindow = Math.Max(1, slopeWindow);
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

        _entropyHistory.Enqueue(entropy);
        while (_entropyHistory.Count > _slopeWindow + 1)
            _entropyHistory.Dequeue();

        // Both slopes span the same window, so whatever the window costs in lag it costs both
        // equally and their ratio stays meaningful.
        int window = _entropyHistory.Count - 1;
        if (window < 1 || ReferenceFor(rollout - window) is not float windowStartReference)
            return LearningRate;

        // Slopes are entropy *decrease* per rollout, so a healthy run has a positive slope.
        _selfSlope = (_entropyHistory.Peek() - entropy) / window;
        _referenceSlope = (windowStartReference - reference) / window;
        _samples++;

        double levelError = entropy - reference;

        // Act only on a full window; a partial one is the noisy estimate this is meant to avoid.
        if (window < _slopeWindow)
            return LearningRate;

        double adjustment;
        if (_referenceSlope <= _slopeFloor)
        {
            // The reference has levelled off, so there is no slope to match.
            TargetSlopeRatio = 1.0;
            adjustment = 1.0;
        }
        else
        {
            // Descend fast enough to erase the current gap over the horizon and no faster. The
            // target decays towards the reference's own slope as the gap closes, so the run
            // arrives on the trajectory already matching its pace rather than still correcting.
            double targetSlope = _referenceSlope + levelError / _convergenceHorizon;
            TargetSlopeRatio = Math.Clamp(
                targetSlope / _referenceSlope,
                1.0 / _maxSlopeRatioTarget,
                _maxSlopeRatioTarget);

            // Assume slope responds roughly in proportion to the rate near the current
            // operating point, and invert. The assumption need not hold exactly: this is a
            // fixed point iteration, and the exponent plus the per-rollout clamp keep a bad
            // estimate from overshooting.
            // Clamped before inverting: a noisy rollout can otherwise ask for an adjustment
            // far larger than the evidence supports. The clamp also covers a run that has
            // stopped descending, or is drifting back upwards, without a special case -- both
            // land on the floor, which asks for a large but not unbounded increase.
            double observedRatio = Math.Clamp(
                _selfSlope / _referenceSlope,
                1.0 / _maxObservedRatio,
                _maxObservedRatio);
            adjustment = Math.Pow(TargetSlopeRatio / observedRatio, _responseExponent);
        }

        adjustment = Math.Clamp(adjustment, 1.0 / _maxAdjustmentPerRollout, _maxAdjustmentPerRollout);
        if (adjustment < 1.0 && !policyIsResponding)
            adjustment = 1.0;

        _multiplier = Math.Clamp(_multiplier * adjustment, _minMultiplier, _maxMultiplier);
        return LearningRate;
    }

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
