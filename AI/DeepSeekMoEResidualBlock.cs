namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public readonly record struct MoERoutingStats(
    float[] ExpertTokenFractions,
    float LoadBalancingLoss,
    int ActiveExpertCount,
    long RoutedTokenCount);

public sealed class DeepSeekMoEResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _inputLayerNorm;
    readonly LayerNorm _routerLayerNorm;
    readonly FeedForwardExpert _sharedExpert;
    readonly BatchedFeedForwardExperts _routedExperts;
    readonly Linear _router;
    readonly Tensor _routingBias;
    readonly List<MoERoutingStats> _routingStats = [];
    readonly int _residualWidth;
    readonly int _routerInputWidth;
    readonly int _routedExpertCount;
    readonly int _chosenExpertCount;
    readonly float _routingBiasUpdateSpeed;

    public int ResidualWidth => _residualWidth;
    public int SharedExpertHiddenWidth => _residualWidth;
    public int RoutedExpertHiddenWidth => _residualWidth / 2;
    public int RoutedExpertCount => _routedExpertCount;
    public int ChosenExpertCount => _chosenExpertCount;

    public List<MoERoutingStats> DrainRoutingStats()
    {
        List<MoERoutingStats> stats = [.. _routingStats];
        _routingStats.Clear();
        return stats;
    }

    public DeepSeekMoEResidualBlock(
        int residualWidth,
        int routerInputWidth,
        int routedExpertCount,
        int chosenExpertCount,
        float routingBiasUpdateSpeed,
        ResidualBlock.ActivationType activationType,
        float routedExpertHiddenRatio = 0.5f,
        Device device = null) : base(nameof(DeepSeekMoEResidualBlock))
    {
        _residualWidth = residualWidth;
        _routerInputWidth = routerInputWidth;
        _routedExpertCount = routedExpertCount;
        _chosenExpertCount = chosenExpertCount;
        _routingBiasUpdateSpeed = routingBiasUpdateSpeed;

        Device targetDevice = device ?? CPU;
        _inputLayerNorm = LayerNorm(residualWidth, device: targetDevice);
        if (routerInputWidth != residualWidth)
            _routerLayerNorm = LayerNorm(routerInputWidth, device: targetDevice);
        _sharedExpert = new(
            inputWidth: residualWidth,
            hiddenWidth: residualWidth,
            outputWidth: residualWidth,
            activationType: activationType,
            device: targetDevice);
        _routedExperts = new(
            expertCount: routedExpertCount,
            inputWidth: residualWidth,
            hiddenWidth: Math.Max(1, (int)(residualWidth * routedExpertHiddenRatio)),
            outputWidth: residualWidth,
            activationType: activationType,
            device: targetDevice);
        _router = Linear(routerInputWidth, routedExpertCount, hasBias: false, device: targetDevice);
        _routingBias = zeros([routedExpertCount], device: targetDevice);
        register_buffer(nameof(_routingBias), _routingBias);

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        return Forward(
            input,
            input,
            updateLoadBalance: false,
            routerInputIsBlockInput: true);
    }

    public Tensor Forward(
        Tensor input,
        Tensor routerInput,
        bool updateLoadBalance,
        bool routerInputIsBlockInput)
    {
        using var dScope = NewDisposeScope();

        Tensor normalizedInput = _inputLayerNorm.forward(input);
        Tensor flatInput = normalizedInput.reshape([-1, _residualWidth]);
        Tensor flatRouterInput = (
            routerInputIsBlockInput
                ? normalizedInput
                : _routerLayerNorm.forward(routerInput))
            .reshape([-1, _routerInputWidth]);
        Tensor affinities = sigmoid(_router.forward(flatRouterInput));
        (Tensor _, Tensor routedExpertIndices) = (affinities + _routingBias).topk(
            _chosenExpertCount,
            dim: 1,
            largest: true,
            sorted: false);
        Tensor routingWeights = affinities.gather(dim: 1, index: routedExpertIndices);
        routingWeights = routingWeights / routingWeights.sum(dim: 1, keepdim: true);
        Tensor flatRoutedExpertIndices = routedExpertIndices.reshape([-1]);
        Tensor expertLoads = zeros(
            [_routedExpertCount],
            dtype: ScalarType.Float32,
            device: input.device);
        expertLoads.index_add_(
            dim: 0,
            index: flatRoutedExpertIndices,
            source: ones(
                [flatRoutedExpertIndices.size(0)],
                dtype: ScalarType.Float32,
                device: input.device),
            alpha: 1);

        if (updateLoadBalance)
        {
            using var noGrad = no_grad();

            long totalAssignments = flatRoutedExpertIndices.size(0);
            Tensor weightSums = zeros_like(expertLoads);
            weightSums.index_add_(
                dim: 0,
                index: flatRoutedExpertIndices,
                source: routingWeights.reshape([-1]).to_type(ScalarType.Float32),
                alpha: 1);

            // Both vectors come back in a single device->host copy; the per-expert
            // reduction is a few dozen floats, so finishing it here is cheaper than
            // the extra kernel launches and sync points it would cost on device.
            float[] loadsThenWeights = [.. stack([expertLoads, weightSums]).reshape([-1]).cpu().data<float>()];
            float[] expertTokenFractions = new float[_routedExpertCount];
            float loadBalancingLoss = 0f;
            int activeExpertCount = 0;
            for (int expertIndex = 0; expertIndex < _routedExpertCount; ++expertIndex)
            {
                float expertLoad = loadsThenWeights[expertIndex];
                float tokenFraction = expertLoad / totalAssignments;
                expertTokenFractions[expertIndex] = tokenFraction;
                loadBalancingLoss +=
                    tokenFraction * (loadsThenWeights[_routedExpertCount + expertIndex] / totalAssignments);
                if (expertLoad > 0f)
                    ++activeExpertCount;
            }

            _routingStats.Add(new(
                ExpertTokenFractions: expertTokenFractions,
                LoadBalancingLoss: loadBalancingLoss * _routedExpertCount,
                ActiveExpertCount: activeExpertCount,
                RoutedTokenCount: totalAssignments));

            float meanExpertLoad = routedExpertIndices.numel() / (float)_routedExpertCount;
            Tensor biasUpdates = where(
                expertLoads.lt(meanExpertLoad),
                full_like(expertLoads, _routingBiasUpdateSpeed),
                where(
                    expertLoads.gt(meanExpertLoad),
                    full_like(expertLoads, -_routingBiasUpdateSpeed),
                    zeros_like(expertLoads)));
            _routingBias.add_(biasUpdates);
        }

        long routedTokenGroupCount = flatRouterInput.size(0);
        long tokensPerRouterInput = flatInput.size(0) / routedTokenGroupCount;
        Tensor groupedInput = flatInput.reshape([
            routedTokenGroupCount,
            tokensPerRouterInput,
            _residualWidth]);
        Tensor groupedRoutedOutput = zeros_like(groupedInput);
        long expectedExpertLoad = (
            routedTokenGroupCount * _chosenExpertCount + _routedExpertCount - 1) /
            _routedExpertCount;
        long maxExpertLoad = (long)expertLoads.max().item<float>();
        long expertBatchSize = 8;
        while (expertBatchSize < expectedExpertLoad)
            expertBatchSize *= 2;
        expertBatchSize *= 2;
        while (expertBatchSize < maxExpertLoad)
            expertBatchSize *= 2;

        (Tensor sortedExpertIndices, Tensor sortedRouteIndices) =
            flatRoutedExpertIndices.sort();
        Tensor expertStarts = (expertLoads.cumsum(dim: 0) - expertLoads)
            .to_type(ScalarType.Int64);
        Tensor withinExpertIndices = arange(
                flatRoutedExpertIndices.size(0),
                dtype: ScalarType.Int64,
                device: input.device) -
            expertStarts.index_select(dim: 0, index: sortedExpertIndices);
        Tensor packedRouteIndices =
            sortedExpertIndices * expertBatchSize + withinExpertIndices;
        Tensor groupIndices = arange(
                routedTokenGroupCount,
                dtype: ScalarType.Int64,
                device: input.device)
            .unsqueeze(1)
            .expand([-1, _chosenExpertCount])
            .reshape([-1])
            .index_select(dim: 0, index: sortedRouteIndices);
        Tensor flatRoutingWeights = routingWeights.reshape([-1]);
        Tensor packedInputs = zeros([
            _routedExpertCount * expertBatchSize,
            tokensPerRouterInput,
            _residualWidth],
            device: input.device,
            dtype: input.dtype);
        packedInputs.scatter_(
            dim: 0,
            index: packedRouteIndices
                .reshape([-1, 1, 1])
                .expand([-1, tokensPerRouterInput, _residualWidth]),
            src: groupedInput.index_select(dim: 0, index: groupIndices));
        Tensor packedRoutingWeights = zeros(
            [_routedExpertCount * expertBatchSize],
            device: input.device,
            dtype: input.dtype);
        packedRoutingWeights.scatter_(
            dim: 0,
            index: packedRouteIndices,
            src: flatRoutingWeights.index_select(dim: 0, index: sortedRouteIndices));
        Tensor packedOutputs = _routedExperts.forward(packedInputs.reshape([
                _routedExpertCount,
                expertBatchSize,
                tokensPerRouterInput,
                _residualWidth])) *
            packedRoutingWeights.reshape([
                _routedExpertCount,
                expertBatchSize,
                1,
                1]);
        Tensor routedOutputs = packedOutputs
            .reshape([
                _routedExpertCount * expertBatchSize,
                tokensPerRouterInput,
                _residualWidth])
            .index_select(dim: 0, index: packedRouteIndices);
        groupedRoutedOutput.index_add_(
            dim: 0,
            index: groupIndices,
            source: routedOutputs,
            alpha: 1);

        Tensor output = input + (
            _sharedExpert.forward(flatInput) + groupedRoutedOutput.reshape([-1, _residualWidth]))
            .reshape(input.shape);
        output.ToOuterScope();
        return output;
    }

    sealed class FeedForwardExpert : Module<Tensor, Tensor>
    {
        readonly Linear _hiddenProjection;
        readonly Linear _gateProjection;
        readonly Linear _valueProjection;
        readonly Linear _outputProjection;
        readonly ResidualBlock.ActivationType _activationType;

        public FeedForwardExpert(
            int inputWidth,
            int hiddenWidth,
            int outputWidth,
            ResidualBlock.ActivationType activationType,
            Device device) : base(nameof(FeedForwardExpert))
        {
            _activationType = activationType;
            if (activationType == ResidualBlock.ActivationType.SwiGLU)
            {
                _gateProjection = Linear(inputWidth, hiddenWidth, hasBias: false, device: device);
                _valueProjection = Linear(inputWidth, hiddenWidth, hasBias: false, device: device);
                _outputProjection = Linear(hiddenWidth, outputWidth, hasBias: false, device: device);
            }
            else
            {
                _hiddenProjection = Linear(inputWidth, hiddenWidth, device: device);
                _outputProjection = Linear(hiddenWidth, outputWidth, device: device);
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var dScope = NewDisposeScope();

            Tensor output = _activationType switch
            {
                ResidualBlock.ActivationType.GELU =>
                    _outputProjection.forward(functional.gelu(_hiddenProjection.forward(input))),
                ResidualBlock.ActivationType.SwiGLU =>
                    _outputProjection.forward(functional.silu(_gateProjection.forward(input)) * _valueProjection.forward(input)),
                ResidualBlock.ActivationType.ReluSquared =>
                    _outputProjection.forward(functional.relu(_hiddenProjection.forward(input)).square()),
                _ => throw new InvalidOperationException($"Unknown activation function {_activationType}."),
            };
            output.ToOuterScope();
            return output;
        }
    }

    sealed class BatchedFeedForwardExperts : Module<Tensor, Tensor>
    {
        readonly Parameter _hiddenWeights;
        readonly Parameter _hiddenBiases;
        readonly Parameter _gateWeights;
        readonly Parameter _valueWeights;
        readonly Parameter _outputWeights;
        readonly Parameter _outputBiases;
        readonly ResidualBlock.ActivationType _activationType;
        readonly int _expertCount;
        readonly int _inputWidth;
        readonly int _hiddenWidth;
        readonly int _outputWidth;

        public BatchedFeedForwardExperts(
            int expertCount,
            int inputWidth,
            int hiddenWidth,
            int outputWidth,
            ResidualBlock.ActivationType activationType,
            Device device) : base(nameof(BatchedFeedForwardExperts))
        {
            _expertCount = expertCount;
            _inputWidth = inputWidth;
            _hiddenWidth = hiddenWidth;
            _outputWidth = outputWidth;
            _activationType = activationType;

            if (activationType == ResidualBlock.ActivationType.SwiGLU)
            {
                _gateWeights = Parameter(empty([
                    expertCount,
                    hiddenWidth,
                    inputWidth],
                    device: device));
                _valueWeights = Parameter(empty([
                    expertCount,
                    hiddenWidth,
                    inputWidth],
                    device: device));
            }
            else
            {
                _hiddenWeights = Parameter(empty([
                    expertCount,
                    hiddenWidth,
                    inputWidth],
                    device: device));
                _hiddenBiases = Parameter(empty([
                    expertCount,
                    hiddenWidth],
                    device: device));
            }
            _outputWeights = Parameter(empty([
                expertCount,
                outputWidth,
                hiddenWidth],
                device: device));
            if (activationType != ResidualBlock.ActivationType.SwiGLU)
            {
                _outputBiases = Parameter(empty([
                    expertCount,
                    outputWidth],
                    device: device));
            }

            using var noGrad = no_grad();
            float inputBound = 1f / MathF.Sqrt(inputWidth);
            float hiddenBound = 1f / MathF.Sqrt(hiddenWidth);
            if (activationType == ResidualBlock.ActivationType.SwiGLU)
            {
                _gateWeights.uniform_(-inputBound, inputBound);
                _valueWeights.uniform_(-inputBound, inputBound);
            }
            else
            {
                _hiddenWeights.uniform_(-inputBound, inputBound);
                _hiddenBiases.uniform_(-inputBound, inputBound);
                _outputBiases.uniform_(-hiddenBound, hiddenBound);
            }
            _outputWeights.uniform_(-hiddenBound, hiddenBound);

            RegisterComponents();
        }

        public override Tensor forward(Tensor input)
        {
            using var dScope = NewDisposeScope();

            Tensor flatInput = input.reshape([
                _expertCount,
                -1,
                _inputWidth]);
            Tensor hidden;
            if (_activationType == ResidualBlock.ActivationType.SwiGLU)
            {
                Tensor gate = bmm(
                    flatInput,
                    _gateWeights.transpose(1, 2));
                Tensor value = bmm(
                    flatInput,
                    _valueWeights.transpose(1, 2));
                hidden = functional.silu(gate) * value;
            }
            else
            {
                Tensor projected = bmm(
                    flatInput,
                    _hiddenWeights.transpose(1, 2)) +
                    _hiddenBiases.unsqueeze(1);
                hidden = _activationType switch
                {
                    ResidualBlock.ActivationType.GELU => functional.gelu(projected),
                    ResidualBlock.ActivationType.ReluSquared => functional.relu(projected).square(),
                    _ => throw new InvalidOperationException($"Unknown activation function {_activationType}."),
                };
            }

            Tensor flatOutput = bmm(
                hidden,
                _outputWeights.transpose(1, 2));
            if (_outputBiases is not null)
                flatOutput = flatOutput + _outputBiases.unsqueeze(1);

            Tensor output = flatOutput.reshape([
                _expertCount,
                input.size(1),
                input.size(2),
                _outputWidth]);
            output.ToOuterScope();
            return output;
        }
    }
}
