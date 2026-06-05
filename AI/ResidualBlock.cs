namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class ResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _hiddenProjection;
    readonly Linear _gateProjection;
    readonly Linear _valueProjection;
    readonly Linear _outputProjection;
    readonly ActivationType _activationFunction;
    readonly int _residualWidth;
    readonly int _hiddenWidth;

    public enum ActivationType
    {
        GELU,
        SwiGLU,
        ReluSquared,
    }

    public int ResidualWidth => _residualWidth;
    public int HiddenWidth => _hiddenWidth;

    public ResidualBlock(int residualWidth, float hiddenRatio, ActivationType activationType, Device device = null) : base(nameof(ResidualBlock))
    {
        if (residualWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(residualWidth), "Residual width must be positive.");
        if (hiddenRatio <= 0f)
            throw new ArgumentOutOfRangeException(nameof(hiddenRatio), "Hidden-to-residual width ratio must be positive.");

        _residualWidth = residualWidth;
        _activationFunction = activationType;
        _hiddenWidth = GetHiddenWidth(residualWidth, hiddenRatio, activationType);

        Device targetDevice = device ?? CPU;
        _layerNorm = LayerNorm(residualWidth, device: targetDevice);

        if (_activationFunction == ActivationType.SwiGLU)
        {
            _gateProjection = Linear(residualWidth, _hiddenWidth, hasBias: false, device: targetDevice);
            _valueProjection = Linear(residualWidth, _hiddenWidth, hasBias: false, device: targetDevice);
            _outputProjection = Linear(_hiddenWidth, residualWidth, hasBias: false, device: targetDevice);
        }
        else
        {
            _hiddenProjection = Linear(residualWidth, _hiddenWidth, device: targetDevice);
            _outputProjection = Linear(_hiddenWidth, residualWidth, device: targetDevice);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor normalized = _layerNorm.forward(input);
        Tensor residual = _activationFunction switch
        {
            ActivationType.GELU => _outputProjection.forward(functional.gelu(_hiddenProjection.forward(normalized))),
            ActivationType.SwiGLU => _outputProjection.forward(functional.silu(_gateProjection.forward(normalized)) * _valueProjection.forward(normalized)),
            ActivationType.ReluSquared => _outputProjection.forward(functional.relu(_hiddenProjection.forward(normalized)).square()),
            _ => throw new InvalidOperationException($"Unknown activation function {_activationFunction}."),
        };
        Tensor output = input + residual;

        output.MoveToOuterDisposeScope();
        return output;
    }

    static int GetHiddenWidth(int residualWidth, float hiddenRatio, ActivationType activationType)
    {
        float rawHiddenWidth = residualWidth * hiddenRatio;
        if (activationType == ActivationType.SwiGLU)
            rawHiddenWidth *= 2f / 3f;

        return Math.Max(1, (int)MathF.Round(rawHiddenWidth));
    }
}
