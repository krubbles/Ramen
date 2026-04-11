namespace Ramen.AgentTools;

using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public sealed class ValueNetwork : Module, IValueNetwork
{
    public static readonly Device EvalDevice = mps_is_available() ? MPS : CPU;

    public const int CardSetEmbeddingWidth = 48;
    public const int CombinedCardSetEmbeddingWidth = 96;
    public const int HandsAndDiscardsEmbeddingWidth = 31;
    public const int StateWidth = 128;

    readonly MeanPooledCardSetEmbedding _handEmbedding = new(embeddingSize: CardSetEmbeddingWidth, device: EvalDevice);
    readonly MeanPooledCardSetEmbedding _remainingDeckEmbedding = new(embeddingSize: CardSetEmbeddingWidth, device: EvalDevice);
    readonly TorchSharp.Modules.Embedding _handsAndDiscardsEmbedding = Embedding(25, HandsAndDiscardsEmbeddingWidth, device: EvalDevice);
    readonly ValueResidualBlock _residualBlock0 = new(width: StateWidth, hiddenWidth: StateWidth, device: EvalDevice);
    readonly ValueResidualBlock _residualBlock1 = new(width: StateWidth, hiddenWidth: StateWidth, device: EvalDevice);
    readonly GELU _finalActivation = GELU();
    readonly Linear _outputProjection = Linear(StateWidth, 1, device: EvalDevice);

    public ValueNetwork() : base(nameof(ValueNetwork))
    {
        RegisterComponents();
    }


    public Tensor GetAdvantages(GameStateTensors gameStateTensors)
    {
        // Embed each state component.
        Tensor embeddedHand = _handEmbedding.forward(gameStateTensors.FullHand);
        Tensor embeddedRemainingDeck = _remainingDeckEmbedding.forward(gameStateTensors.RemainingDeck);
        Tensor embeddedHandsAndDiscards = _handsAndDiscardsEmbedding.forward(gameStateTensors.HandsAndDiscards);

        // Build the width-128 state vector expected by the residual stack.
        Tensor stateVector = cat([embeddedHand, embeddedRemainingDeck, embeddedHandsAndDiscards, gameStateTensors.Score], dim: -1);

        // Score each state with the same two-block residual MLP used by the preference model.
        Tensor encodedState = _residualBlock0.forward(stateVector);
        encodedState = _residualBlock1.forward(encodedState);

        Tensor advantages = _outputProjection.forward(_finalActivation.forward(encodedState));
        return advantages.squeeze(-1);
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }
}

public sealed class SimpleValueNetwork : Module, IValueNetwork
{
    public const int ScoreEmbeddingWidth = 16;
    public const int ScoreBucketCount = 8;
    public const int HandsAndDiscardsEmbeddingWidth = 32;
    public const int StateWidth = StandardProcessor.OutputWidth + ScoreEmbeddingWidth + HandsAndDiscardsEmbeddingWidth;

    readonly StandardProcessor _handEmbedding = new();
    readonly ThresholdScoreEmbedding _scoreEmbedding;
    readonly TorchSharp.Modules.Embedding _handsAndDiscardsEmbedding = Embedding(25, HandsAndDiscardsEmbeddingWidth, device: ValueNetwork.EvalDevice);
    readonly Sequential _valueHead;

    public SimpleValueNetwork(float scoreThreshold = 300, int hiddenWidth0 = 128, int hiddenWidth1 = 64, int hiddenWidth2 = 0) : base(nameof(SimpleValueNetwork))
    {
        _scoreEmbedding = new(
            threshold: scoreThreshold,
            bucketCount: 8,
            embeddingWidth: ScoreEmbeddingWidth,
            device: ValueNetwork.EvalDevice);
        if (hiddenWidth2 > 0)
        {
            _valueHead =
                Sequential(
                    Linear(StateWidth, hiddenWidth0, device: ValueNetwork.EvalDevice),
                    GELU(),
                    Linear(hiddenWidth0, hiddenWidth1, device: ValueNetwork.EvalDevice),
                    GELU(),
                    Linear(hiddenWidth1, hiddenWidth2, device: ValueNetwork.EvalDevice),
                    GELU(),
                    Linear(hiddenWidth2, 1, device: ValueNetwork.EvalDevice)
                );
        }
        else
        {
            _valueHead =
                Sequential(
                    Linear(StateWidth, hiddenWidth0, device: ValueNetwork.EvalDevice),
                    GELU(),
                    Linear(hiddenWidth0, hiddenWidth1, device: ValueNetwork.EvalDevice),
                    GELU(),
                    Linear(hiddenWidth1, 1, device: ValueNetwork.EvalDevice)
                );
        }

        RegisterComponents();
    }


    public Tensor GetAdvantages(GameStateTensors gameStateTensors)
    {
        Tensor embeddedHand = _handEmbedding.forward(gameStateTensors.FullHand);
        Tensor embeddedScore = _scoreEmbedding.forward(gameStateTensors.Score).squeeze(1);
        Tensor embeddedHandsAndDiscards = _handsAndDiscardsEmbedding.forward(gameStateTensors.HandsAndDiscards);
        Tensor stateVector = cat([embeddedHand, embeddedScore, embeddedHandsAndDiscards], dim: -1);
        Tensor advantages = _valueHead.forward(stateVector);
        return advantages.squeeze(-1);
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }
}

public sealed class PaddedSwiGLUValueNetwork : Module, IValueNetwork
{
    public const int ScoreEmbeddingWidth = 16;
    public const int ScoreBucketCount = 8;
    public const int HandsAndDiscardsEmbeddingWidth = 32;
    public const int InputWidth = StandardProcessor.OutputWidth + ScoreEmbeddingWidth + HandsAndDiscardsEmbeddingWidth;
    public const int DefaultResidualWidth = 384;
    public const int SwiGLUHiddenWidth = 284;

    readonly StandardProcessor _handEmbedding = new();
    readonly ThresholdScoreEmbedding _scoreEmbedding;
    readonly TorchSharp.Modules.Embedding _handsAndDiscardsEmbedding = Embedding(25, HandsAndDiscardsEmbeddingWidth, device: ValueNetwork.EvalDevice);
    readonly ModuleList<PaddedSwiGLUResidualBlock> _residualBlocks = new();
    readonly GELU _finalActivation = GELU();
    readonly Linear _outputProjection;
    readonly int _residualWidth;

    public PaddedSwiGLUValueNetwork(
        float scoreThreshold = 300,
        int residualWidth = DefaultResidualWidth,
        int residualLayerCount = 3) : base(nameof(PaddedSwiGLUValueNetwork))
    {
        _residualWidth = residualWidth;
        _scoreEmbedding = new(
            threshold: scoreThreshold,
            bucketCount: ScoreBucketCount,
            embeddingWidth: ScoreEmbeddingWidth,
            device: ValueNetwork.EvalDevice);
        _outputProjection = Linear(_residualWidth, 1, device: ValueNetwork.EvalDevice);

        for (int layerIndex = 0; layerIndex < residualLayerCount; ++layerIndex)
            _residualBlocks.append(new PaddedSwiGLUResidualBlock(
                width: _residualWidth,
                hiddenWidth: SwiGLUHiddenWidth,
                device: ValueNetwork.EvalDevice));

        RegisterComponents();
    }


    public Tensor GetAdvantages(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor embeddedHand = _handEmbedding.forward(gameStateTensors.FullHand);
        Tensor embeddedScore = _scoreEmbedding.forward(gameStateTensors.Score).squeeze(1);
        Tensor embeddedHandsAndDiscards = _handsAndDiscardsEmbedding.forward(gameStateTensors.HandsAndDiscards);
        Tensor stateVector = cat([embeddedHand, embeddedScore, embeddedHandsAndDiscards], dim: -1);

        Tensor zeroPadding = zeros(
            [stateVector.size(0), _residualWidth - InputWidth],
            dtype: stateVector.dtype,
            device: stateVector.device);
        Tensor residualStream = cat([stateVector, zeroPadding], dim: -1);

        for (int layerIndex = 0; layerIndex < _residualBlocks.Count; ++layerIndex)
            residualStream = residualStream + _residualBlocks[layerIndex].forward(residualStream);

        Tensor advantages = _outputProjection.forward(_finalActivation.forward(residualStream));
        Tensor squeezedAdvantages = advantages.squeeze(-1);
        squeezedAdvantages.MoveToOuterDisposeScope();
        return squeezedAdvantages;
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }
}

sealed class ValueResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _hiddenProjection;
    readonly GELU _activation = GELU();
    readonly Linear _outputProjection;

    public ValueResidualBlock(int width, int hiddenWidth, Device device) : base(nameof(ValueResidualBlock))
    {
        _layerNorm = LayerNorm(width, device: device);
        _hiddenProjection = Linear(width, hiddenWidth, device: device);
        _outputProjection = Linear(hiddenWidth, width, device: device);

        RegisterComponents();
    }


    public override Tensor forward(Tensor input)
    {
        Tensor normalized = _layerNorm.forward(input);
        Tensor hidden = _hiddenProjection.forward(normalized);
        Tensor activated = _activation.forward(hidden);
        Tensor projected = _outputProjection.forward(activated);
        return input + projected;
    }
}

sealed class PaddedSwiGLUResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _gateProjection;
    readonly Linear _valueProjection;
    readonly Linear _downProjection;

    public PaddedSwiGLUResidualBlock(int width, int hiddenWidth, Device device) : base(nameof(PaddedSwiGLUResidualBlock))
    {
        _layerNorm = LayerNorm(width, device: device);
        _gateProjection = Linear(width, hiddenWidth, hasBias: false, device: device);
        _valueProjection = Linear(width, hiddenWidth, hasBias: false, device: device);
        _downProjection = Linear(hiddenWidth, width, hasBias: false, device: device);

        RegisterComponents();
    }


    public override Tensor forward(Tensor input)
    {
        Tensor normalized = _layerNorm.forward(input);
        Tensor gate = functional.silu(_gateProjection.forward(normalized));
        Tensor value = _valueProjection.forward(normalized);
        return _downProjection.forward(gate * value);
    }
}
