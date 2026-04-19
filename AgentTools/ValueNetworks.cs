namespace Ramen.AgentTools;

using Ramen.AI;
using Ramen.Game;
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

public sealed class QuantilePaddedSwiGLUValueNetwork : Module, IValueNetwork
{
    public const int ScoreEmbeddingWidth = 16;
    public const int ScoreBucketCount = 8;
    public const int HandsAndDiscardsEmbeddingWidth = 32;
    public const int InputWidth = StandardProcessor.OutputWidth + ScoreEmbeddingWidth + HandsAndDiscardsEmbeddingWidth;
    public const int DefaultResidualWidth = 192;
    public const int SwiGLUHiddenWidth = 284;
    public const int QuantileCount = 50;

    readonly StandardProcessor _handEmbedding = new();
    readonly ThresholdScoreEmbedding _scoreEmbedding;
    readonly TorchSharp.Modules.Embedding _handsAndDiscardsEmbedding = Embedding(25, HandsAndDiscardsEmbeddingWidth, device: ValueNetwork.EvalDevice);
    readonly ModuleList<PaddedSwiGLUResidualBlock> _residualBlocks = new();
    readonly GELU _finalActivation = GELU();
    readonly Linear _outputProjection;
    readonly bool _useHalfPrecisionForward;
    readonly int _residualWidth;

    public QuantilePaddedSwiGLUValueNetwork(
        float scoreThreshold = 300,
        int residualWidth = DefaultResidualWidth,
        int residualLayerCount = 2,
        bool useHalfPrecisionForward = false) : base(nameof(QuantilePaddedSwiGLUValueNetwork))
    {
        _useHalfPrecisionForward = useHalfPrecisionForward;
        _residualWidth = residualWidth;
        _scoreEmbedding = new(
            threshold: scoreThreshold,
            bucketCount: ScoreBucketCount,
            embeddingWidth: ScoreEmbeddingWidth,
            device: ValueNetwork.EvalDevice);
        _outputProjection = Linear(_residualWidth, QuantileCount, device: ValueNetwork.EvalDevice);

        for (int layerIndex = 0; layerIndex < residualLayerCount; ++layerIndex)
            _residualBlocks.append(new PaddedSwiGLUResidualBlock(
                width: _residualWidth,
                hiddenWidth: SwiGLUHiddenWidth,
                device: ValueNetwork.EvalDevice));

        RegisterComponents();
        this.to(ValueNetwork.EvalDevice, ScalarType.Float32);
    }


    public Tensor GetAdvantages(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor quantiles = GetQuantiles(gameStateTensors);
        Tensor expectedValue = quantiles.mean([quantiles.Dimensions - 1]);
        expectedValue.MoveToOuterDisposeScope();
        return expectedValue;
    }


    public Tensor GetQuantiles(GameStateTensors gameStateTensors)
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
        {
            Tensor blockOutput = _useHalfPrecisionForward
                ? ForwardResidualBlockHalfPrecision(_residualBlocks[layerIndex], residualStream)
                : _residualBlocks[layerIndex].forward(residualStream);
            residualStream = residualStream + blockOutput;
        }

        Tensor activatedResidualStream = _finalActivation.forward(residualStream);
        Tensor quantiles = _useHalfPrecisionForward
            ? LinearHalfPrecision(_outputProjection, activatedResidualStream).to_type(ScalarType.Float32)
            : _outputProjection.forward(activatedResidualStream);
        quantiles.MoveToOuterDisposeScope();
        return quantiles;
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }


    Tensor ForwardResidualBlockHalfPrecision(PaddedSwiGLUResidualBlock block, Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor normalized = block.GetLayerNorm().forward(input);
        Tensor gate = functional.silu(LinearHalfPrecision(block.GetGateProjection(), normalized));
        Tensor value = LinearHalfPrecision(block.GetValueProjection(), normalized);
        Tensor projected = LinearHalfPrecision(block.GetDownProjection(), gate * value).to_type(ScalarType.Float32);

        projected.MoveToOuterDisposeScope();
        return projected;
    }


    static Tensor LinearHalfPrecision(Linear linear, Tensor input)
    {
        using var scope = NewDisposeScope();

        Tensor inputHalf = input.to_type(ScalarType.Float16);
        Tensor weightHalf = linear.weight.to_type(ScalarType.Float16);
        Tensor biasHalf = linear.bias is null ? null : linear.bias.to_type(ScalarType.Float16);
        Tensor output = functional.linear(inputHalf, weightHalf, biasHalf);

        output.MoveToOuterDisposeScope();
        return output;
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


    public LayerNorm GetLayerNorm() => _layerNorm;


    public Linear GetGateProjection() => _gateProjection;


    public Linear GetValueProjection() => _valueProjection;


    public Linear GetDownProjection() => _downProjection;


    public override Tensor forward(Tensor input)
    {
        Tensor normalized = _layerNorm.forward(input);
        Tensor gate = functional.silu(_gateProjection.forward(normalized));
        Tensor value = _valueProjection.forward(normalized);
        return _downProjection.forward(gate * value);
    }
}

public sealed class DiscardRewardTransformerNetwork : Module
{
    public const int TokenWidth = 96;
    public const int ScoreBucketCount = 8;
    public const int AttentionHeadCount = 4;
    public const int FeedForwardHiddenWidth = 96;
    public const int HandTokenCount = GameData.HandSize;
    public const int ExtraTokenCount = 3;
    public const int SequenceLength = HandTokenCount + ExtraTokenCount;

    public static readonly byte[][] DiscardOptionCardIndices = BuildDiscardOptionCardIndices();
    public static readonly int DiscardOptionCount = DiscardOptionCardIndices.Length;

    readonly Embedding _rankEmbedding = Embedding(Card.RankCount + 1, TokenWidth, device: ValueNetwork.EvalDevice);
    readonly Embedding _suitEmbedding = Embedding(Card.SuitCount + 1, TokenWidth, device: ValueNetwork.EvalDevice);
    readonly ThresholdScoreEmbedding _scoreEmbedding;
    readonly Embedding _handsAndDiscardsEmbedding = Embedding(25, TokenWidth, device: ValueNetwork.EvalDevice);
    readonly Parameter _constantToken;
    readonly LayerNorm _attentionNorm0 = LayerNorm(TokenWidth, device: ValueNetwork.EvalDevice);
    readonly MultiheadAttention _attention0 = MultiheadAttention(TokenWidth, AttentionHeadCount);
    readonly TransformerSwiGLUResidualBlock _feedForward0 = new(width: TokenWidth, hiddenWidth: FeedForwardHiddenWidth, device: ValueNetwork.EvalDevice);
    readonly LayerNorm _attentionNorm1 = LayerNorm(TokenWidth, device: ValueNetwork.EvalDevice);
    readonly MultiheadAttention _attention1 = MultiheadAttention(TokenWidth, AttentionHeadCount);
    readonly TransformerSwiGLUResidualBlock _feedForward1 = new(width: TokenWidth, hiddenWidth: FeedForwardHiddenWidth, device: ValueNetwork.EvalDevice);
    readonly Tensor _discardMaskMatrix;

    public DiscardRewardTransformerNetwork(float scoreThreshold = 1f) : base(nameof(DiscardRewardTransformerNetwork))
    {
        _scoreEmbedding = new(
            threshold: scoreThreshold,
            bucketCount: ScoreBucketCount,
            embeddingWidth: TokenWidth,
            device: ValueNetwork.EvalDevice);
        _constantToken = Parameter(zeros([1, 1, TokenWidth], device: ValueNetwork.EvalDevice));
        _discardMaskMatrix = tensor(BuildDiscardMaskMatrix(), dtype: ScalarType.Float32, device: ValueNetwork.EvalDevice).DetachFromScope();

        using IDisposable noGrad = no_grad();
        _rankEmbedding.weight[0].fill_(0f);
        _suitEmbedding.weight[0].fill_(0f);

        RegisterComponents();
        this.to(ValueNetwork.EvalDevice);
    }


    public Tensor GetDiscardRewards(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor cardIndices = gameStateTensors.FullHand.to_type(ScalarType.Int64);
        Tensor validCardMask = cardIndices.gt(0).to_type(ScalarType.Int64);
        Tensor rankIndices = (((cardIndices - 1).clamp_min(0).remainder(Card.RankCount) + 1) * validCardMask).to_type(ScalarType.Int64);
        Tensor suitIndices = (((cardIndices - 1).clamp_min(0).div(Card.RankCount) + 1) * validCardMask).to_type(ScalarType.Int64);

        Tensor cardTokens = _rankEmbedding.forward(rankIndices) + _suitEmbedding.forward(suitIndices);
        Tensor scoreToken = _scoreEmbedding.forward(gameStateTensors.Score).squeeze(1).unsqueeze(1);
        Tensor handsAndDiscardsToken = _handsAndDiscardsEmbedding.forward(gameStateTensors.HandsAndDiscards.to_type(ScalarType.Int64)).unsqueeze(1);

        long batchSize = cardTokens.size(0);
        Tensor constantToken = _constantToken.expand(batchSize, 1, TokenWidth);
        Tensor tokens = cat([cardTokens, scoreToken, handsAndDiscardsToken, constantToken], dim: 1);

        tokens = tokens + GetAttentionOutput(_attention0, _attentionNorm0, tokens);
        tokens = tokens + _feedForward0.forward(tokens);
        tokens = tokens + GetAttentionOutput(_attention1, _attentionNorm1, tokens);
        tokens = tokens + _feedForward1.forward(tokens);

        Tensor finalCardTokens = tokens.slice(1, 0, HandTokenCount, 1);
        Tensor finalConstantToken = tokens.slice(1, SequenceLength - 1, SequenceLength, 1).squeeze(1);
        Tensor discardMaskBatch = _discardMaskMatrix.unsqueeze(0).expand(batchSize, DiscardOptionCount, HandTokenCount);
        Tensor discardedTokenSums = matmul(discardMaskBatch, finalCardTokens);
        Tensor diff = finalConstantToken.unsqueeze(1) - discardedTokenSums;
        Tensor rewards = diff.square().sum(-1);

        rewards.MoveToOuterDisposeScope();
        return rewards;
    }


    public void Save(string filePath)
    {
        save(filePath);
    }


    public void Load(string filePath)
    {
        load(filePath);
    }


    Tensor GetAttentionOutput(MultiheadAttention attention, LayerNorm layerNorm, Tensor tokens)
    {
        Tensor normalizedTokens = layerNorm.forward(tokens);
        Tensor sequenceFirstTokens = normalizedTokens.transpose(0, 1);
        Tensor attentionOutput = attention.forward(
            sequenceFirstTokens,
            sequenceFirstTokens,
            sequenceFirstTokens,
            null,
            false,
            null).Item1.transpose(0, 1);
        return attentionOutput;
    }


    static byte[][] BuildDiscardOptionCardIndices()
    {
        List<byte[]> discardOptions = [];
        Span<byte> cardIndices = stackalloc byte[HandTokenCount];

        for (int mask = 1; mask < (1 << HandTokenCount); ++mask)
        {
            int selectedCount = 0;
            for (byte cardIndex = 0; cardIndex < HandTokenCount; ++cardIndex)
            {
                if (((mask >> cardIndex) & 1) == 0)
                    continue;

                if (selectedCount >= 5)
                {
                    selectedCount = -1;
                    break;
                }

                cardIndices[selectedCount] = cardIndex;
                selectedCount++;
            }

            if (selectedCount <= 0)
                continue;

            discardOptions.Add(cardIndices[0..selectedCount].ToArray());
        }

        return [.. discardOptions];
    }


    static float[,] BuildDiscardMaskMatrix()
    {
        float[,] discardMasks = new float[DiscardOptionCount, HandTokenCount];

        for (int optionIndex = 0; optionIndex < DiscardOptionCount; ++optionIndex)
        {
            byte[] discardOption = DiscardOptionCardIndices[optionIndex];
            for (int cardIndex = 0; cardIndex < discardOption.Length; ++cardIndex)
                discardMasks[optionIndex, discardOption[cardIndex]] = 1f;
        }

        return discardMasks;
    }
}

sealed class TransformerSwiGLUResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _gateProjection;
    readonly Linear _valueProjection;
    readonly Linear _downProjection;

    public TransformerSwiGLUResidualBlock(int width, int hiddenWidth, Device device) : base(nameof(TransformerSwiGLUResidualBlock))
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
