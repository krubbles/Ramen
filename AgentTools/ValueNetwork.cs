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
