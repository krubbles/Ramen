namespace Ramen.AI;

using Ramen.AgentTools;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public sealed class PreferenceValueModel : Module
{
    public static readonly Device EvalDevice = mps_is_available() ? MPS : CPU;

    public const int CardSetEmbeddingWidth = 48;
    public const int CombinedCardSetEmbeddingWidth = 96;
    public const int HandsAndDiscardsEmbeddingWidth = 31;
    public const int StateWidth = 128;

    readonly MeanPooledCardSetEmbedding _handEmbedding = new(embeddingSize: CardSetEmbeddingWidth, device: EvalDevice);
    readonly MeanPooledCardSetEmbedding _remainingDeckEmbedding = new(embeddingSize: CardSetEmbeddingWidth, device: EvalDevice);
    readonly TorchSharp.Modules.Embedding _handsAndDiscardsEmbedding = Embedding(25, HandsAndDiscardsEmbeddingWidth, device: EvalDevice);
    readonly PreferenceResidualBlock _residualBlock0 = new(width: StateWidth, hiddenWidth: StateWidth, device: EvalDevice);
    readonly PreferenceResidualBlock _residualBlock1 = new(width: StateWidth, hiddenWidth: StateWidth, device: EvalDevice);
    readonly GELU _finalActivation = GELU();
    readonly Linear _outputProjection = Linear(StateWidth, 1, device: EvalDevice);

    public PreferenceValueModel() : base(nameof(PreferenceValueModel))
    {
        RegisterComponents();
    }


    public Tensor GetLogits(GameStateTensors gameStateTensors)
    {
        // Embed each state component.
        Tensor embeddedHand = _handEmbedding.forward(gameStateTensors.FullHand);
        Tensor embeddedRemainingDeck = _remainingDeckEmbedding.forward(gameStateTensors.RemainingDeck);
        Tensor embeddedHandsAndDiscards = _handsAndDiscardsEmbedding.forward(gameStateTensors.HandsAndDiscards);

        // Build the width-128 state vector the residual stack expects.
        Tensor stateVector = cat([embeddedHand, embeddedRemainingDeck, embeddedHandsAndDiscards, gameStateTensors.Score], dim: -1);

        // Score the state with two residual MLP blocks.
        Tensor encodedState = _residualBlock0.forward(stateVector);
        encodedState = _residualBlock1.forward(encodedState);

        Tensor logits = _outputProjection.forward(_finalActivation.forward(encodedState));
        return logits.squeeze(-1);
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

sealed class PreferenceResidualBlock : Module<Tensor, Tensor>
{
    readonly LayerNorm _layerNorm;
    readonly Linear _hiddenProjection;
    readonly GELU _activation = GELU();
    readonly Linear _outputProjection;

    public PreferenceResidualBlock(int width, int hiddenWidth, Device device) : base(nameof(PreferenceResidualBlock))
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
