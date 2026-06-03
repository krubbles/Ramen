namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class GameStateVectorizer : Module
{
    public const int RemainingDiscardBucketCount = 5;
    public const int RemainingHandDiscardEmbeddingCount = RemainingDiscardBucketCount * RemainingDiscardBucketCount;
    public const float InitializationMean = 0f;
    public const float InitializationStdDev = 0.03f;
    public const float ScoreClampEpsilon = 1e-6f;

    readonly Embedding _remainingHandDiscardEmbedding;
    readonly Embedding _scoreBucketEmbedding;
    readonly Embedding _handCardEmbedding;
    readonly Embedding _deckCardEmbedding;
    readonly int _scoreBucketCount;
    readonly int _embeddingWidth;
    readonly Device _device;

    public int ScoreBucketCount => _scoreBucketCount;
    public int EmbeddingWidth => _embeddingWidth;

    public GameStateVectorizer(int embeddingWidth, int scoreBucketCount, Device device = null) : base(nameof(GameStateVectorizer))
    {
        if (embeddingWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(embeddingWidth), "Embedding width must be positive.");
        if (scoreBucketCount < 2)
            throw new ArgumentOutOfRangeException(nameof(scoreBucketCount), "Score bucket count must be at least 2.");

        _embeddingWidth = embeddingWidth;
        _scoreBucketCount = scoreBucketCount;
        _device = device ?? CPU;

        _remainingHandDiscardEmbedding = Embedding(RemainingHandDiscardEmbeddingCount, embeddingWidth, device: _device);
        _scoreBucketEmbedding = Embedding(scoreBucketCount, embeddingWidth, device: _device);
        _handCardEmbedding = Embedding(Card.RankCount * Card.SuitCount + 1, embeddingWidth, device: _device);
        _deckCardEmbedding = Embedding(Card.RankCount * Card.SuitCount + 1, embeddingWidth, device: _device);

        using var noGrad = no_grad();
        InitializeEmbedding(_remainingHandDiscardEmbedding, hasNullCard: false);
        InitializeEmbedding(_scoreBucketEmbedding, hasNullCard: false);
        InitializeEmbedding(_handCardEmbedding, hasNullCard: true);
        InitializeEmbedding(_deckCardEmbedding, hasNullCard: true);

        RegisterComponents();
    }


    public Tensor forward(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor handEmbedding = MeanPoolCardEmbeddings(_handCardEmbedding, gameStateTensors.FullHand.to(_device));
        Tensor deckEmbedding = MeanPoolCardEmbeddings(_deckCardEmbedding, gameStateTensors.RemainingDeck.to(_device));
        Tensor remainingHandDiscardEmbedding = EmbedRemainingHandsAndDiscards(gameStateTensors);
        Tensor scoreEmbedding = EmbedScore(gameStateTensors);
        Tensor vector = handEmbedding + deckEmbedding + remainingHandDiscardEmbedding + scoreEmbedding;

        vector.MoveToOuterDisposeScope();
        return vector;
    }


    void InitializeEmbedding(Embedding embedding, bool hasNullCard)
    {
        embedding.weight.normal_(InitializationMean, InitializationStdDev);
        if (hasNullCard)
            embedding.weight[0].fill_(0f);
    }


    Tensor MeanPoolCardEmbeddings(Embedding embedding, Tensor cardSet)
    {
        using var scope = NewDisposeScope();

        Tensor cardIndices = cardSet.to_type(ScalarType.Int64);
        Tensor embeddedCards = embedding.forward(cardIndices);
        Tensor pooledEmbedding = embeddedCards.mean([embeddedCards.Dimensions - 2]);

        pooledEmbedding.MoveToOuterDisposeScope();
        return pooledEmbedding;
    }


    Tensor EmbedRemainingHandsAndDiscards(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor remainingHands = gameStateTensors.RemainingHands.to(_device).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(_device).to_type(ScalarType.Int64);
        Tensor embeddingIndices = remainingHands * RemainingDiscardBucketCount + remainingDiscards;
        Tensor embedding = _remainingHandDiscardEmbedding.forward(embeddingIndices);

        embedding.MoveToOuterDisposeScope();
        return embedding;
    }


    Tensor EmbedScore(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor score = gameStateTensors.Score.to(_device).to_type(ScalarType.Float32).view([-1]);
        Tensor threshold = gameStateTensors.ScoreThreshold.to(_device).to_type(ScalarType.Float32).view([-1]);
        Tensor normalizedScore = (score / threshold).clamp(ScoreClampEpsilon, 1f - ScoreClampEpsilon);
        Tensor bucketPosition = normalizedScore * (_scoreBucketCount - 1);
        Tensor lowerBucketIndex = bucketPosition.floor().to_type(ScalarType.Int64);
        Tensor upperBucketIndex = bucketPosition.ceil().to_type(ScalarType.Int64);
        Tensor upperWeight = (bucketPosition - lowerBucketIndex.to_type(ScalarType.Float32)).unsqueeze(-1);
        Tensor lowerWeight = 1f - upperWeight;

        Tensor lowerEmbedding = _scoreBucketEmbedding.forward(lowerBucketIndex);
        Tensor upperEmbedding = _scoreBucketEmbedding.forward(upperBucketIndex);
        Tensor embedding = lowerEmbedding * lowerWeight + upperEmbedding * upperWeight;

        embedding.MoveToOuterDisposeScope();
        return embedding;
    }
}
