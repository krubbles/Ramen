namespace Ramen.AI;

using TorchSharp.Modules;
using static TorchSharp.torch.nn;

public sealed class MoveVectorizer : Module
{
    public const int CardOneHotWidth = Card.RankCount * Card.SuitCount + 1;
    public const int RemainingDiscardBucketCount = 5;
    public const int RemainingHandDiscardEmbeddingCount = RemainingDiscardBucketCount * RemainingDiscardBucketCount;
    public const float InitializationMean = 0f;
    public const float InitializationStdDev = 0.03f;
    public const float ScoreClampEpsilon = 1e-6f;

    public static readonly int PlayableHandCount = Combinatorics.CalculateCombinationCount(
        setSize: GameData.HandSize,
        minSubsetSize: 1,
        maxSubsetSize: GameData.MaxPlayedHandSize);

    readonly Linear _cardSetProjection;
    readonly Embedding _scoreBucketEmbedding;
    readonly Embedding _remainingHandDiscardEmbedding;
    readonly Parameter _winningMoveEmbedding;
    readonly Tensor _combinationMatrix;
    readonly int _moveEmbeddingWidth;
    readonly int _scoreBucketCount;
    readonly bool _addWinningMoveEmbedding;
    readonly Device _device;

    public int MoveEmbeddingWidth => _moveEmbeddingWidth;
    public int ScoreBucketCount => _scoreBucketCount;

    public MoveVectorizer(
        int moveEmbeddingWidth,
        int scoreBucketCount,
        bool addWinningMoveEmbedding = true,
        Device device = null) : base(nameof(MoveVectorizer))
    {
        if (moveEmbeddingWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(moveEmbeddingWidth), "Move embedding width must be positive.");
        if (scoreBucketCount < 2)
            throw new ArgumentOutOfRangeException(nameof(scoreBucketCount), "Score bucket count must be at least 2.");

        _moveEmbeddingWidth = moveEmbeddingWidth;
        _scoreBucketCount = scoreBucketCount;
        _addWinningMoveEmbedding = addWinningMoveEmbedding;
        _device = device ?? CPU;

        _cardSetProjection = Linear(CardOneHotWidth, moveEmbeddingWidth, device: _device);
        _scoreBucketEmbedding = Embedding(scoreBucketCount, moveEmbeddingWidth, device: _device);
        _remainingHandDiscardEmbedding = Embedding(RemainingHandDiscardEmbeddingCount, moveEmbeddingWidth, device: _device);
        _winningMoveEmbedding = Parameter(randn([moveEmbeddingWidth], device: _device) * InitializationStdDev + InitializationMean);
        _combinationMatrix = CombinationMatrices.GetCombinationMatrices(
                setSize: GameData.HandSize,
                minSubsetSize: 1,
                maxSubsetSize: GameData.MaxPlayedHandSize)
            .to(_device);
        TensorManager.PersistForever(_combinationMatrix);

        using var noGrad = no_grad();
        _cardSetProjection.weight.normal_(InitializationMean, InitializationStdDev);
        _cardSetProjection.bias.fill_(0f);
        _scoreBucketEmbedding.weight.normal_(InitializationMean, InitializationStdDev);
        _remainingHandDiscardEmbedding.weight.normal_(InitializationMean, InitializationStdDev);
        _winningMoveEmbedding.normal_(InitializationMean, InitializationStdDev);

        RegisterComponents();
    }


    public Tensor forward(GameStateTensors gameStateTensors)
    {
        using var scope = NewDisposeScope();

        Tensor playableCardSetEmbedding = EmbedPlayableCardSets(gameStateTensors.FullHand.to(_device));
        Tensor postPlayScore = GetPostPlayScores(gameStateTensors).to(_device);
        Tensor playMoveVectors = VectorizeMoves(
            playableCardSetEmbedding: playableCardSetEmbedding,
            scoreEmbedding: EmbedScore(postPlayScore, gameStateTensors.ScoreThreshold.to(_device)),
            remainingHandDiscardEmbedding: EmbedRemainingHandsAndDiscards(
                remainingHands: (gameStateTensors.RemainingHands.to(_device).to_type(ScalarType.Int64) - 1).clamp_min(0),
                remainingDiscards: gameStateTensors.RemainingDiscards.to(_device).to_type(ScalarType.Int64),
                moveCount: PlayableHandCount),
            winningMoveMask: postPlayScore.greater_equal(gameStateTensors.ScoreThreshold.to(_device)));
        Tensor discardMoveVectors = VectorizeMoves(
            playableCardSetEmbedding: playableCardSetEmbedding,
            scoreEmbedding: EmbedScore(gameStateTensors.Score.to(_device), gameStateTensors.ScoreThreshold.to(_device)),
            remainingHandDiscardEmbedding: EmbedRemainingHandsAndDiscards(
                remainingHands: gameStateTensors.RemainingHands.to(_device).to_type(ScalarType.Int64),
                remainingDiscards: (gameStateTensors.RemainingDiscards.to(_device).to_type(ScalarType.Int64) - 1).clamp_min(0),
                moveCount: PlayableHandCount),
            winningMoveMask: null);
        Tensor stackedMoveVectors = stack([playMoveVectors, discardMoveVectors], dim: 2);
        Tensor flattenedMoveVectors = stackedMoveVectors.view([stackedMoveVectors.size(0), PlayableHandCount * 2, _moveEmbeddingWidth]);

        flattenedMoveVectors.MoveToOuterDisposeScope();
        return flattenedMoveVectors;
    }


    Tensor EmbedPlayableCardSets(Tensor fullHand)
    {
        using var scope = NewDisposeScope();

        Tensor cardOneHot = functional.one_hot(fullHand.to_type(ScalarType.Int64), CardOneHotWidth).to_type(ScalarType.Float32);
        Tensor combinationMatrix = _combinationMatrix.to(fullHand.device);
        Tensor cardSetOneHot = combinationMatrix.unsqueeze(0).matmul(cardOneHot);
        Tensor cardSetEmbedding = _cardSetProjection.forward(cardSetOneHot);

        cardSetEmbedding.MoveToOuterDisposeScope();
        return cardSetEmbedding;
    }


    Tensor VectorizeMoves(
        Tensor playableCardSetEmbedding,
        Tensor scoreEmbedding,
        Tensor remainingHandDiscardEmbedding,
        Tensor winningMoveMask)
    {
        using var scope = NewDisposeScope();

        Tensor moveVector = playableCardSetEmbedding + scoreEmbedding + remainingHandDiscardEmbedding;
        if (_addWinningMoveEmbedding && winningMoveMask is not null)
            moveVector = moveVector + winningMoveMask.unsqueeze(-1).to_type(ScalarType.Float32) * _winningMoveEmbedding;

        moveVector.MoveToOuterDisposeScope();
        return moveVector;
    }


    Tensor EmbedScore(Tensor score, Tensor threshold)
    {
        using var scope = NewDisposeScope();

        Tensor normalizedScore = (score.to(_device).to_type(ScalarType.Float32) / threshold.to(_device).to_type(ScalarType.Float32))
            .clamp(ScoreClampEpsilon, 1f - ScoreClampEpsilon);
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


    Tensor EmbedRemainingHandsAndDiscards(Tensor remainingHands, Tensor remainingDiscards, int moveCount)
    {
        using var scope = NewDisposeScope();

        Tensor embeddingIndices = remainingHands * RemainingDiscardBucketCount + remainingDiscards;
        Tensor embedding = _remainingHandDiscardEmbedding.forward(embeddingIndices)
            .unsqueeze(1)
            .expand(remainingHands.size(0), moveCount, _moveEmbeddingWidth);

        embedding.MoveToOuterDisposeScope();
        return embedding;
    }


    static Tensor GetPostPlayScores(GameStateTensors gameStateTensors)
    {
        return gameStateTensors.Score.to_type(ScalarType.Float32).to(gameStateTensors.PlayHandScores.device) + gameStateTensors.PlayHandScores;
    }
}
