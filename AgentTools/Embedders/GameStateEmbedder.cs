namespace Ramen.AgentTools;

using Ramen.Game;
using static TorchSharp.torch;

/// <summary>
/// An embedded GameState.
/// </summary>
public class GameStateTensors : ITensorGroup
{
    /// <summary>
    /// Vector of cards in <see cref="HandState.Hand"/> encoded with <see cref="Card.ToIndex"/>.
    /// </summary>
    public Tensor FullHand;

    /// <summary>
    /// Vector of cards in <see cref="DeckState.RemainingDeck"/> encoded with <see cref="Card.ToIndex"/>.
    /// </summary>
    public Tensor RemainingDeck;

    /// <summary>
    /// <see cref="ScoringState.CurrentRoundTotalScore"/>
    /// </summary>
    public Tensor Score;

    /// <summary>
    /// <see cref="ScoringState.CurrentRoundThresholdScore"/>
    /// </summary>
    public Tensor ScoreThreshold;

    /// <summary>
    /// Ordered joker slots from <see cref="JokerState.Jokers"/>, encoded as 1-based joker indices with 0 as null.
    /// Shape: (batch, <see cref="MaxOwnedJokerCount"/>).
    /// </summary>
    public Tensor OwnedJokers;

    /// <summary>
    /// Ordered joker slots from <see cref="ShopState.ShopOfferings"/>, encoded as 1-based joker indices with 0 as null.
    /// Shape: (batch, <see cref="MaxStoreJokerCount"/>).
    /// </summary>
    public Tensor StoreJokers;

    /// <summary>
    /// Prices for each store offering slot. Null offerings use price 0.
    /// Shape: (batch, <see cref="MaxStoreJokerCount"/>).
    /// </summary>
    public Tensor StorePrices;

    /// <summary>
    /// <see cref="ShopState.CurrentRerollCost"/>.
    /// </summary>
    public Tensor RerollPrice;

    /// <summary>
    /// <see cref="ShopState.Money"/>
    /// </summary>
    public Tensor Money;

    /// <summary>
    /// <see cref="GameState.Round"/>
    /// </summary>
    public Tensor Round;

    /// <summary>
    /// Encodes whether the state is in-round (`0`) or in-store (`1`).
    /// </summary>
    public Tensor Stage;

    /// <summary>
    /// Optional tensor of per-play hand scores in the standard hand ordering.
    /// Shape: (batch, <see cref="GameStateEmbedder.PlayHandScoreCount"/>).
    /// </summary>
    public Tensor PlayHandScores;

    /// <summary>
    /// <see cref="HandState.RemainingHands"/>
    /// </summary>
    public Tensor RemainingHands;

    /// <summary>
    /// <see cref="HandState.RemainingDiscards"/>
    /// </summary>
    public Tensor RemainingDiscards;
}

/// <summary>
/// An embedded Move.
/// </summary>
public class UseHandTensors : ITensorGroup
{
    /// <summary>
    /// <see cref="ScoringState.CurrentRoundTotalScore"/> after hand is played.
    /// </summary>
    public Tensor Score;

    /// <summary>
    /// Score contributed by the played hand itself.
    /// </summary>
    public Tensor HandScore;
}

public class GameStateEmbedder
{
    public const int MaxOwnedJokerCount = 5;
    public const int MaxStoreJokerCount = 2;
    public static readonly int PlayHandScoreCount = Combinatorics.CalculateCombinationCount(
        setSize: GameData.HandSize,
        maxSubsetSize: 5,
        minSubsetSize: 1);

    readonly long[,] _fullHand;
    readonly long[,] _remainingDeck;
    readonly long[] _remainingHands;
    readonly long[] _remainingDiscards;
    readonly long[,] _ownedJokers;
    readonly long[,] _storeJokers;
    readonly long[,] _storePrices;
    readonly long[] _rerollPrice;
    readonly long[] _money;
    readonly long[] _round;
    readonly long[] _stage;
    readonly float[,] _playHandScores;
    readonly float[] _score;
    readonly float[] _scoreThreshold;

    int _addedGameStateCount;

    public GameStateEmbedder(int gameStateCount)
    {
        _fullHand = new long[gameStateCount, GameData.HandSize];
        _remainingDeck = new long[gameStateCount, 52];
        _remainingHands = new long[gameStateCount];
        _remainingDiscards = new long[gameStateCount];
        _ownedJokers = new long[gameStateCount, MaxOwnedJokerCount];
        _storeJokers = new long[gameStateCount, MaxStoreJokerCount];
        _storePrices = new long[gameStateCount, MaxStoreJokerCount];
        _rerollPrice = new long[gameStateCount];
        _money = new long[gameStateCount];
        _round = new long[gameStateCount];
        _stage = new long[gameStateCount];
        _playHandScores = new float[gameStateCount, PlayHandScoreCount];
        _score = new float[gameStateCount];
        _scoreThreshold = new float[gameStateCount];
    }


    public void AddGameState(GameState gameState)
    {
        if (_addedGameStateCount >= _remainingHands.Length)
            throw new InvalidOperationException("GameStateEmbedder is already full.");

        ReadOnlySpan<Card> hand = gameState.HandState.Hand;
        for (int cardIndex = 0; cardIndex < hand.Length; ++cardIndex)
            _fullHand[_addedGameStateCount, cardIndex] = hand[cardIndex].ToIndex();

        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int cardIndex = 0; cardIndex < deck.Length; ++cardIndex)
            _remainingDeck[_addedGameStateCount, cardIndex] = deck[cardIndex].ToIndex();

        _remainingHands[_addedGameStateCount] = gameState.HandState.RemainingHands;
        _remainingDiscards[_addedGameStateCount] = gameState.HandState.RemainingDiscards;
        _money[_addedGameStateCount] = gameState.ShopState.Money;
        _rerollPrice[_addedGameStateCount] = gameState.ShopState.CurrentRerollCost;
        _round[_addedGameStateCount] = gameState.Round;
        _stage[_addedGameStateCount] = IsStoreStage(gameState.Stage) ? 1 : 0;
        WriteJokerSlots(gameState.JokerState.Jokers, gameState.GameData, _ownedJokers, _addedGameStateCount);
        WriteJokerSlots(gameState.ShopState.ShopOfferings, gameState.GameData, _storeJokers, _addedGameStateCount);
        WriteStorePrices(gameState.ShopState.ShopOfferings, _storePrices, _addedGameStateCount);
        WritePlayHandScores(gameState, _addedGameStateCount);
        _score[_addedGameStateCount] = GetScoreValue(gameState);
        _scoreThreshold[_addedGameStateCount] = GetScoreThresholdValue(gameState);
        _addedGameStateCount++;
    }


    public GameStateTensors ToTensors()
    {
        return ToTensors(CPU, includePlayHandScores: false);
    }


    public GameStateTensors ToTensors(bool includePlayHandScores)
    {
        return ToTensors(CPU, includePlayHandScores);
    }


    public GameStateTensors ToTensors(Device device, bool includePlayHandScores = false)
    {
        long[,] fullHand = new long[_addedGameStateCount, GameData.HandSize];
        long[,] remainingDeck = new long[_addedGameStateCount, 52];
        long[] remainingHands = new long[_addedGameStateCount];
        long[] remainingDiscards = new long[_addedGameStateCount];
        long[,] ownedJokers = new long[_addedGameStateCount, MaxOwnedJokerCount];
        long[,] storeJokers = new long[_addedGameStateCount, MaxStoreJokerCount];
        long[,] storePrices = new long[_addedGameStateCount, MaxStoreJokerCount];
        long[] rerollPrice = new long[_addedGameStateCount];
        long[] money = new long[_addedGameStateCount];
        long[] round = new long[_addedGameStateCount];
        long[] stage = new long[_addedGameStateCount];
        float[,] score2D = new float[_addedGameStateCount, 1];
        float[,] scoreThreshold2D = new float[_addedGameStateCount, 1];
        float[,] playHandScores = includePlayHandScores ? new float[_addedGameStateCount, PlayHandScoreCount] : null;

        for (int stateIndex = 0; stateIndex < _addedGameStateCount; ++stateIndex)
        {
            for (int cardIndex = 0; cardIndex < GameData.HandSize; ++cardIndex)
                fullHand[stateIndex, cardIndex] = _fullHand[stateIndex, cardIndex];

            for (int cardIndex = 0; cardIndex < 52; ++cardIndex)
                remainingDeck[stateIndex, cardIndex] = _remainingDeck[stateIndex, cardIndex];

            remainingHands[stateIndex] = _remainingHands[stateIndex];
            remainingDiscards[stateIndex] = _remainingDiscards[stateIndex];
            money[stateIndex] = _money[stateIndex];
            rerollPrice[stateIndex] = _rerollPrice[stateIndex];
            round[stateIndex] = _round[stateIndex];
            stage[stateIndex] = _stage[stateIndex];
            score2D[stateIndex, 0] = _score[stateIndex];
            scoreThreshold2D[stateIndex, 0] = _scoreThreshold[stateIndex];

            for (int jokerIndex = 0; jokerIndex < MaxOwnedJokerCount; ++jokerIndex)
                ownedJokers[stateIndex, jokerIndex] = _ownedJokers[stateIndex, jokerIndex];

            for (int jokerIndex = 0; jokerIndex < MaxStoreJokerCount; ++jokerIndex)
            {
                storeJokers[stateIndex, jokerIndex] = _storeJokers[stateIndex, jokerIndex];
                storePrices[stateIndex, jokerIndex] = _storePrices[stateIndex, jokerIndex];
            }

            if (!includePlayHandScores)
                continue;

            for (int handIndex = 0; handIndex < PlayHandScoreCount; ++handIndex)
                playHandScores[stateIndex, handIndex] = _playHandScores[stateIndex, handIndex];
        }

        return new()
        {
            FullHand = tensor(fullHand, dtype: ScalarType.Int64, device: device),
            RemainingDeck = tensor(remainingDeck, dtype: ScalarType.Int64, device: device),
            RemainingHands = tensor(remainingHands, dtype: ScalarType.Int64, device: device),
            RemainingDiscards = tensor(remainingDiscards, dtype: ScalarType.Int64, device: device),
            OwnedJokers = tensor(ownedJokers, dtype: ScalarType.Int64, device: device),
            StoreJokers = tensor(storeJokers, dtype: ScalarType.Int64, device: device),
            StorePrices = tensor(storePrices, dtype: ScalarType.Int64, device: device),
            RerollPrice = tensor(rerollPrice, dtype: ScalarType.Int64, device: device),
            Money = tensor(money, dtype: ScalarType.Int64, device: device),
            Round = tensor(round, dtype: ScalarType.Int64, device: device),
            Stage = tensor(stage, dtype: ScalarType.Int64, device: device),
            PlayHandScores = includePlayHandScores ? tensor(playHandScores, dtype: ScalarType.Float32, device: device) : null,
            Score = tensor(score2D, dtype: ScalarType.Float32, device: device),
            ScoreThreshold = tensor(scoreThreshold2D, dtype: ScalarType.Float32, device: device),
        };
    }


    static float GetScoreValue(GameState gameState)
    {
        return (float)gameState.ScoringState.CurrentRoundTotalScore;
    }

    static float GetScoreThresholdValue(GameState gameState)
    {
        return (float)gameState.ScoringState.CurrentRoundThresholdScore;
    }

    static bool IsStoreStage(StageOfGame stage)
    {
        return stage == StageOfGame.EnterShop || stage == StageOfGame.InShop;
    }

    static void WriteJokerSlots(IReadOnlyList<JokerInstance> jokers, GameData gameData, long[,] output, int stateIndex)
    {
        int slotCount = output.GetLength(1);
        for (int jokerListIndex = 0; jokerListIndex < jokers.Count && jokerListIndex < slotCount; ++jokerListIndex)
        {
            JokerInstance joker = jokers[jokerListIndex];
            if (joker is null)
                continue;

            int jokerTypeIndex = GetJokerTypeIndex(gameData, joker.JokerData);
            output[stateIndex, jokerListIndex] = jokerTypeIndex + 1;
        }
    }

    static void WriteStorePrices(IReadOnlyList<JokerInstance> jokers, long[,] output, int stateIndex)
    {
        int slotCount = output.GetLength(1);
        for (int jokerListIndex = 0; jokerListIndex < jokers.Count && jokerListIndex < slotCount; ++jokerListIndex)
        {
            JokerInstance joker = jokers[jokerListIndex];
            output[stateIndex, jokerListIndex] = joker?.JokerData.BasePrice ?? 0;
        }
    }

    static int GetJokerTypeIndex(GameData gameData, Joker joker)
    {
        for (int jokerIndex = 0; jokerIndex < gameData.Jokers.Length; ++jokerIndex)
        {
            if (ReferenceEquals(gameData.Jokers[jokerIndex], joker))
                return jokerIndex;
        }

        throw new InvalidOperationException($"Joker {joker.Name} was not found in the current game data.");
    }

    void WritePlayHandScores(GameState gameState, int stateIndex)
    {
        if (gameState.Stage != StageOfGame.InRoundPlayerChoice)
            return;

        int[][] playHandOptions = Combinatorics.GetCombinations(
            setSize: GameData.HandSize,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        float roundScoreBefore = (float)gameState.ScoringState.CurrentRoundTotalScore;
        int handCardCount = gameState.HandState.HandCardCount;

        for (int handIndex = 0; handIndex < playHandOptions.Length; ++handIndex)
        {
            int[] cardIndices = playHandOptions[handIndex];
            if (cardIndices[^1] >= handCardCount)
            {
                _playHandScores[stateIndex, handIndex] = 0f;
                continue;
            }

            UseHandMove useHandMove = new(false, cardIndices);
            useHandMove.Apply(gameState);
            float roundScoreAfter = (float)gameState.ScoringState.CurrentRoundTotalScore;
            _playHandScores[stateIndex, handIndex] = roundScoreAfter - roundScoreBefore;
            useHandMove.Revert(gameState);
        }
    }
}
