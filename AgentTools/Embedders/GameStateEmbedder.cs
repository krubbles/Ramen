namespace Ramen.AgentTools;

using Ramen.Game;
using static TorchSharp.torch;

/// <summary>
/// An embedded GameState.
/// </summary>
public class GameStateTensors : ITensorGroup
{
    // ROUND STATE

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

    // STORE STATE

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

    // PERSISTENT STATE

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
}

public class GameStateEmbedder
{
    public const int MaxOwnedJokerCount = 5;
    public const int MaxStoreJokerCount = 2;
    public static readonly int PlayHandScoreCount = Combinatorics.CalculateCombinationCount(
        setSize: GameData.HandSize,
        maxSubsetSize: 5,
        minSubsetSize: 1);

    // in round data
    readonly long[,] _fullHand;
    readonly long[,] _remainingDeck;
    readonly long[] _remainingHands;
    readonly long[] _remainingDiscards;
    readonly float[] _score;
    readonly float[] _scoreThreshold;
    readonly float[,] _playHandScores;

    // in store data
    readonly long[,] _storeJokers;
    readonly long[,] _storePrices;
    readonly long[] _rerollPrice;

    // persistent data
    readonly long[,] _ownedJokers;
    readonly long[] _money;
    readonly long[] _round;
    readonly long[] _isInStore;

    public int AddedGameStateCount { get; private set; }

    public int MaxGameStateCount => _remainingHands.Length;

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
        _isInStore = new long[gameStateCount];
        _playHandScores = new float[gameStateCount, PlayHandScoreCount];
        _score = new float[gameStateCount];
        _scoreThreshold = new float[gameStateCount];
    }


    public void AddGameState(GameState gameState)
    {
        if (AddedGameStateCount >= _remainingHands.Length)
            throw new InvalidOperationException("GameStateEmbedder is already full.");

        // in round data

        ReadOnlySpan<Card> hand = gameState.HandState.Hand;
        for (int cardIndex = 0; cardIndex < hand.Length; ++cardIndex)
            _fullHand[AddedGameStateCount, cardIndex] = hand[cardIndex].ToIndex();

        ReadOnlySpan<Card> deck = gameState.DeckState.RemainingDeck;
        for (int cardIndex = 0; cardIndex < deck.Length; ++cardIndex)
            _remainingDeck[AddedGameStateCount, cardIndex] = deck[cardIndex].ToIndex();

        _remainingHands[AddedGameStateCount] = gameState.HandState.RemainingHands;
        _remainingDiscards[AddedGameStateCount] = gameState.HandState.RemainingDiscards;

        _score[AddedGameStateCount] = (float)gameState.ScoringState.CurrentRoundTotalScore;
        _scoreThreshold[AddedGameStateCount] = (float)gameState.ScoringState.CurrentRoundThresholdScore;
        WritePlayHandScores(gameState, AddedGameStateCount);

        // store data

        WriteJokerSlots(gameState.ShopState.ShopOfferings, gameState.GameData, _storeJokers, AddedGameStateCount);
        WriteStorePrices(gameState.ShopState.ShopOfferings, _storePrices, AddedGameStateCount);
        _rerollPrice[AddedGameStateCount] = gameState.ShopState.CurrentRerollCost;


        WriteJokerSlots(gameState.JokerState.Jokers, gameState.GameData, _ownedJokers, AddedGameStateCount);
        _money[AddedGameStateCount] = gameState.ShopState.Money;
        _round[AddedGameStateCount] = gameState.Round;
        _isInStore[AddedGameStateCount] =
            (gameState.Stage == StageOfGame.EnterShop || gameState.Stage == StageOfGame.InShop) ? 1 : 0;
        AddedGameStateCount++;
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
        return new()
        {
            FullHand = tensor(_fullHand, dtype: ScalarType.Int64, device: device),
            RemainingDeck = tensor(_remainingDeck, dtype: ScalarType.Int64, device: device),
            RemainingHands = tensor(_remainingHands, dtype: ScalarType.Int64, device: device),
            RemainingDiscards = tensor(_remainingDiscards, dtype: ScalarType.Int64, device: device),
            OwnedJokers = tensor(_ownedJokers, dtype: ScalarType.Int64, device: device),
            StoreJokers = tensor(_storeJokers, dtype: ScalarType.Int64, device: device),
            StorePrices = tensor(_storePrices, dtype: ScalarType.Int64, device: device),
            RerollPrice = tensor(_rerollPrice, dtype: ScalarType.Int64, device: device),
            Money = tensor(_money, dtype: ScalarType.Int64, device: device),
            Round = tensor(_round, dtype: ScalarType.Int64, device: device),
            Stage = tensor(_isInStore, dtype: ScalarType.Int64, device: device),
            PlayHandScores = includePlayHandScores ? tensor(_playHandScores, dtype: ScalarType.Float32, device: device) : null,
            Score = tensor(_score, dtype: ScalarType.Float32, device: device).unsqueeze(1),
            ScoreThreshold = tensor(_scoreThreshold, dtype: ScalarType.Float32, device: device).unsqueeze(1),
        };
    }
}
