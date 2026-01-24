namespace Ramen.AI;

using Ramen.Game;

using static TorchSharp.torch;

/// <summary>
/// AI agent that plays Balatro.
/// </summary>
public class RamenAgent
{
    /// <summary>
    /// The Balatro game state this AI agent is attached to.
    /// </summary>
    public readonly GameState GameState;

    /// <summary>
    /// A reference to the policy network used by this agent.
    /// </summary>
    public readonly PolicyModel Model;

    /// <summary>
    /// The PS-RNG used by this agent to make decisions like which move to play.
    /// </summary>
    public readonly FastRandom Random;

    GameStateTensors _tensors = new();

    bool _handValid;
    bool _remainingDeckValid;
    bool _fullDeckValid;
    bool _otherStateValid;

    bool _disposeTensorsOnRegen = true;

    public RamenAgent(GameState gameState, PolicyModel model)
    {
        GameState = gameState;
        Model = model;
        Random = FastRandom.SeededByClock();
        RegisterCallbacks();
    }

    /// <summary>
    /// Returns the embedded version of the current game state.
    /// Caches each individual component of the <see cref="AI.GameStateTensors"/> class and only updates them when they change.
    /// DOES NOT automatically clone when called. If persistent embedding objects are needed, call .Clone() on the return value.
    /// You may also need to call DetachFromDisposeScope().
    /// </summary>
    public GameStateTensors GameStateTensors => new()
    {
        FullHand = HandTensor,
        RemainingDeck = RemainingDeckTensor,
        HandsAndDiscards = HandsAndDiscardsTensor,
        Score = ScoreTensor,
    };

    /// <summary>
    /// Returns whether or not the game is complete from the agent's perspective. Many tests and training runs involve subsets of the game,
    /// so this is not the same as when a game of Balatro is typically complete.
    /// </summary>
    public bool GameIsDone() =>
        GameState.ScoringState.CurrentRoundTotalChips >= 1 &&
        (GameState.HandState.RemainingHands <= 0 ||
        GameState.ScoringState.CurrentRoundTotalChips >= 300);

    /// <summary>
    /// Returns the agent's reward at the current GameState. Note: the reward function is only valid when <see cref="GameIsDone"/> returns true.
    /// There is no intermediate reward function in the middle of the game.
    /// </summary>
    public float GetCurrentReward()
    {
        if (GameState.ScoringState.CurrentRoundTotalChips >= 300)
        {
            return 1f + GameState.HandState.RemainingHands * 0.2f;
        }
        return (float)GameState.ScoringState.CurrentRoundTotalChips / 1000f;
    }

    /// <summary>
    /// Makes a move based on the policy model's predicted probability distribution.
    /// </summary>
    public void MakeMove(float temp)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameState.AdvanceToNextPlayerChoice();

        if (GameIsDone())
            return;

        (UseHandTensors moveTensors, Tensor probs) = GetPolicyProbDist(temp);

        Tensor indexTensor = multinomial(probs, num_samples: 1);
        long index = indexTensor.item<long>();
        UseHandMove move = MoveForIndex((int)index);
        move.Apply(GameState);
    }

    public UseHandMove MoveForIndex(int index)
    {
        int[][] useHandOptions = Combinatorics.GetCombinations(
            setSize: GameState.HandState.HandCardCount,
            minSubsetSize: 1,
            maxSubsetSize: 5);
        return new UseHandMove(index % 2 == 1, useHandOptions[index / 2]);
    }

    /// <summary>
    /// Returns the policy model's predicted probability distribution for the best next move.
    /// Returned probs is a 1xN tensor where N is the number of moves.
    /// </summary>
    internal (UseHandTensors moveTensors, Tensor probs) GetPolicyProbDist(float temp)
    {
        (UseHandTensors useHandTensors, _) = CreateUseHandTensors();
        Tensor logits = Model.GetPolicyLogits(GameStateTensors, useHandTensors);
        if (GameState.HandState.RemainingDiscards == 0)
            logits += _maskOutDiscards;
        Tensor probs = (logits / Math.Max(temp, 0.0001f)).softmax(1);
        return (useHandTensors, probs);
    }

    static readonly Tensor _maskOutDiscards = tensor([0.0f, -1e8f]).repeat(218).view(-1).DetachFromDisposeScope();

    /// <summary>
    /// Embeds a list of moves into tensors.
    /// </summary>
    internal (UseHandTensors useHandTensors, int moveCount) CreateUseHandTensors()
    {
        int useHandCount = Combinatorics.CalculateCombinationCount(GameState.HandState.HandCardCount, 5, 1);
        float[] scores = new float[useHandCount];

        HandState handState = GameState.HandState;
        int move = 0;
        foreach (int[] cardIndices in Combinatorics.GetCombinations(handState.HandCardCount, 5))
        {
            UseHandMove useHandMove = new(false, cardIndices);
            useHandMove.Apply(GameState);           
            scores[move] = (float)GameState.ScoringState.CurrentRoundTotalChips / 300f;
            useHandMove.Revert(GameState);
        }
        UseHandTensors useHandTensors = new UseHandTensors
        {
            Score = tensor(scores).view([1, -1]).DetachFromDisposeScope(),
        };
        return (useHandTensors, useHandCount * 2);
    }

    /// <summary>
    /// Plays the hand that scores the most points in the current position. Mostly used for test scenarios.
    /// </summary>
    public bool MakeHighestScoringMove()
    {
        Move[] moves = GameState.GetMoveOptions();
        if (moves.Length == 0)
            return false;
        if (moves.Length == 1)
        {
            moves[0].Apply(GameState);
            return true;
        }
        double bestScore = double.MinValue;
        Move bestMove = null;
        foreach (Move move in moves)
        {
            move.Apply(GameState);
            if (GameState.ScoringState.CurrentRoundTotalChips > bestScore)
            {
                bestScore = GameState.ScoringState.CurrentRoundTotalChips;
                bestMove = move;
            }
            move.Revert(GameState);
        }
        bestMove.Apply(GameState);
        return true;
    }


    /// <summary>
    /// Shortcut for <see cref="GameStateTensors"/>.FullHand
    /// </summary>
    public Tensor HandTensor
    {
        get
        {
            if (!_handValid || _tensors.FullHand.IsInvalid)
            {
                _handValid = true;
                EmbedHand();
            }
            return _tensors.FullHand;
        }
    }

    /// <summary>
    /// Shortcut for <see cref="GameStateTensors"/>.RemainingDeck
    /// </summary>
    public Tensor RemainingDeckTensor
    {
        get
        {
            if (!_remainingDeckValid || _tensors.RemainingDeck.IsInvalid)
            {
                _remainingDeckValid = true;
                EmbedRemainingDeck();
            }
            return _tensors.RemainingDeck;
        }
    }

    /// <summary>
    /// Shortcut for <see cref="GameStateTensors"/>.HandsAndDiscards
    /// </summary>
    public Tensor HandsAndDiscardsTensor
    {
        get
        {
            if (!_otherStateValid || _tensors.HandsAndDiscards.IsInvalid)
            {
                _otherStateValid = true;
                EmbedHandsAndDiscards();
            }
            return _tensors.HandsAndDiscards;
        }
    }

    /// <summary>
    /// Shortcut for <see cref="GameStateTensors">.Score
    /// </summary>
    public Tensor ScoreTensor
    {
        get
        {
            if (!_otherStateValid || (_tensors.Score?.IsInvalid ?? true))
            {
                _otherStateValid = true;
                EmbedScore();
            }
            return _tensors.Score;
        }
    }

    void EmbedHand()
    {
        if (_disposeTensorsOnRegen)
            _tensors.FullHand?.Dispose();
        _tensors.FullHand = TensorizeCardSet(GameState.HandState.Hand, 8).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedRemainingDeck()
    {
        if (_disposeTensorsOnRegen)
            _tensors.RemainingDeck?.Dispose();
        _tensors.RemainingDeck = TensorizeCardSet(GameState.DeckState.RemainingDeck, 52).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedHandsAndDiscards()
    {
        if (_disposeTensorsOnRegen)
            _tensors.HandsAndDiscards?.Dispose();
        int handsAndDiscards = GameState.HandState.RemainingHands * 5 + GameState.HandState.RemainingDiscards;
        _tensors.HandsAndDiscards = tensor(handsAndDiscards).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedScore()
    {
        if (_disposeTensorsOnRegen)
            _tensors.Score?.Dispose();
        float score = (float)GameState.ScoringState.CurrentRoundTotalChips;
        _tensors.Score = tensor(score).view([1, 1]).DetachFromDisposeScope();
    }

    void RegisterCallbacks()
    {
        GameState.HandState.OnRemainingHandsOrDiscardsChanged += () => _otherStateValid = false;
        GameState.HandState.OnHandChanged += () => _handValid = false;
        GameState.DeckState.OnRemainingDeckChanged += () => _remainingDeckValid = false;
        GameState.DeckState.OnFullDeckChanged += () => _fullDeckValid = false;
    }

    static Tensor TensorizeCardSet(ReadOnlySpan<Card> hand, int embedSize)
    {
        long[,] cards = new long[embedSize, 2];
        for (int i = 0; i < embedSize; ++i)
        {
            if (i < hand.Length)
            {
                Card card = hand[i];
                cards[i, 0] = card.Rank - 2;
                cards[i, 1] = (int)card.Suit;
            }
        }

        Tensor handTensor = tensor(cards);
        return handTensor.MoveToOuterDisposeScope();
    }
}
