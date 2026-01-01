namespace BalatroAI;

using static TorchSharp.torch;

public class RamenAgent
{
    public readonly GameState GameState;
    public readonly GameEvalModel Model;
    public readonly FastRandom Random;

    GameStateTensors _tensors;

    bool _handValid;
    bool _remainingDeckValid;
    bool _fullDeckValid;
    bool _otherStateValid;

    bool _disposeTensorsOnRegen = true;

    public RamenAgent(GameState gameState, GameEvalModel model)
    {
        GameState = gameState;
        Model = model;
        Random = FastRandom.SeededByClock();
        RegisterCallbacks();
    }

    public GameStateTensors Tensors => new()
    {
        Hand = HandTensor,
        FullDeck = FullDeckTensor,
        RemainingDeck = RemainingDeckTensor,
        OtherState = OtherStateTensor,
    };

    public float GetCurrentReward()
    {
        return (float)GameState.ScoringState.CurrentRoundTotalChips / 100f;
    }

    public (float mean, float dev) GetPredictedRewardDistribution()
    {
        Tensor result = Model.forward(Tensors);
        float mean = result.data<float>()[0];
        float dev = MathF.Sqrt(result.data<float>()[1]); // what model actually predicts is variance 
        return (mean, dev);
    }

    public bool MakeMoveStochastic(float temp)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        List<Move> moves = GameState.GetMoveOptions();
        if (moves.Count == 0)
            return false;
        if (moves.Count == 1)
        {
            moves[0].Apply(GameState);
            return true;
        }

        int moveCount = moves.Count;

        int[,] hands = new int[moveCount, 8];
        float[,] otherStates = new float[moveCount, 3];

        HandState handState = GameState.HandState;
        ScoringState scoringState = GameState.ScoringState;
        for (int move = 0; move < moveCount; ++move)
        {
            moves[move].Apply(GameState);

            Span<Card> hand = handState.Hand;
            for (int i = 0; i < 8; ++i)
            {
                hands[move, i] = i < hand.Length ? hand[i].ToIndex() : 0;
            }

            otherStates[move, 0] = (float)scoringState.CurrentRoundTotalChips;
            otherStates[move, 1] = handState.RemainingHands;
            otherStates[move, 2] = handState.RemainingDiscards;

            moves[move].Revert(GameState);
        }

        Tensor handsTensor = tensor(hands);
        Tensor otherStatesTensor = tensor(otherStates);

        GameStateTensors batch = new()
        {
            Hand = handsTensor,
            OtherState = otherStatesTensor
        };

        Tensor rewardDist = Model.forward(batch);
        float[] rewards = rewardDist[TensorIndex.Colon, 0].data<float>().ToArray();
        float max = float.MinValue;
        for (int i = 0; i < rewards.Length; ++i)
            max = Math.Max(max, rewards[i]);
        float total = 0;
        for (int i = 0; i < rewards.Length; ++i)
        {
            float r = MathF.Exp((rewards[i] - max) / Math.Max(temp, 0.0001f));
            rewards[i] = r;
            total += r;
        }
        for (int i = 0; i < rewards.Length; ++i)
            rewards[i] /= total;

        int moveIndex = MeanDistributionAnalyzer.SampleFromDistribution(Random, rewards);
        moves[moveIndex].Apply(GameState);

        return true;
    }



    public GameStateTensors TensorsCloned => Tensors.Clone().DetachFromDisposeScope();

    public Tensor HandTensor
    {
        get
        {
            if (!_handValid || _tensors.Hand.IsInvalid)
            {
                _handValid = true;
                EmbedHand();
            }
            return _tensors.Hand;
        }
    }

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

    public Tensor FullDeckTensor
    {
        get
        {
            if (!_fullDeckValid || _tensors.FullDeck.IsInvalid)
            {
                _fullDeckValid = true;
                EmbedFullDeck();
            }
            return _tensors.FullDeck;
        }
    }

    public Tensor OtherStateTensor
    {
        get
        {
            if (!_otherStateValid || _tensors.OtherState.IsInvalid)
            {
                _otherStateValid = true;
                EmbedOtherState();
            }
            return _tensors.OtherState;
        }
    }

    void EmbedHand()
    {
        if (_disposeTensorsOnRegen)
            _tensors.Hand?.Dispose();
        _tensors.Hand = EmbedCardSet(GameState.HandState.Hand, 8).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedRemainingDeck()
    {
        if (_disposeTensorsOnRegen)
            _tensors.RemainingDeck?.Dispose();
        _tensors.RemainingDeck = EmbedCardSet(GameState.DeckState.RemainingDeck, 52).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedFullDeck()
    {
        if (_disposeTensorsOnRegen)
            _tensors.FullDeck?.Dispose();
        _tensors.FullDeck = EmbedCardSet(GameState.DeckState.FullDeck, 52).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedOtherState()
    {
        if (_disposeTensorsOnRegen)
            _tensors.OtherState?.Dispose();
        _tensors.OtherState = tensor(
        [
            (float)GameState.ScoringState.CurrentRoundTotalChips,
            GameState.HandState.RemainingHands,
            GameState.HandState.RemainingDiscards
        ]).unsqueeze(0).DetachFromDisposeScope();
    }

    void RegisterCallbacks()
    {
        GameState.HandState.OnRemainingHandsOrDiscardsChanged += () => _otherStateValid = false;
        GameState.HandState.OnHandChanged += () => _handValid = false;
        GameState.DeckState.OnRemainingDeckChanged += () => _remainingDeckValid = false;
        GameState.DeckState.OnFullDeckChanged += () => _fullDeckValid = false;
    }

    static Tensor EmbedCardSet(ReadOnlySpan<Card> hand, int embedSize)
    {
        int[] cards = new int[embedSize];
        for (int i = 0; i < embedSize; ++i)
        {
            cards[i] = i < hand.Length ? hand[i].ToIndex() : 0;
        }

        Tensor handTensor = tensor(cards);
        return handTensor.MoveToOuterDisposeScope();
    }
}

