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

    Tensor _processedHand;

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

    public float CalculateCurrentReward()
    {
        return (float)GameState.ScoringState.CurrentRoundTotalChips / 100f;
    }

    public (float mean, float dev) GetPredictedRewardDistribution()
    {
        Tensor result = Model.GetPredictedRewardDistribution(ProcessedHandTensor, OtherStateTensor);
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
        GameStateTensors[] states = new GameStateTensors[moves.Count];
        for (int i = 0; i < moves.Count; ++i)
        {
            moves[i].Apply(GameState);
            states[i] = Tensors.Clone();
            moves[i].Revert(GameState);
        }
        GameStateTensors batch = GameStateTensors.Stack(states, true);
        Tensor rewardDist = Model.forward(batch);
        float[] predictedRewards = rewardDist[TensorIndex.Colon, 0].data<float>().ToArray();
        float[] predictedDevs = rewardDist[TensorIndex.Colon, 1].data<float>().ToArray();
        for (int i = 0; i < predictedDevs.Length; ++i)
            predictedDevs[i] = MathF.Sqrt(predictedDevs[i]);
        float[] probDist = MeanDistributionAnalyzer.GetProbabilityDistribution(predictedRewards, predictedDevs);
        int moveIndex = MeanDistributionAnalyzer.SampleFromDistribution(Random, probDist);

        moves[moveIndex].Apply(GameState);
        return true;
    }

    public GameStateTensors TensorsCloned => _tensors.Clone().DetachFromDisposeScope();

    public Tensor HandTensor
    {
        get
        {
            if (!_handValid)
            {
                _handValid = true;
                EmbedHand();
            }
            return _tensors.Hand;
        }
    }

    public Tensor ProcessedHandTensor
    {
        get
        {
            if (!_handValid)
            {
                _handValid = true;
                EmbedHand();
            }
            return _processedHand;
        }
    }

    public Tensor RemainingDeckTensor
    {
        get
        {
            if (!_remainingDeckValid)
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
            if (!_fullDeckValid)
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
            if (!_otherStateValid)
            {
                _otherStateValid = true;
                EmbedOtherState();
            }
            return _tensors.OtherState;
        }
    }

    void EmbedHand()
    {
        _tensors.Hand?.Dispose();
        _tensors.Hand = EmbedCardSet(GameState.HandState.Hand, 8).unsqueeze(0).DetachFromDisposeScope();
        _processedHand = Model.ProcessHand(_tensors.Hand);
    }

    void EmbedRemainingDeck()
    {
        _tensors.RemainingDeck?.Dispose();
        _tensors.RemainingDeck = EmbedCardSet(GameState.DeckState.RemainingDeck, 52).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedFullDeck()
    {
        _tensors.FullDeck?.Dispose();
        _tensors.FullDeck = EmbedCardSet(GameState.DeckState.FullDeck, 52).unsqueeze(0).DetachFromDisposeScope();
    }

    void EmbedOtherState()
    {
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
        using var scope = NewDisposeScope();
        Tensor[] cards = new Tensor[embedSize];
        for (int i = 0; i < embedSize; ++i)
        {
            cards[i] = i < hand.Length ? EmbedCard(hand[i]) : EmbedCard(Card.Null);
        }

        Tensor handTensor = stack(cards);
        return handTensor.MoveToOuterDisposeScope();
    }

    static Tensor EmbedCard(Card card)
    {
        long value;
        if (card.IsNull)
        {
            value = 52; // Use index 52 for null cards
        }
        else
        {
            value = card.Rank - 2 + ((int)card.Suit - 1) * 13;
        }
        return tensor(value, dtype: ScalarType.Int64);
    }
}

