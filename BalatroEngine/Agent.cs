namespace BalatroAI;

using static TorchSharp.torch;

public class RamenAgent
{
    public readonly GameState GameState;
    public readonly GameEvalModel Model;

    GameStateTensors _tensors;

    bool _handValid;
    bool _remainingDeckValid;
    bool _fullDeckValid;
    bool _otherStateValid;

    Tensor _processedHand;

    public RamenAgent(GameState gameState, GameEvalModel model)
    {
        GameState = gameState;
        RegisterCallbacks();
    }

    public GameStateTensors Tensors => new()
    {
        Hand = HandTensor,
        FullDeck = FullDeckTensor,
        RemainingDeck = RemainingDeckTensor,
        OtherState = OtherStateTensor,
    };

    public Tensor GetPredictedRewardDistribution()
    {
        return Model.GetPredictedRewardDistribution(ProcessedHandTensor, OtherStateTensor);
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
            return _tensors.FullDeck;
        }
    }

    void EmbedHand()
    {
        _tensors.Hand?.Dispose();
        _tensors.Hand = EmbedCardSet(GameState.HandState.Hand).DetachFromDisposeScope();
        _processedHand = Model.ProcessHand(_tensors.Hand);
    }

    void EmbedRemainingDeck()
    {
        _tensors.RemainingDeck?.Dispose();
        _tensors.RemainingDeck = EmbedCardSet(GameState.DeckState.RemainingDeck).DetachFromDisposeScope();
    }

    void EmbedFullDeck()
    {
        _tensors.FullDeck?.Dispose();
        _tensors.FullDeck = EmbedCardSet(GameState.DeckState.FullDeck).DetachFromDisposeScope();
    }

    void EmbedOtherState()
    {
        _tensors.OtherState?.Dispose();
        _tensors.OtherState = tensor(
        [
            (float)GameState.ScoringState.CurrentRoundTotalChips,
            GameState.HandState.RemainingHands,
            GameState.HandState.RemainingDiscards
        ]).DetachFromDisposeScope();
    }

    void RegisterCallbacks()
    {
        GameState.HandState.OnRemainingHandsOrDiscardsChanged += () => _otherStateValid = false;
        GameState.HandState.OnHandChanged += () => _handValid = false;
        GameState.DeckState.OnRemainingDeckChanged += () => _remainingDeckValid = false;
        GameState.DeckState.OnFullDeckChanged += () => _fullDeckValid = false;
    }

    static Tensor EmbedCardSet(ReadOnlySpan<Card> hand)
    {
        using var scope = NewDisposeScope();
        Tensor[] cards = new Tensor[hand.Length + 1];
        for (int i = 0; i < cards.Length - 1; ++i)
        {
            cards[i] = EmbedCard(hand[i]);
        }
        cards[^1] = EmbedCard(Card.Null);

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

