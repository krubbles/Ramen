namespace Ramen.AI;

using Ramen.Game;

using static TorchSharp.torch;
using static TorchSharp.torch.optim.lr_scheduler.impl.CyclicLR;
using System.Runtime.InteropServices;

public class RamenAgent
{
    public readonly GameState GameState;
    public readonly GameEvalModel Model;
    public readonly FastRandom Random;

    GameStateTensors _tensors = new();

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
        if (GameState.ScoringState.CurrentRoundTotalChips >= 300)
        {
            return 1f + GameState.HandState.RemainingHands * 0.2f;
        }
        return (float)GameState.ScoringState.CurrentRoundTotalChips / 3000f;
    }

    public (float mean, float dev) GetPredictedRewardDistribution()
    {
        Tensor result = Model.forward(Tensors);
        float mean = result.data<float>()[0];
        float dev = MathF.Sqrt(result.data<float>()[1]); // what model actually predicts is variance 
        return (mean, dev);
    }

    public bool MakeMoveStochastic(float temp) => MakeMoveStochastic(temp, out _, out _, 1);

    public bool MakeMoveStochastic(float temp, out EvaluationTrainingSample sample, out float nlProb, int sampleCount = 12, bool generateSample = false)
    {
        nlProb = 0f;
        sample = default;
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
        float[,] otherStates = new float[moveCount, 14];

        Tensor fullHand = HandTensor.clone();

        HandState handState = GameState.HandState;
        ScoringState scoringState = GameState.ScoringState;
        int hash = GameState.GetHashCode();
        for (int move = 0; move < moveCount; ++move)
        {
            moves[move].Apply(GameState);

            Span<Card> hand = handState.Hand;
            for (int i = 0; i < 8; ++i)
                hands[move, i] = i < hand.Length ? hand[i].ToIndex() : 0;

            Span<float> otherStatesBuffer = MemoryMarshal.CreateSpan(ref otherStates[move, 0], 14);
            FillOtherStateData(GameState, otherStatesBuffer, ((UseHandMove)moves[move]).IsDiscard);

            moves[move].Revert(GameState);
        }
        if (GameState.GetHashCode() != hash)
            throw new Exception("eeee err");

        Tensor handsTensor = tensor(hands);
        Tensor otherStatesTensor = tensor(otherStates);

        GameStateTensors batch = new()
        {
            Hand = handsTensor,
            OtherState = otherStatesTensor,
            RemainingDeck = RemainingDeckTensor.expand([moveCount, RemainingDeckTensor.size(1)]),
            FullHand = fullHand.expand([moveCount, fullHand.size(1)]),
        };

        Tensor rewardDist = (Model.forward(batch) / Math.Max(temp, 0.0001f)).softmax(0).squeeze_(1);
        Tensor indices = multinomial(rewardDist, sampleCount, replacement: false);

        long moveIndex = indices.data<long>()[0];
        if (generateSample)
        {
            Tensor target = zeros(sampleCount, 1);
            target[0, 0] = 1;
            sample = new()
            {
                ProbDist = rewardDist.index_select(0, indices).DetachFromDisposeScope(),
                GameStateTensors = batch.IndexSelect(0, indices).DetachFromDisposeScope(),
            };
        }
        moves[(int)moveIndex].Apply(GameState);
        nlProb = -MathF.Log(Math.Max(rewardDist[(int)moveIndex].item<float>(), 1e-9f));
        return true;
    }

    public bool GameIsDone() => GameState.HandState.RemainingHands < 0 || GameState.ScoringState.CurrentRoundTotalChips >= 300;

    public void FillOtherStateData(GameState gameState, Span<float> otherStates, bool isDiscard)
    {
        HandState handState = GameState.HandState;
        ScoringState scoringState = GameState.ScoringState;

        int index = 0;
        // 0
        otherStates[index++] = (float)scoringState.CurrentRoundTotalChips / 300f;
        otherStates[index++] = handState.RemainingHands;
        otherStates[index++] = handState.RemainingHands == 4 ? 1f : 0f;
        otherStates[index++] = handState.RemainingHands == 3 ? 1f : 0f;
        otherStates[index++] = handState.RemainingHands == 2 ? 1f : 0f;
        // 5
        otherStates[index++] = handState.RemainingHands == 1 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards;
        otherStates[index++] = handState.RemainingDiscards == 4 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards == 3 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards == 2 ? 1f : 0f;
        // 10
        otherStates[index++] = handState.RemainingDiscards == 1 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards == 0 ? 1f : 0f;
        otherStates[index++] = isDiscard ? 0 : 1;
        otherStates[index++] = isDiscard ? 1 : 0;
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
        float[] otherState = new float[14];
        FillOtherStateData(GameState, otherState, false);
        _tensors.OtherState = tensor(otherState).unsqueeze(0).DetachFromDisposeScope();
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

