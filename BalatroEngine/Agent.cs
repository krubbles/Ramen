namespace Ramen.AI;

using Ramen.Game;

using static TorchSharp.torch;
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
        FullHand = HandTensor,
        RemainingDeck = RemainingDeckTensor,
        HandsAndDiscards = HandsAndDiscardsTensor,
        Score = ScoreTensor,
    };

    public float GetCurrentReward()
    {
        if (GameState.ScoringState.CurrentRoundTotalChips >= 300)
        {
            return 1f + GameState.HandState.RemainingHands * 0.01f;
        }
        return (float)GameState.ScoringState.CurrentRoundTotalChips / 10000f;
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
        
        // generate embedded moves

        int moveCount = moves.Count;

        int[,] playedHands = new int[moveCount, 5];
        int[,] remainingHands = new int[moveCount, 8];
        int[] handsAndDiscards = new int[moveCount];
        float[] scores = new float[moveCount];
        
        GameStateTensors stateTensors = Tensors;
        Tensor processedState = Model.EmbedState(stateTensors);
        Tensor forcastProbs = null, tier = null;
        if (GameEvalModel.Tiers > 1)
        {
            forcastProbs = Model.GetForcastLogits(processedState).softmax(1);
            tier = multinomial(forcastProbs, 1);
        }
        HandState handState = GameState.HandState;
        ScoringState scoringState = GameState.ScoringState;
        int hash = GameState.GetHashCode();
        for (int move = 0; move < moveCount; ++move)
        {
            moves[move].Apply(GameState);

            UseHandMove useHandMove = (UseHandMove)moves[move];
            for (int i = 0; i < 5; ++i)
                playedHands[move, i] = i < useHandMove.UsedCards.Length ? useHandMove.UsedCards[i].ToIndex() : 0;
            Span<Card> hand = handState.Hand;
            for (int i = 0; i < 8; ++i)
                remainingHands[move, i] = i < hand.Length ? hand[i].ToIndex() : 0;

            handsAndDiscards[move] = GameState.HandState.RemainingHands * 5 + GameState.HandState.RemainingDiscards;
            scores[move] = (float)GameState.ScoringState.CurrentRoundTotalChips / 300f;

            moves[move].Revert(GameState);
        }
        if (GameState.GetHashCode() != hash)
            throw new Exception("eeee err");

        MoveTensors moveTensors = new()
        {
            RemainingHand = tensor(remainingHands).unsqueeze_(0),
            PlayedHand = tensor(playedHands).unsqueeze_(0),
            HandsAndDiscards = tensor(handsAndDiscards).unsqueeze_(0),
            Score = tensor(scores).unsqueeze_(0),
        };

        //

        Tensor logits = Model.ProcessMove(moveTensors, processedState);

        Tensor rewardDist = (logits / Math.Max(temp, 0.0001f)).softmax(1);
        Tensor indices = multinomial(rewardDist, sampleCount, replacement: false).squeeze_(0);
        long moveIndex = indices.data<long>()[0];

        if (generateSample)
        {
            Tensor target = zeros(sampleCount, 1);
            target[0, 0] = 1;
            sample = new()
            {
                ForcastProbDist = forcastProbs?.DetachFromDisposeScope(),
                MoveProbDist = rewardDist.index_select(1, indices).DetachFromDisposeScope(),
                State = stateTensors.Clone().DetachFromDisposeScope(),
                Moves = moveTensors.IndexSelect(1, indices).DetachFromDisposeScope(),
                ForcastTier = tier?.DetachFromDisposeScope()
            };
        }
        moves[(int)moveIndex].Apply(GameState);
        nlProb = -MathF.Log(Math.Max(rewardDist[0, (int)moveIndex].item<float>(), 1e-9f));
        return true;
    }

    public bool GameIsDone() => GameState.HandState.RemainingHands <= 0 || GameState.ScoringState.CurrentRoundTotalChips >= 300;

    public void FillMoveOtherStateData(GameState gameState, Span<float> otherStates, bool isDiscard, float threshold = 0f)
    {
        HandState handState = GameState.HandState;
        ScoringState scoringState = GameState.ScoringState;

        int index = 0;

        otherStates[index++] = (float)scoringState.CurrentRoundTotalChips / 300f;

        otherStates[index++] = handState.RemainingHands == 4 ? 1f : 0f;
        otherStates[index++] = handState.RemainingHands == 3 ? 1f : 0f;
        otherStates[index++] = handState.RemainingHands == 2 ? 1f : 0f;
        otherStates[index++] = handState.RemainingHands == 1 ? 1f : 0f;

        otherStates[index++] = handState.RemainingDiscards == 4 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards == 3 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards == 2 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards == 1 ? 1f : 0f;
        otherStates[index++] = handState.RemainingDiscards == 0 ? 1f : 0f;

        otherStates[index++] = isDiscard ? 0 : 1;
        otherStates[index++] = isDiscard ? 1 : 0;
        otherStates[index++] = threshold;
    }


    public GameStateTensors TensorsCloned => Tensors.Clone().DetachFromDisposeScope();

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

    public Tensor ScoreTensor
    {
        get
        {
            if (!_otherStateValid || _tensors.HandsAndDiscards.IsInvalid)
            {
                _otherStateValid = true;
                EmbedScore();
            }
            return _tensors.HandsAndDiscards;
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
        _tensors.HandsAndDiscards = tensor(score).unsqueeze(0).DetachFromDisposeScope();
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
        int[] cards = new int[embedSize];
        for (int i = 0; i < embedSize; ++i)
        {
            cards[i] = i < hand.Length ? hand[i].ToIndex() : 0;
        }

        Tensor handTensor = tensor(cards);
        return handTensor.MoveToOuterDisposeScope();
    }
}

