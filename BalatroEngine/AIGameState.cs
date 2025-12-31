namespace BalatroAI;

using BalatroAI.ConsoleApp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

class AIGameState
{
    public AIModels Models;
    public GameState GameState;
    public GameStateTensors _embeddedGameState;
    public FastRandom Random;

    Tensor _embeddedFullHandCache;
    bool _handCachesValid;
    bool _embeddedGameStateValid;

    public int ToUseCount;
    public int[] ToUseIndices = new int[30];
    public bool[] AlreadyPlayedCards = new bool[30];

    readonly float[] _cumWeight = new float[30];

    public Tensor InUseMaskTensor = zeros(1, GameData.HandSize + 1);

    public AIGameState(GameState gameState, AIModels models)
    {
        GameState = gameState;
        Models = models;
        Random = FastRandom.SeededByClock();
    }

    public void CloneFrom(AIGameState other)
    {
        Models = other.Models;
        GameState.CloneFrom(other.GameState);
        GameState.Reseed(Random.Next());
        _embeddedGameState = other._embeddedGameState.Clone();
        _embeddedFullHandCache = other._embeddedFullHandCache?.clone();
        _handCachesValid = other._handCachesValid;
        _embeddedGameStateValid = other._embeddedGameStateValid;

        ToUseCount = other.ToUseCount;
        other.ToUseIndices.CopyTo(ToUseIndices.AsSpan(0, 30));
        other.AlreadyPlayedCards.CopyTo(AlreadyPlayedCards.AsSpan(0, 30));

        InUseMaskTensor.copy_(other.InUseMaskTensor);
    }

    public int CurrentMaxMoveCount()
    {
        if (GameState.HandState.RemainingHands < 4)
            return 0;

        int cardsInHand = GameState.HandState.HandCardCount;
        if (ToUseCount == 0)
            return cardsInHand;
        else
            return cardsInHand + 1; // extra for play hand 
    }

    public bool MoveIsValid(int move) // assumes move is in [0, CurrentMaxMoveCount())
    {
        return !AlreadyPlayedCards[move];
    }

    public void MakeMove(int move) // assumes move is valid
    {
        ToUseIndices[ToUseCount] = move;
        ToUseCount++;

        if (ToUseCount >= GameData.MaxPlayedHandSize || move >= GameState.HandState.HandCardCount) // play hand
        {
            PlayCurrentHand();
            return;
        }
        else
        {
            AlreadyPlayedCards[move] = true;
            InUseMaskTensor[0, move] = 1f;
        }

    }

    public float GetCurrentReward()
    {
        return (float)GameState.ScoringState.CurrentRoundTotalChips / 300f;
    }

#if false
    public Tensor HandTensor
    {
        get
        {
            UpdateHandCaches();
            return _embeddedFullHandCache;
        }
    }
#endif

    void UpdateHandCaches()
    {
        if (_handCachesValid)
            return;
        _handCachesValid = true;
        _embeddedFullHandCache?.Dispose();
        _embeddedFullHandCache = Models.EmbedCards(GameStateTensors.Hand);
        _embeddedFullHandCache.DetachFromDisposeScope();
    }

    public GameStateTensors GameStateTensors
    {
        get
        {
            UpdateGameStateTensors();
            return _embeddedGameState;
        }
    }

    public void UpdateGameStateTensors()
    {
        if (_embeddedGameStateValid)
            return;
        _embeddedGameStateValid = true;
        _embeddedGameState = GameStateTensors.Create(GameState);
        _embeddedGameState.Hand.DetachFromDisposeScope();
        _embeddedGameState.OtherState.DetachFromDisposeScope();
    }

    public void MarkGameStateChanged()
    {
        _handCachesValid = false;
        _embeddedGameStateValid = false;
    }

    public void ResetUseCardsState()
    {
        InUseMaskTensor.zero_();
        ToUseCount = 0;
        Array.Clear(AlreadyPlayedCards);
    }

    public float[] GetExpectedRewardsForMoves(int samples)
    {
        int moveMax = CurrentMaxMoveCount();
        AIGameState copy = new(new GameState(GameState.GameData), Models);
        float greatestReward = -1;
        float[] output = new float[9];
        for (int move = 0; move < output.Length; ++move)
        {
            if (move >= moveMax || !MoveIsValid(move))
            {
                output[move] = -1;
                continue;
            }
            copy.CloneFrom(this);
            copy.MakeMove(move);
            float reward = copy.EstimateTrueReward(samples, TrainingConfig.GoodPlayTemp);
            output[move] = reward;
            if (reward > greatestReward)
                greatestReward = reward;
        }
        return output;
    }

    public float EstimateTrueReward(int samples, float temp)
    {
        AIGameState tester = new(new GameState(GameState.GameData), Models);
        float totalReward = 0;
        for (int i = 0; i < samples; ++i)
        {
            tester.CloneFrom(this);
            while (tester.CurrentMaxMoveCount() > 0)
            {
                tester.MakeMoveStochastic(temp);
            }
            totalReward += tester.GetCurrentReward();
        }
        return totalReward / samples;
    }

    int PlayRandomMove(float[] weights, float temp, bool allowPlayNothing)
    {
        int moves = GameState.HandState.HandCardCount + (allowPlayNothing ? 1 : 0);
        float totalWeight = 0;
        for (int i = 0; i < moves; ++i)
        {
            totalWeight += weights[i];
        }
        float averageWeight = totalWeight / moves;
        for (int i = 0; i < moves; ++i)
        {
            if (!AlreadyPlayedCards[i])
            {
                totalWeight += MathF.Exp(Math.Clamp((weights[i] - averageWeight) / Math.Max(0.00001f, temp), -40, 40));
            }
            _cumWeight[i] = totalWeight;
        }

        float sampleValue = Random.NextPortion() * totalWeight; // [0, 1)
        int playIndex = 0;
        for (int i = 0; i < _cumWeight.Length; ++i)
        {
            if (sampleValue < _cumWeight[i])
            {
                playIndex = i;
                break;
            }
        }

        MakeMove(playIndex);
        return playIndex;
    }

    public void MakeMoveStochastic(float temperature)
    {
        using var scope = NewDisposeScope();

        Tensor cardRewardsTensor = Models.GetCardUseRewards(GameStateTensors.Hand, GameStateTensors.OtherState, InUseMaskTensor);

        int playIndex = PlayRandomMove(cardRewardsTensor.data<float>().ToArray(), temperature, ToUseCount > 0);

    }

    void PlayCurrentHand()
    {
        MarkGameStateChanged();
        ResetUseCardsState();
    }
}

