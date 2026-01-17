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
    /// Caches each individual component of the <see cref="GameStateTensors"/> class and only updates them when they change.
    /// DOES NOT automatically clone when called. If persistent embedding objects are needed, call .Clone() on the return value.
    /// You may also need to call DetachFromDisposeScope().
    /// </summary>
    public GameStateTensors Tensors => new()
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
    public bool GameIsDone() => GameState.HandState.RemainingHands <= 0 || GameState.ScoringState.CurrentRoundTotalChips >= 300;

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
        return (float)GameState.ScoringState.CurrentRoundTotalChips / 3000f;
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

        (Move[] moves, MoveTensors moveTensors, Tensor probs) = GetPolicyProbDist(temp);
        
        Tensor index = multinomial(probs, num_samples: 1);
        Move move = moves[index.item<long>()];
        move.Apply(GameState);
    }

    /// <summary>
    /// Makes a move based on the policy model's predicted probability distribution.
    /// Also generates a training sample with the chosen move and <paramref name="sampleCount"/> other moves.
    /// Intended to create PPO/GRPO training data.
    /// </summary>
    public EvaluationTrainingSample MakeMoveAndTrainingSample(float temp, int sampleCount = 20)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameState.AdvanceToNextPlayerChoice();

        if (GameIsDone())
            return null;

        (Move[] moves, MoveTensors moveTensors, Tensor probs) = GetPolicyProbDist(temp);
        (Move[] sampledMoves, MoveTensors sampledMoveTensors, Tensor sampledProbs) = SampleMoves(moves, moveTensors, probs, sampleCount);
        Tensor indices = multinomial(probs.view([-1]), sampleCount, replacement: false);

        moves[(int)indices[0].item<long>()].Apply(GameState);

        return CreateEvaluationTrainingSample(sampledMoveTensors, sampledProbs);
    }

    /// <summary>
    /// Samples <paramref name="sampleCount"/> moves based on the policy model's prediction, 
    /// then plays <paramref name="continuationCount"/> continuations after each move. 
    /// Calculates the average reward for all continuations from each move, and makes the move with the best average.
    /// </summary>
    /// <returns>An array containing the indices to all sampled moves, with the highest average reward move at index 0.</returns>
    public ushort[] MakeMoveMonteCarlo(float temp, int sampleCount, int continuationCount)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameState.AdvanceToNextPlayerChoice();

        if (GameIsDone())
            return null;

        (Move[] moves, MoveTensors moveTensors, Tensor probs) = GetPolicyProbDist(temp);

        long[] indices = multinomial(probs, sampleCount).data<long>().ToArray();

        float[] avgRewards = new float[sampleCount];
        int initialStep = GameState.MoveState.MoveStep;

        for (int i = 0; i < sampleCount; i++)
        {
            Move candidateMove = moves[indices[i]];
            float totalReward = 0f;

            candidateMove.Apply(GameState);
            int afterCandidateStep = GameState.MoveState.MoveStep;

            for (int c = 0; c < continuationCount; c++)
            {
                GameState.Reseed();

                while (!GameIsDone())
                {
                    GameState.AdvanceToNextPlayerChoice();
                    MakeMove(temp);
                }

                totalReward += GetCurrentReward();

                GameState.MoveState.RevertToStep(afterCandidateStep);
            }

            avgRewards[i] = totalReward / continuationCount;

            GameState.MoveState.RevertToStep(initialStep);
        }

        // Find the best move based on average rewards
        int bestIndexIndex = 0;
        float bestReward = avgRewards[0];
        for (int i = 1; i < sampleCount; i++)
        {
            if (avgRewards[i] > bestReward)
            {
                bestReward = avgRewards[i];
                bestIndexIndex = i;
            }
        }

        // Swap the best move to index 0
        if (bestIndexIndex != 0)
        {
            (indices[0], indices[bestIndexIndex]) = (indices[bestIndexIndex], indices[0]);
        }

        // Apply the best move (now at index 0)
       moves[indices[bestIndexIndex]].Apply(GameState);

        ushort[] compressedIndices = new ushort[indices.Length];
        for (int i = 0; i < compressedIndices.Length; ++i)
        {
            compressedIndices[i] = (ushort)indices[i];
        }
        return compressedIndices;
    }

    /// <summary>
    /// Returns the policy model's predicted probability distribution for the best next move.
    /// Returned probs is a 1xN tensor where N is the number of moves.
    /// </summary>
    public (Move[] moves, MoveTensors moveTensors, Tensor probs) GetPolicyProbDist(float temp) 
    {
        Move[] moves = GameState.GetMoveOptions();
        (MoveTensors moveTensors, Tensor probs) = GetPolicyProbDistForMoves(temp, moves);
        return (moves, moveTensors, probs);
    }

    /// <summary>
    /// Returns the policy model's predicted probability distribution for the best next move given a selected subset of them.
    /// Returned probs is a 1xN tensor where N is the number of moves.
    /// </summary>
    public (MoveTensors moveTensors, Tensor probs) GetPolicyProbDistForMoves(float temp, Move[] moves) 
    {
        GameStateTensors stateTensors = Tensors;
        Tensor processedState = Model.ProcessState(stateTensors);
        MoveTensors moveTensors = CreateMoveTensors(moves);
        Tensor logits = Model.GetPolicyLogits(moveTensors, processedState);

        Tensor probs = (logits / Math.Max(temp, 0.0001f)).softmax(1);
        return (moveTensors, probs);
    }


    internal EvaluationTrainingSample CreateEvaluationTrainingSample(MoveTensors moveTensors, Tensor probs)
    {
        EvaluationTrainingSample sample = new()
        {
            MoveProbDist = probs.DetachFromDisposeScope(),
            State = Tensors.Clone().DetachFromDisposeScope(),
            Moves = moveTensors.DetachFromDisposeScope(),
            ChosenMoveNLProb = -MathF.Log(Math.Max(probs[0, 0].item<float>(), 1e-9f)),
        };
        return sample;
    }

    public EvaluationTrainingSample CreateMonteCarloTrainingSample(ushort[] moveIndices, float temp) 
    {
        Move[] moves = GameState.GetMoveOptions();
        Move[] sampledMoves = new Move[moveIndices.Length];
        for (int i = 0; i < moveIndices.Length; ++i)
            sampledMoves[i] = moves[moveIndices[i]];
        (MoveTensors sampledMoveTensors, Tensor sampledProbs) = GetPolicyProbDistForMoves(temp, sampledMoves);
        return CreateEvaluationTrainingSample(sampledMoveTensors, sampledProbs);
    }


    private (Move[] sampledMoves, MoveTensors sampledMoveTensors, Tensor sampledProbs) SampleMoves(Move[] moves, MoveTensors moveTensors, Tensor probs, int sampleCount)
    {
        Tensor indices = multinomial(probs.view([-1]), sampleCount, replacement: false);
        Tensor sampledProbs = probs.index_select(dim: 1, indices);
        MoveTensors sampledMoveTensors = moveTensors.IndexSelect(dim: 1, indices);

        long[] indicesArray = indices.data<long>().ToArray();
        Move[] sampledMoves = new Move[sampleCount];
        for (int i = 0; i < sampledMoves.Length; ++i)
            sampledMoves[i] = moves[indicesArray[i]];

        indices.Dispose();

        return (sampledMoves, sampledMoveTensors, sampledProbs);
    }

    /// <summary>
    /// Embeds a list of moves into tensors.
    /// </summary>
    private MoveTensors CreateMoveTensors(Move[] moves)
    {
        int moveCount = moves.Length;
        int[,] playedHands = new int[moveCount, 5];
        int[,] remainingHands = new int[moveCount, 8];
        int[] handsAndDiscards = new int[moveCount];
        float[] scores = new float[moveCount];

        HandState handState = GameState.HandState;
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
            throw new Exception("eee err");

        return new MoveTensors
        {
            RemainingHand = tensor(remainingHands).unsqueeze_(0),
            PlayedHand = tensor(playedHands).unsqueeze_(0),
            HandsAndDiscards = tensor(handsAndDiscards).view([1, -1]),
            Score = tensor(scores).view([1, -1])
        };
    }
    
    static int FindMatchingMoveIndex(Move[] moves, Move expectedMove)
    {
        int expectedIndex = 0;
        for (; expectedIndex < moves.Length; ++expectedIndex)
        {
            UseHandMove move = moves[expectedIndex] as UseHandMove;
            UseHandMove expected = expectedMove as UseHandMove;
            if (move.IsDiscard != expected.IsDiscard)
                continue;
            if (move.CardIndices.Length != expected.CardIndices.Length)
                continue;
            bool match = true;
            for (int i = 0; i < move.CardIndices.Length; ++i)
            {
                if (move.CardIndices[i] != expected.CardIndices[i])
                {
                    match = false;
                    break;
                }
            }
            if (match)
                break;
        }
        if (expectedIndex == moves.Length)
            expectedIndex = -1;
        return expectedIndex;
    }

    /// <summary>
    /// Creates a training sample for a specific move. This will be getting reworked shortly.
    /// </summary>
    public bool CreateTrainingSample(Move expectedMove, float temp, out EvaluationTrainingSample sample, out float nlProb, int sampleCount = 20)
    {
        nlProb = 0f;
        sample = default;
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        Move[] allMoves = GameState.GetMoveOptions();
        if (allMoves.Length == 0)
            return false;
    
        int expectedIndex = FindMatchingMoveIndex(allMoves, expectedMove);
        if (expectedIndex == -1)
            throw new ArgumentException("expectedMove is not in available moves");

        (Move[] moves, MoveTensors moveTensors, Tensor probs) = GetPolicyProbDist(temp);

        float expectedProb = probs[0, expectedIndex].item<float>();
        probs[0, expectedIndex] = 0;
        int samplesNeeded = sampleCount - 1;
        Tensor sampleIndices = multinomial(probs.view(-1), samplesNeeded);
        probs[0, expectedIndex] = expectedProb;
         
        Tensor indices = concat([tensor(expectedIndex).unsqueeze_(0), sampleIndices], dim: 0);
        MoveTensors sampledMoveTensors = moveTensors.IndexSelect(1, indices);
        Tensor sampledProbs = probs.index_select(1, indices);
        sample = CreateEvaluationTrainingSample(moveTensors, probs);

        nlProb = -MathF.Log(Math.Max(expectedProb, 1e-9f));
        return true;
    }

    /// <summary>
    /// Stochastically samples multiple moves using the agent's policy model, not making any of them.
    /// </summary>
    public List<(Move move, float probability)> SampleMoves(float temp, int maxUniqueMoves)
    {
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        Move[] allMoves = GameState.GetMoveOptions();
        if (allMoves.Length == 0)
            return new List<(Move, float)>();
        if (allMoves.Length == 1)
            return new List<(Move, float)> { (allMoves[0], 1f) };

        GameStateTensors stateTensors = Tensors;
        Tensor processedState = Model.ProcessState(stateTensors);

        MoveTensors moveTensors = CreateMoveTensors(allMoves);

        Tensor logits = Model.GetPolicyLogits(moveTensors, processedState);

        Tensor probDist = (logits / Math.Max(temp, 0.0001f)).softmax(1);

        int samplesToGet = Math.Min(maxUniqueMoves, allMoves.Length);
        Tensor indices = multinomial(probDist.view([-1]), samplesToGet, replacement: false);
        Tensor sampledProbs = probDist.index_select(1, indices);
        List<(Move move, float probability)> selected = new();
        long[] sampledIndices = indices.data<long>().ToArray();
        float[] sampledProbsArray = sampledProbs.data<float>().ToArray();
        for (int i = 0; i < samplesToGet; i++)
        {
            int idx = (int)sampledIndices[i];
            selected.Add((allMoves[idx], sampledProbsArray[i]));
        }

        return selected;
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
    /// Evaluates a list of candidate moves using Monte Carlo sampling and returns the move with the highest average reward.
    /// </summary>
    /// <param name="candidateMoves">List of (Move, probability) tuples to evaluate</param>
    /// <param name="samples">Number of Monte Carlo samples to run for each move</param>
    /// <param name="temp">Temperature for the policy when playing out simulations</param>
    /// <returns>The move with the highest average reward across Monte Carlo samples</returns>
    public Move SelectBestMoveMonteCarlo(List<(Move move, float probability)> candidateMoves, int samples, float temp)
    {
        if (candidateMoves.Count == 0)
            return null;
        if (candidateMoves.Count == 1)
            return candidateMoves[0].move;

        float bestAvgReward = float.MinValue;
        Move bestMove = null;

        foreach (var (candidateMove, _) in candidateMoves)
        {
            float totalReward = 0f;

            candidateMove.Apply(GameState);
            int afterCandidateStep = GameState.MoveState.MoveStep;

            for (int c = 0; c < samples; c++)
            {
                GameState.Reseed();

                while (!GameIsDone())
                {
                    GameState.AdvanceToNextPlayerChoice();
                    MakeMove(temp);
                }

                totalReward += GetCurrentReward();

                GameState.MoveState.RevertToStep(afterCandidateStep);
            }

            float avgReward = totalReward / samples;

            if (avgReward > bestAvgReward)
            {
                bestAvgReward = avgReward;
                bestMove = candidateMove;
            }

            GameState.MoveState.RevertToStep(GameState.MoveState.MoveStep - 1);
        }

        return bestMove;
    }

    /// <summary>
    /// Shortcut for <see cref="Tensors"/>.FullHand
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
    /// Shortcut for <see cref="Tensors"/>.RemainingDeck
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
    /// Shortcut for <see cref="Tensors"/>.HandsAndDiscards
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
    /// Shortcut for <see cref="Tensors">.Score    
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
        int[] cards = new int[embedSize];
        for (int i = 0; i < embedSize; ++i)
        {
            cards[i] = i < hand.Length ? hand[i].ToIndex() : 0;
        }

        Tensor handTensor = tensor(cards);
        return handTensor.MoveToOuterDisposeScope();
    }
}

