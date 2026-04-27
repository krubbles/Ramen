namespace Ramen.AI;

using System.Diagnostics;
using System.Globalization;
using Ramen.AgentTools;
using Ramen.Game;
using static TorchSharp.torch;

public sealed class GameTurnTrace
{
    public int TurnIndex { get; init; }
    public string StateText { get; init; }
    public float Score { get; init; }
    public int RemainingHands { get; init; }
    public int RemainingDiscards { get; init; }
    public int RemainingDeck { get; init; }
    public float ThresholdValue { get; init; }
    public float ElapsedSeconds { get; init; }
    public int TotalTrajectoryCount { get; init; }
    public int ChosenMoveIndex { get; init; }
    public string ChosenMoveText { get; init; }
    public string TopMoveText { get; init; }
    public string ConsideredMovesText { get; init; }
    public List<CandidateMoveTrace> Candidates { get; init; } = [];
}

public sealed class CandidateMoveTrace
{
    public int OriginalPolicyRank { get; init; }
    public float PolicyProbability { get; init; }
    public int MoveIndex { get; init; }
    public string MoveText { get; init; }
    public bool IsTopMove { get; init; }
    public bool IsConsidered { get; set; } = true;
    public RunningStats TrajectoryStats { get; } = new();
    public List<TrajectorySampleTrace> TrajectorySamples { get; } = [];
}

public readonly record struct TrajectorySampleTrace(float Value, string HashPath);

public sealed class RunningStats
{
    float _m2;

    public int Count { get; private set; }

    public float Mean { get; private set; }

    public float SampleVariance => Count > 1 ? _m2 / (Count - 1) : 0f;

    public float SampleStandardDeviation => MathF.Sqrt(MathF.Max(0f, SampleVariance));


    public void Add(float value)
    {
        Count++;
        float delta = value - Mean;
        Mean += delta / Count;
        float delta2 = value - Mean;
        _m2 += delta * delta2;
    }
}

public sealed class TrajectoryPruningAgent : IAgent, IDisposable
{
    static readonly Device EvalDevice = mps_is_available() ? MPS : CPU;
    static readonly int[][] HandCombinations = Combinatorics.GetCombinations(
        setSize: GameData.HandSize,
        minSubsetSize: 1,
        maxSubsetSize: GameData.MaxPlayedHandSize);
    static readonly Dictionary<string, int> MoveIndexLookup = BuildMoveIndexLookup();

    readonly IPolicyValueModel _model;
    readonly PolicyOnlyAgent _policyAgent;
    readonly FastRandom _random;
    readonly float _entropyLimit;
    readonly int _topMoveTrajectoryCount;
    readonly int _initialOtherMoveTrajectoryCount;
    readonly int _topMoveCount;
    readonly int _maxTrajectoryCount;
    readonly int _additionalTrajectoryCountPerRound;

    public TrajectoryPruningAgent(IPolicyValueModel model, float entropyLimit, int randomSeed, int topMoveTrajectoryCount, int initialOtherMoveTrajectoryCount, int topMoveCount, int maxTrajectoryCount, int additionalTrajectoryCountPerRound)
    {
        _model = model;
        _policyAgent = new(model, ownsModel: false);
        _random = new((ulong)randomSeed);
        _entropyLimit = entropyLimit;
        _topMoveTrajectoryCount = topMoveTrajectoryCount;
        _initialOtherMoveTrajectoryCount = initialOtherMoveTrajectoryCount;
        _topMoveCount = topMoveCount;
        _maxTrajectoryCount = maxTrajectoryCount;
        _additionalTrajectoryCountPerRound = additionalTrajectoryCountPerRound;
    }


    public void Dispose()
    {
        _policyAgent.Dispose();
    }


    public void MakeMove(float temp, bool annotatePolicy, params ReadOnlySpan<GameState> gameStates)
    {
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            gameState.AdvanceToNextPlayerChoice();
            if (gameState.GameIsDone)
                continue;

            GameTurnTrace trace = ChooseMoveWithTrace(gameState, turnIndex: stateIndex + 1);
            PolicyOnlyAgent.MoveForIndex(gameState, trace.ChosenMoveIndex).Apply(gameState);
            _ = temp;
            _ = annotatePolicy;
        }
    }


    public float[][] GetPolicy(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        return _policyAgent.GetPolicy(temp, gameStates);
    }


    public bool IsGameDone(GameState gameState)
    {
        return _policyAgent.IsGameDone(gameState);
    }


    public GameTurnTrace ChooseMoveWithTrace(GameState gameState, int turnIndex)
    {
        Stopwatch stopwatch = Stopwatch.StartNew();
        gameState.AdvanceToNextPlayerChoice();

        RankedMove[] rankedMoves = GetRankedLegalMoves(gameState);
        int candidateCount = Math.Min(_topMoveCount, rankedMoves.Length);
        List<CandidateMoveTrace> candidates = [];
        for (int rankIndex = 0; rankIndex < candidateCount; ++rankIndex)
        {
            RankedMove rankedMove = rankedMoves[rankIndex];
            candidates.Add(new()
            {
                OriginalPolicyRank = rankIndex + 1,
                PolicyProbability = rankedMove.PolicyProbability,
                MoveIndex = rankedMove.MoveIndex,
                MoveText = rankedMove.MoveText,
                IsTopMove = rankIndex == 0,
            });
        }

        byte[] serializedState = SerializeGameState(gameState);

        int totalTrajectoryCount = 0;
        for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
        {
            CandidateMoveTrace candidate = candidates[candidateIndex];
            int sampleCount = candidate.IsTopMove ? _topMoveTrajectoryCount : _initialOtherMoveTrajectoryCount;
            AddTrajectorySamples(gameState.GameData, serializedState, candidate, sampleCount);
            totalTrajectoryCount += sampleCount;
        }

        float thresholdValue = candidates[0].TrajectoryStats.Mean - candidates[0].TrajectoryStats.SampleStandardDeviation;
        PruneCandidates(candidates, thresholdValue);

        while (totalTrajectoryCount < _maxTrajectoryCount && CountConsidered(candidates) > 1)
        {
            for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
            {
                CandidateMoveTrace candidate = candidates[candidateIndex];
                if (!candidate.IsConsidered)
                    continue;

                for (int sampleIndex = 0; sampleIndex < _additionalTrajectoryCountPerRound && totalTrajectoryCount < _maxTrajectoryCount; ++sampleIndex)
                {
                    AddTrajectorySamples(gameState.GameData, serializedState, candidate, sampleCount: 1);
                    totalTrajectoryCount++;
                }
            }

            PruneCandidates(candidates, thresholdValue);
        }

        CandidateMoveTrace chosenMove = candidates
            .Where(candidate => candidate.IsConsidered)
            .OrderByDescending(candidate => candidate.TrajectoryStats.Mean)
            .ThenBy(candidate => candidate.OriginalPolicyRank)
            .First();

        string consideredMovesText = string.Join(
            " | ",
            candidates
                .Where(candidate => candidate.IsConsidered)
                .Select(candidate => $"#{candidate.OriginalPolicyRank} {candidate.MoveText}"));

        return new()
        {
            TurnIndex = turnIndex,
            StateText = DescribeState(gameState),
            Score = (float)gameState.ScoringState.CurrentRoundTotalScore,
            RemainingHands = gameState.HandState.RemainingHands,
            RemainingDiscards = gameState.HandState.RemainingDiscards,
            RemainingDeck = gameState.DeckState.RemainingDeckCardCount,
            ThresholdValue = thresholdValue,
            ElapsedSeconds = (float)stopwatch.Elapsed.TotalSeconds,
            TotalTrajectoryCount = totalTrajectoryCount,
            ChosenMoveIndex = chosenMove.MoveIndex,
            ChosenMoveText = chosenMove.MoveText,
            TopMoveText = candidates[0].MoveText,
            ConsideredMovesText = consideredMovesText,
            Candidates = candidates,
        };
    }


    void AddTrajectorySamples(GameData gameData, byte[] serializedState, CandidateMoveTrace candidate, int sampleCount)
    {
        TrajectorySampleTrace[] rewards = SimulateTrajectories(gameData, serializedState, candidate.MoveIndex, sampleCount);
        for (int rewardIndex = 0; rewardIndex < rewards.Length; ++rewardIndex)
        {
            candidate.TrajectoryStats.Add(rewards[rewardIndex].Value);
            candidate.TrajectorySamples.Add(rewards[rewardIndex]);
        }
    }


    TrajectorySampleTrace[] SimulateTrajectories(GameData gameData, byte[] serializedState, int initialMoveIndex, int sampleCount)
    {
        TrajectorySampleTrace[] rewards = new TrajectorySampleTrace[sampleCount];
        List<TrajectorySimulation> initialSimulations = [];

        for (int simulationIndex = 0; simulationIndex < sampleCount; ++simulationIndex)
        {
            GameState simulationState = CloneGameState(gameData, serializedState);
            PolicyOnlyAgent.MoveForIndex(simulationState, initialMoveIndex).Apply(simulationState);
            simulationState.Reseed();
            simulationState.AdvanceToNextPlayerChoice();

            TrajectorySimulation simulation = new(simulationState, simulationIndex);
            initialSimulations.Add(simulation);
        }

        AddCurrentStateRewards(initialSimulations);

        List<TrajectorySimulation> activeSimulations = [];
        for (int simulationIndex = 0; simulationIndex < initialSimulations.Count; ++simulationIndex)
        {
            TrajectorySimulation simulation = initialSimulations[simulationIndex];
            if (simulation.GameState.GameIsDone)
                rewards[simulation.OutputIndex] = simulation.ToSample();
            else
                activeSimulations.Add(simulation);
        }

        while (activeSimulations.Count > 0)
        {
            GameState[] states = new GameState[activeSimulations.Count];
            for (int stateIndex = 0; stateIndex < activeSimulations.Count; ++stateIndex)
                states[stateIndex] = activeSimulations[stateIndex].GameState;

            float[][] policies = _policyAgent.GetPolicy(temp: 1f, states);

            for (int stateIndex = 0; stateIndex < activeSimulations.Count; ++stateIndex)
            {
                TrajectorySimulation simulation = activeSimulations[stateIndex];
                float[] policy = policies[stateIndex];
                float entropy = CalculateEntropy(policy);
                int sampledMoveIndex = AgentUtilities.SampleIndex(policy, _random);
                PolicyOnlyAgent.MoveForIndex(simulation.GameState, sampledMoveIndex).Apply(simulation.GameState);
                simulation.GameState.AdvanceToNextPlayerChoice();
                simulation.CumulativeEntropy += entropy;
            }

            AddCurrentStateRewards(activeSimulations);

            List<TrajectorySimulation> nextActiveSimulations = [];
            for (int stateIndex = 0; stateIndex < activeSimulations.Count; ++stateIndex)
            {
                TrajectorySimulation simulation = activeSimulations[stateIndex];
                if (simulation.GameState.GameIsDone || simulation.CumulativeEntropy >= _entropyLimit)
                    rewards[simulation.OutputIndex] = simulation.ToSample();
                else
                    nextActiveSimulations.Add(simulation);
            }

            activeSimulations = nextActiveSimulations;
        }

        return rewards;
    }


    void AddCurrentStateRewards(IReadOnlyList<TrajectorySimulation> simulations)
    {
        List<TrajectorySimulation> nonTerminalSimulations = [];
        for (int simulationIndex = 0; simulationIndex < simulations.Count; ++simulationIndex)
        {
            TrajectorySimulation simulation = simulations[simulationIndex];
            if (simulation.GameState.GameIsDone)
            {
                simulation.AddReward(GetStandardReward(simulation.GameState));
                continue;
            }

            nonTerminalSimulations.Add(simulation);
        }

        if (nonTerminalSimulations.Count == 0)
            return;

        using var scope = NewDisposeScope();
        using var noGrad = no_grad();

        GameStateEmbedder gameStateEmbedder = new(nonTerminalSimulations.Count);
        for (int simulationIndex = 0; simulationIndex < nonTerminalSimulations.Count; ++simulationIndex)
            gameStateEmbedder.AddGameState(nonTerminalSimulations[simulationIndex].GameState);

        GameStateTensors gameStateTensors = gameStateEmbedder.ToTensors(EvalDevice);
        Tensor values = _model.GetValues(gameStateTensors).to(CPU);
        float[] valueData = values.data<float>().ToArray();
        gameStateTensors.Dispose();

        for (int simulationIndex = 0; simulationIndex < nonTerminalSimulations.Count; ++simulationIndex)
            nonTerminalSimulations[simulationIndex].AddReward(valueData[simulationIndex]);
    }


    RankedMove[] GetRankedLegalMoves(GameState gameState)
    {
        float[][] policies = _policyAgent.GetPolicy(temp: 1f, gameState);
        float[] policy = policies[0];
        Move[] legalMoves = gameState.GetMoveOptions();
        RankedMove[] rankedMoves = new RankedMove[legalMoves.Length];
        for (int moveIndex = 0; moveIndex < legalMoves.Length; ++moveIndex)
        {
            Move move = legalMoves[moveIndex];
            UseHandMove useHandMove = (UseHandMove)move;
            int policyMoveIndex = GetMoveIndex(useHandMove);
            rankedMoves[moveIndex] = new(
                MoveIndex: policyMoveIndex,
                PolicyProbability: policy[policyMoveIndex],
                MoveText: FormatMove(gameState, move));
        }

        Array.Sort(rankedMoves, static (left, right) => right.PolicyProbability.CompareTo(left.PolicyProbability));
        return rankedMoves;
    }


    static void PruneCandidates(List<CandidateMoveTrace> candidates, float thresholdValue)
    {
        for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
        {
            CandidateMoveTrace candidate = candidates[candidateIndex];
            if (candidate.IsTopMove)
            {
                candidate.IsConsidered = true;
                continue;
            }

            if (!candidate.IsConsidered)
                continue;

            float upperConfidence = candidate.TrajectoryStats.Mean + 2f * candidate.TrajectoryStats.SampleStandardDeviation;
            if (upperConfidence < thresholdValue)
                candidate.IsConsidered = false;
        }
    }


    static int CountConsidered(List<CandidateMoveTrace> candidates)
    {
        int count = 0;
        for (int candidateIndex = 0; candidateIndex < candidates.Count; ++candidateIndex)
        {
            if (candidates[candidateIndex].IsConsidered)
                count++;
        }

        return count;
    }


    static float CalculateEntropy(ReadOnlySpan<float> probabilities)
    {
        float entropy = 0f;
        for (int probabilityIndex = 0; probabilityIndex < probabilities.Length; ++probabilityIndex)
        {
            float probability = probabilities[probabilityIndex];
            if (probability <= 0f)
                continue;

            entropy -= probability * MathF.Log(MathF.Max(probability, 1e-9f));
        }

        return entropy;
    }


    static string FormatMove(GameState gameState, Move move)
    {
        if (move is not UseHandMove useHandMove)
            return move.ToString();

        Card[] cards = new Card[useHandMove.CardIndices.Length];
        for (int cardIndex = 0; cardIndex < useHandMove.CardIndices.Length; ++cardIndex)
            cards[cardIndex] = gameState.HandState.Hand[useHandMove.CardIndices[cardIndex]];

        string action = useHandMove.IsDiscard ? "Discard" : "Play";
        string cardText = CardParseUtils.SerializeHand(cards);
        string indexText = string.Join(",", useHandMove.CardIndices);
        return $"{action} Hand: {cardText} [idx:{indexText}]";
    }


    static byte[] SerializeGameState(GameState gameState)
    {
        using MemoryStream stream = new();
        gameState.Serialize(stream);
        return stream.ToArray();
    }


    static GameState CloneGameState(GameData gameData, byte[] serializedState)
    {
        GameState clonedState = new(gameData);
        using MemoryStream stream = new(serializedState, writable: false);
        clonedState.Deserialize(stream);
        return clonedState;
    }


    static int GetMoveIndex(UseHandMove move)
    {
        string key = GetMoveKey(move.CardIndices);
        int handIndex = MoveIndexLookup[key];
        return handIndex * 2 + (move.IsDiscard ? 1 : 0);
    }


    static Dictionary<string, int> BuildMoveIndexLookup()
    {
        Dictionary<string, int> lookup = [];
        for (int handIndex = 0; handIndex < HandCombinations.Length; ++handIndex)
            lookup[GetMoveKey(HandCombinations[handIndex])] = handIndex;
        return lookup;
    }


    static string GetMoveKey(ReadOnlySpan<byte> cardIndices)
    {
        if (cardIndices.Length == 0)
            return "";

        string[] parts = new string[cardIndices.Length];
        for (int index = 0; index < cardIndices.Length; ++index)
            parts[index] = cardIndices[index].ToString(CultureInfo.InvariantCulture);
        return string.Join(",", parts);
    }


    static string GetMoveKey(ReadOnlySpan<int> cardIndices)
    {
        if (cardIndices.Length == 0)
            return "";

        string[] parts = new string[cardIndices.Length];
        for (int index = 0; index < cardIndices.Length; ++index)
            parts[index] = cardIndices[index].ToString(CultureInfo.InvariantCulture);
        return string.Join(",", parts);
    }


    static string DescribeState(GameState gameState)
    {
        return $"{gameState} | Score {gameState.ScoringState.CurrentRoundTotalScore:F1} | Deck {gameState.DeckState.RemainingDeckCardCount}";
    }


    static float GetStandardReward(GameState gameState)
    {
        float roundsSurvived = gameState.Round / 3f;
        return roundsSurvived * roundsSurvived;
    }


    readonly record struct RankedMove(int MoveIndex, float PolicyProbability, string MoveText);

    sealed class TrajectorySimulation
    {
        public readonly GameState GameState;
        public readonly int OutputIndex;
        readonly List<int> _rewardStateHashes = [];

        public float CumulativeEntropy;
        float _rewardTotal;
        int _rewardCount;

        public TrajectorySimulation(GameState gameState, int outputIndex)
        {
            GameState = gameState;
            OutputIndex = outputIndex;
        }

        public float MeanReward => _rewardCount == 0 ? 0f : _rewardTotal / _rewardCount;


        public void AddReward(float reward)
        {
            _rewardTotal += reward;
            _rewardCount++;
            _rewardStateHashes.Add(GameState.GetHashCode());
        }


        public TrajectorySampleTrace ToSample()
        {
            string[] hashParts = new string[_rewardStateHashes.Count];
            for (int hashIndex = 0; hashIndex < _rewardStateHashes.Count; ++hashIndex)
                hashParts[hashIndex] = _rewardStateHashes[hashIndex].ToString(CultureInfo.InvariantCulture);

            return new(MeanReward, string.Join("|", hashParts));
        }
    }
}
