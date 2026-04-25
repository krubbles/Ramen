namespace Ramen.AI;

using System;
using Ramen.AgentTools;
using Ramen.Game;

public sealed class SimpleFlushAgent : IAgent
{
    public bool IsGameDone(GameState gameState)
    {
        return gameState.GameIsDone;
    }


    public void MakeMove(float temp, bool annotatePolicy, params ReadOnlySpan<GameState> gameStates)
    {
        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            gameState.AdvanceToNextPlayerChoice();
            if (gameState.GameIsDone)
                continue;

            UseHandMove[] legalMoves = GetLegalMoves(gameState);
            int chosenMoveIndex = SelectMoveIndex(gameState, legalMoves);
            legalMoves[chosenMoveIndex].Apply(gameState);

            if (annotatePolicy)
                AnnotatePolicy(gameState, legalMoves.Length, chosenMoveIndex);
        }
    }


    public float[][] GetPolicy(float temp, params ReadOnlySpan<GameState> gameStates)
    {
        float[][] policies = new float[gameStates.Length][];

        for (int stateIndex = 0; stateIndex < gameStates.Length; ++stateIndex)
        {
            GameState gameState = gameStates[stateIndex];
            gameState.AdvanceToNextPlayerChoice();
            if (gameState.GameIsDone)
                continue;

            UseHandMove[] legalMoves = GetLegalMoves(gameState);
            int chosenMoveIndex = SelectMoveIndex(gameState, legalMoves);
            policies[stateIndex] = CreateDeterministicPolicy(legalMoves.Length, chosenMoveIndex);
        }

        return policies;
    }


    UseHandMove[] GetLegalMoves(GameState gameState)
    {
        Move[] moves = gameState.GetMoveOptions();
        UseHandMove[] legalMoves = new UseHandMove[moves.Length];
        for (int moveIndex = 0; moveIndex < moves.Length; ++moveIndex)
            legalMoves[moveIndex] = (UseHandMove)moves[moveIndex];
        return legalMoves;
    }


    int SelectMoveIndex(GameState gameState, UseHandMove[] legalMoves)
    {
        EvaluatedMove[] evaluatedMoves = EvaluateMoves(gameState, legalMoves);

        int winningMoveIndex = FindBestWinningPlayIndex(evaluatedMoves);
        if (winningMoveIndex >= 0)
            return winningMoveIndex;

        if (gameState.HandState.RemainingDiscards == 0)
        {
            int highScoreNoDiscardMoveIndex = FindBestPlayOverThresholdIndex(evaluatedMoves, scoreThreshold: 150f);
            if (highScoreNoDiscardMoveIndex >= 0)
                return highScoreNoDiscardMoveIndex;
        }

        if ((float)gameState.ScoringState.CurrentRoundTotalScore > 150f)
            return FindHighestScoringPlayIndex(evaluatedMoves, preferFiveCardHands: true);

        int suitDrivenMoveIndex = FindSuitDrivenMoveIndex(gameState, legalMoves, evaluatedMoves);
        if (suitDrivenMoveIndex >= 0)
            return suitDrivenMoveIndex;

        return FindHighestScoringPlayIndex(evaluatedMoves, preferFiveCardHands: false);
    }


    EvaluatedMove[] EvaluateMoves(GameState gameState, UseHandMove[] legalMoves)
    {
        EvaluatedMove[] evaluatedMoves = new EvaluatedMove[legalMoves.Length];
        Card[] handSnapshot = gameState.HandState.Hand.ToArray();
        float roundScoreBefore = (float)gameState.ScoringState.CurrentRoundTotalScore;

        for (int moveIndex = 0; moveIndex < legalMoves.Length; ++moveIndex)
        {
            UseHandMove move = legalMoves[moveIndex];
            move.Apply(gameState);

            float roundScoreAfter = (float)gameState.ScoringState.CurrentRoundTotalScore;
            float handScore = roundScoreAfter - roundScoreBefore;
            HandPatterns handPatterns = gameState.HandState.ActiveHandPatterns;
            bool isWinningHand = !move.IsDiscard && roundScoreAfter >= 300f;

            evaluatedMoves[moveIndex] = new(
                MoveIndex: moveIndex,
                IsDiscard: move.IsDiscard,
                CardCount: move.CardIndices.Length,
                HandScore: handScore,
                IsWinningHand: isWinningHand,
                FillerKey: GetFillerKey(handSnapshot, move, handPatterns.PlayedCardsMask));

            move.Revert(gameState);
        }

        return evaluatedMoves;
    }


    int FindBestWinningPlayIndex(ReadOnlySpan<EvaluatedMove> evaluatedMoves)
    {
        int bestMoveIndex = -1;
        float bestScore = float.NegativeInfinity;

        for (int moveIndex = 0; moveIndex < evaluatedMoves.Length; ++moveIndex)
        {
            EvaluatedMove move = evaluatedMoves[moveIndex];
            if (!move.IsWinningHand)
                continue;

            if (move.HandScore > bestScore)
            {
                bestScore = move.HandScore;
                bestMoveIndex = move.MoveIndex;
            }
        }

        return bestMoveIndex;
    }


    int FindBestPlayOverThresholdIndex(ReadOnlySpan<EvaluatedMove> evaluatedMoves, float scoreThreshold)
    {
        int bestMoveIndex = -1;
        float bestScore = float.NegativeInfinity;

        for (int moveIndex = 0; moveIndex < evaluatedMoves.Length; ++moveIndex)
        {
            EvaluatedMove move = evaluatedMoves[moveIndex];
            if (move.IsDiscard || move.HandScore <= scoreThreshold)
                continue;

            if (move.HandScore > bestScore)
            {
                bestScore = move.HandScore;
                bestMoveIndex = move.MoveIndex;
            }
        }

        return bestMoveIndex;
    }


    int FindHighestScoringPlayIndex(ReadOnlySpan<EvaluatedMove> evaluatedMoves, bool preferFiveCardHands)
    {
        int bestMoveIndex = -1;
        float bestScore = float.NegativeInfinity;
        FillerKey bestFillerKey = FillerKey.MaxValue;

        for (int moveIndex = 0; moveIndex < evaluatedMoves.Length; ++moveIndex)
        {
            EvaluatedMove move = evaluatedMoves[moveIndex];
            if (move.IsDiscard)
                continue;

            if (move.HandScore > bestScore)
            {
                bestScore = move.HandScore;
                bestMoveIndex = move.MoveIndex;
                bestFillerKey = move.FillerKey;
                continue;
            }

            if (move.HandScore < bestScore)
                continue;

            if (!preferFiveCardHands)
                continue;

            bool moveIsFiveCards = move.CardCount == 5;
            bool bestIsFiveCards = bestMoveIndex >= 0 && evaluatedMoves[bestMoveIndex].CardCount == 5;
            if (moveIsFiveCards && !bestIsFiveCards)
            {
                bestMoveIndex = move.MoveIndex;
                bestFillerKey = move.FillerKey;
                continue;
            }

            if (!moveIsFiveCards || !bestIsFiveCards)
                continue;

            FillerKey moveFillerKey = move.FillerKey;
            if (moveFillerKey.CompareTo(bestFillerKey) < 0)
            {
                bestMoveIndex = move.MoveIndex;
                bestFillerKey = moveFillerKey;
            }
        }

        return bestMoveIndex;
    }


    int FindSuitDrivenMoveIndex(GameState gameState, UseHandMove[] legalMoves, ReadOnlySpan<EvaluatedMove> evaluatedMoves)
    {
        Suit targetSuit = ChooseTargetSuit(gameState);
        byte[] desiredIndices = GetCardsToReplace(gameState.HandState.Hand, targetSuit);
        if (desiredIndices.Length == 0)
            return -1;

        bool wantDiscard = gameState.HandState.RemainingDiscards > 0;
        int exactMatchIndex = FindMoveIndex(legalMoves, wantDiscard, desiredIndices);
        if (exactMatchIndex >= 0)
            return exactMatchIndex;

        return FindHighestScoringPlayIndex(evaluatedMoves, preferFiveCardHands: false);
    }


    Suit ChooseTargetSuit(GameState gameState)
    {
        Suit bestSuit = Suit.Diamond;
        int bestHandCount = int.MinValue;
        int bestDeckCount = int.MinValue;
        int bestPointTotal = int.MinValue;

        for (Suit suit = Suit.Diamond; suit <= Suit.Spade; ++suit)
        {
            int handCount = CountSuit(gameState.HandState.Hand, suit);
            int deckCount = CountSuit(gameState.DeckState.RemainingDeck, suit);
            int pointTotal = GetSuitPointTotal(gameState.HandState.Hand, suit);

            if (handCount > bestHandCount)
            {
                bestSuit = suit;
                bestHandCount = handCount;
                bestDeckCount = deckCount;
                bestPointTotal = pointTotal;
                continue;
            }

            if (handCount < bestHandCount)
                continue;

            if (deckCount > bestDeckCount)
            {
                bestSuit = suit;
                bestDeckCount = deckCount;
                bestPointTotal = pointTotal;
                continue;
            }

            if (deckCount < bestDeckCount)
                continue;

            if (pointTotal > bestPointTotal)
            {
                bestSuit = suit;
                bestPointTotal = pointTotal;
            }
        }

        return bestSuit;
    }


    byte[] GetCardsToReplace(ReadOnlySpan<Card> hand, Suit targetSuit)
    {
        byte[] replaceIndices = new byte[hand.Length];
        int replaceCount = 0;
        int chosenSuitCount = 0;
        (byte cardIndex, int rank)[] chosenSuitCards = new (byte cardIndex, int rank)[hand.Length];

        for (int cardIndex = 0; cardIndex < hand.Length; ++cardIndex)
        {
            Card card = hand[cardIndex];
            if (card.Suit == targetSuit)
            {
                chosenSuitCards[chosenSuitCount++] = ((byte)cardIndex, card.Rank);
                continue;
            }

            replaceIndices[replaceCount++] = (byte)cardIndex;
        }

        if (chosenSuitCount > 5)
        {
            Array.Sort(chosenSuitCards, 0, chosenSuitCount, Comparer<(byte cardIndex, int rank)>.Create(static (left, right) =>
            {
                if (left.rank != right.rank)
                    return left.rank.CompareTo(right.rank);
                return left.cardIndex.CompareTo(right.cardIndex);
            }));

            int extraChosenSuitCount = chosenSuitCount - 5;
            for (int cardIndex = 0; cardIndex < extraChosenSuitCount; ++cardIndex)
                replaceIndices[replaceCount++] = chosenSuitCards[cardIndex].cardIndex;
        }

        Array.Sort(replaceIndices, 0, replaceCount);
        return replaceIndices.AsSpan(0, replaceCount).ToArray();
    }

    int FindMoveIndex(UseHandMove[] legalMoves, bool isDiscard, ReadOnlySpan<byte> desiredIndices)
    {
        for (int moveIndex = 0; moveIndex < legalMoves.Length; ++moveIndex)
        {
            UseHandMove move = legalMoves[moveIndex];
            if (move.IsDiscard != isDiscard)
                continue;

            if (CardIndicesEqual(move.CardIndices, desiredIndices))
                return moveIndex;
        }

        return -1;
    }


    static bool CardIndicesEqual(ReadOnlySpan<byte> left, ReadOnlySpan<byte> right)
    {
        if (left.Length != right.Length)
            return false;

        for (int index = 0; index < left.Length; ++index)
        {
            if (left[index] != right[index])
                return false;
        }

        return true;
    }


    static int CountSuit(ReadOnlySpan<Card> cards, Suit suit)
    {
        int count = 0;
        for (int cardIndex = 0; cardIndex < cards.Length; ++cardIndex)
        {
            if (cards[cardIndex].Suit == suit)
                count++;
        }

        return count;
    }


    static int GetSuitPointTotal(ReadOnlySpan<Card> cards, Suit suit)
    {
        int pointTotal = 0;
        for (int cardIndex = 0; cardIndex < cards.Length; ++cardIndex)
        {
            Card card = cards[cardIndex];
            if (card.Suit != suit)
                continue;

            pointTotal += GameData.BaseChipsForCardRank(card.Rank);
        }

        return pointTotal;
    }


    static FillerKey GetFillerKey(ReadOnlySpan<Card> hand, UseHandMove move, int playedCardsMask)
    {
        if (move.CardIndices.Length != 5)
            return FillerKey.MaxValue;

        int nonScoringCount = 0;
        int[] fillerRanks = [15, 15, 15, 15, 15];
        for (int cardOffset = 0; cardOffset < move.CardIndices.Length; ++cardOffset)
        {
            if (((playedCardsMask >> cardOffset) & 1) != 0)
                continue;

            fillerRanks[nonScoringCount++] = hand[move.CardIndices[cardOffset]].Rank;
        }

        Array.Sort(fillerRanks);
        return new(nonScoringCount, fillerRanks[0], fillerRanks[1], fillerRanks[2], fillerRanks[3], fillerRanks[4]);
    }


    static float[] CreateDeterministicPolicy(int moveCount, int chosenMoveIndex)
    {
        float[] policy = new float[moveCount];
        policy[chosenMoveIndex] = 1f;
        return policy;
    }


    static void AnnotatePolicy(GameState gameState, int moveCount, int chosenMoveIndex)
    {
        float[] policy = CreateDeterministicPolicy(moveCount, chosenMoveIndex);
        AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation(policy);
        annotation.Apply(gameState);
    }


    readonly record struct EvaluatedMove(
        int MoveIndex,
        bool IsDiscard,
        int CardCount,
        float HandScore,
        bool IsWinningHand,
        FillerKey FillerKey);


    readonly record struct FillerKey(
        int NonScoringCount,
        int Rank0,
        int Rank1,
        int Rank2,
        int Rank3,
        int Rank4) : IComparable<FillerKey>
    {
        public static readonly FillerKey MaxValue = new(int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue);

        public int CompareTo(FillerKey other)
        {
            if (NonScoringCount != other.NonScoringCount)
                return NonScoringCount.CompareTo(other.NonScoringCount);
            if (Rank0 != other.Rank0)
                return Rank0.CompareTo(other.Rank0);
            if (Rank1 != other.Rank1)
                return Rank1.CompareTo(other.Rank1);
            if (Rank2 != other.Rank2)
                return Rank2.CompareTo(other.Rank2);
            if (Rank3 != other.Rank3)
                return Rank3.CompareTo(other.Rank3);
            return Rank4.CompareTo(other.Rank4);
        }
    }
}
