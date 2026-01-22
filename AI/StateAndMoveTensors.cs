namespace Ramen.AI;

using Ramen.Game;
using static TorchSharp.torch;

// Note: currently embedding is done in Agent.cs, probably worth referencing that.

/// <summary>
/// An embedded GameState.
/// </summary>
public class GameStateTensors : ITensorGroup
{
    /// <summary>
    /// Vector of cards in <see cref="HandState.Hand"/> encoded as [rank, suit] pairs.
    /// </summary>
    public Tensor FullHand;

    /// <summary>
    /// Vector of cards in <see cref="DeckState.RemainingDeck"/> encoded as [rank, suit] pairs.
    /// </summary>
    public Tensor RemainingDeck;

    /// <summary>
    /// <see cref="ScoringState.CurrentRoundTotalChips"/>
    /// </summary>
    public Tensor Score;

    /// <summary>
    /// <see cref="HandState.RemainingHands"/> * 5 + <see cref="HandState.RemainingDiscards"/>
    /// </summary>
    public Tensor HandsAndDiscards;
}

/// <summary>
/// An embedded Move.
/// </summary>
public class MoveTensors : ITensorGroup
{
    /// <summary>
    /// Vector of cards in <see cref="HandState.ActiveHand"/> immediately after move is made. Encoded as [rank, suit] pairs.
    /// </summary>
    public Tensor PlayedHand;

    /// <summary>
    /// Vector of cards in <see cref="HandState.Hand"/> immediately after move is made. Encoded as [rank, suit] pairs.
    /// </summary>
    public Tensor RemainingHand;

    /// <summary>
    /// <see cref="ScoringState.CurrentRoundTotalChips"/> after move is made.
    /// </summary>
    public Tensor Score;
}
