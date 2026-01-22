namespace Ramen.AI;

using Ramen.Game;
using static TorchSharp.torch;

/// <summary>
/// An embedded GameState.
/// </summary>
public class GameStateTensors : ITensorGroup
{
    /// <summary>
    /// Vector of cards in <see cref="HandState.Hand"/> encoded as integers using <see cref="Card.ToIndex"/> .
    /// </summary>
    public Tensor FullHand;

    /// <summary>
    /// Vector of cards in <see cref="DeckState.RemainingDeck"/> encoded as integers using <see cref="Card.ToIndex"/> .
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
    /// Vector of cards in <see cref="HandState.ActiveHand"/> immediately after move is made. Encoded as integers using <see cref="Card.ToIndex"/>.
    /// </summary>
    public Tensor PlayedHand;

    /// <summary>
    /// Vector of cards in <see cref="HandState.Hand"/> immediately after move is made. Encoded as integers using <see cref="Card.ToIndex"/>.
    /// </summary>
    public Tensor RemainingHand;

    /// <summary>
    /// <see cref="ScoringState.CurrentRoundTotalChips"/> after move is made.
    /// </summary>
    public Tensor Score;
}
