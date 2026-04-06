namespace Ramen.AgentTools;

using static TorchSharp.torch;

/// <summary>
/// Scores batched game states with a scalar advantage estimate for each state.
/// </summary>
public interface IValueNetwork
{
    /// <summary>
    /// Returns one advantage value per input game state. Inputs must be batch-first tensors in
    /// <paramref name="gameStateTensors"/>, and the returned tensor must have shape (batch).
    /// </summary>
    Tensor GetAdvantages(GameStateTensors gameStateTensors);
}
