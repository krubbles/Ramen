namespace Ramen.AgentTools;

using static TorchSharp.torch;

/// <summary>
/// Interface for policy models used for move evaluation.
/// </summary>
public interface IPolicyNetwork
{
    /// <summary>
    /// Returns logits for all possible moves.
    /// Inputs use batch-first tensors; implementations must accept a full hand tensor of shape (batch, 8, 1)
    /// in <paramref name="gameStateTensors"/>. Output must be a tensor of shape (batch, useableHandCount * 2).
    /// </summary>
    Tensor GetPolicyLogits(GameStateTensors gameStateTensors);

    /// <summary>
    /// Returns logits for move indices encoded as (handIndex * 2 + actionIndex).
    /// Inputs use batch-first tensors; implementations must accept a full hand tensor of shape (batch, 8, 1)
    /// in <paramref name="gameStateTensors"/>, and move indices of shape (batch, moveCount). Output must be a tensor
    /// of shape (batch, moveCount).
    /// </summary>
    Tensor GetPolicyLogits(GameStateTensors gameStateTensors, Tensor moveIndices);

    public void Save(string filePath);
    public void Load(string filePath);
}

/// <summary>
/// Scores batched game states with a scalar advantage estimate for each state.
/// </summary>
public interface IValueNetwork
{
    /// <summary>
    /// Returns one advantage value per input game state. Inputs must be batch-first tensors in
    /// <paramref name="gameStateTensors"/>, and the returned tensor must have shape (batch).
    /// </summary>
    Tensor GetValues(GameStateTensors gameStateTensors);

    public void Save(string filePath);
    public void Load(string filePath);
}
