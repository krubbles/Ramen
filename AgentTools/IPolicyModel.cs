namespace Ramen.AgentTools;

using static TorchSharp.torch;

/// <summary>
/// Interface for policy models used for move evaluation.
/// </summary>
public interface IPolicyModel
{
    /// <summary>
    /// Returns logits for all possible moves.
    /// Inputs use batch-first tensors; implementations must accept a full hand tensor of shape (batch, 8, 1)
    /// in <paramref name="gameStateTensors"/>, and a use-hand score tensor of shape (batch, useableHandCount)
    /// in <paramref name="useHandTensors"/>. Output must be a tensor of shape (batch, useableHandCount * 2).
    /// </summary>
    Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors);

    /// <summary>
    /// Returns logits for move indices encoded as (handIndex * 2 + actionIndex).
    /// Inputs use batch-first tensors; implementations must accept a full hand tensor of shape (batch, 8, 1)
    /// in <paramref name="gameStateTensors"/>, a use-hand score tensor of shape (batch, useableHandCount)
    /// in <paramref name="useHandTensors"/>, and move indices of shape (batch, moveCount). Output must be a
    /// tensor of shape (batch, moveCount).
    /// </summary>
    Tensor GetPolicyLogits(GameStateTensors gameStateTensors, UseHandTensors useHandTensors, Tensor moveIndices);

    public void Save(string filePath);
    public void Load(string filePath);
}
