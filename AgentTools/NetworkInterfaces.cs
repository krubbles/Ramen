namespace Ramen.AgentTools;

using static TorchSharp.torch;

/// <summary>
/// Interface for policy models used for move evaluation.
/// </summary>
public interface IPolicyNetwork
{
    // batch first
    (Tensor policyLogits, Tensor value) GetPolicyLogitsAndValue(GameStateTensors gameStateTensors);
    (Tensor policyLogits, Tensor value) GetPolicyLogits(GameStateTensors gameStateTensors, Tensor moveIndices);

    public void Save(string filePath);
    public void Load(string filePath);
}
