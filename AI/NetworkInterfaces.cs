namespace Ramen.AI;

/// <summary>
/// Interface for policy models used for move evaluation.
/// </summary>
public interface IPolicyNetwork
{
    // batch first
    (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors);
    (Tensor policyLogits, Tensor value) GetPolicyValue(GameStateTensors gameStateTensors, Tensor moveIndices);

    public void Save(string filePath);
    public void Load(string filePath);
}
