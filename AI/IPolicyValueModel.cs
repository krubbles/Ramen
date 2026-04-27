namespace Ramen.AI;

using Ramen.AgentTools;
using static TorchSharp.torch;

public interface IPolicyValueModel : IPolicyModel
{
    Tensor GetValues(GameStateTensors gameStateTensors);
}
