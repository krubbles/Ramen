namespace Ramen.AI;

using TorchSharp;
using static TorchSharp.torch;

public static class TensorManager
{
    public static DisposeScope DisposeScope { get; private set; }
    
    public static void Init()
    {
        DisposeScope = NewDisposeScope();
    }

    public static void DisposeAll()
    {
        DisposeScope.DisposeEverything();
    }

    public static void ToOuterScope(this Tensor tensor)
    {
        if (!DisposeScope.Contains(tensor))
            tensor.MoveToOuterDisposeScope();
    }

    public static void DetachFromScope(this Tensor tensor)
    {
        if (!DisposeScope.Contains(tensor))
            tensor.MoveToOtherDisposeScope(DisposeScope);
    }
}