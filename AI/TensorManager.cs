namespace Ramen.AI;

using TorchSharp;
using static TorchSharp.torch;

public static class TensorManager
{
    public static DisposeScope DisposeScope { get; private set; }
    public static List<Tensor> ForeverTensors { get; } = new();

    public static void Init()
    {
        DisposeScope = NewDisposeScope();
    }

    public static void DisposeAll()
    {
        DisposeScope.DisposeEverythingBut(ForeverTensors);
    }

    public static void PersistForever(Tensor tensor)
    {
        ForeverTensors.Add(tensor);
    }

    public static Tensor ToOuterScope(this Tensor tensor)
    {
        if (!DisposeScope.Contains(tensor))
            tensor.MoveToOuterDisposeScope();
        return tensor;
    }

    public static Tensor DetachFromScope(this Tensor tensor)
    {
        if (!DisposeScope.Contains(tensor))
            tensor.MoveToOtherDisposeScope(DisposeScope);
        return tensor;
    }
}