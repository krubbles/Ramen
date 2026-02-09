namespace Ramen.AI;

using System.Reflection;
using TorchSharp;
using static TorchSharp.torch;

public interface ITensorGroup
{
}

public static class TensorGroupExtentions
{
    public static ITensorGroup Stack(IReadOnlyList<ITensorGroup> tensorGroups, bool disposeInputs, bool concat, int dim = 0)
    {
        ITensorGroup result = MakeNew(tensorGroups[0].GetType());
        FieldInfo[] fields = GetTensorFields(result.GetType());

        foreach (FieldInfo field in fields)
        {
            if (field.FieldType == typeof(Tensor))
            {
                Tensor[] tensors = new Tensor[tensorGroups.Count];
                for (int i = 0; i < tensorGroups.Count; ++i)
                    tensors[i] = field.GetValue(tensorGroups[i]) as Tensor;
                if (tensors[0] is not null)
                    field.SetValue(result, concat ? cat(tensors, dim) : stack(tensors, dim));
            }
            else if (typeof(ITensorGroup).IsAssignableFrom(field.FieldType))
            {
                ITensorGroup[] tensors = new ITensorGroup[tensorGroups.Count];
                for (int i = 0; i < tensorGroups.Count; ++i)
                    tensors[i] = field.GetValue(tensorGroups[i]) as ITensorGroup;
                field.SetValue(result, Stack(tensors, disposeInputs, concat, dim));
            }

            if (disposeInputs)
            {
                foreach (ITensorGroup tensorGroup in tensorGroups)
                    tensorGroup.Dispose();
            }
        }

        return result;
    }

    public static T Stack<T>(IReadOnlyList<T> tensorGroups, bool disposeInputs, bool concat, int dim = 0) where T : ITensorGroup
    {
        ITensorGroup[] genericGroups = new ITensorGroup[tensorGroups.Count];
        for (int i = 0; i < tensorGroups.Count; ++i)
            genericGroups[i] = tensorGroups[i];
        ITensorGroup result = Stack(genericGroups, disposeInputs, concat, dim);
        return (T)result;
    }

    public static ITensorGroup GetBatch(this ITensorGroup me, int start, int end)
    {
        ITensorGroup result = MakeNew(me.GetType());
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                field.SetValue(result, tensor[start..end]);
            else if (value is ITensorGroup group)
                field.SetValue(result, group.GetBatch(start, end));
        }
        return result;
    }

    public static T GetBatch<T>(this T me, int start, int end) where T : ITensorGroup => (T)GetBatch((ITensorGroup)me, start, end);

    public static ITensorGroup IndexSelect(this ITensorGroup me, int dim, Tensor indices)
    {
        ITensorGroup result = MakeNew(me.GetType());
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                field.SetValue(result, tensor.index_select(dim, indices));
            else if (value is ITensorGroup group)
                field.SetValue(result, group.IndexSelect(dim, indices));
        }
        return result;
    }

    public static T IndexSelect<T>(this T me, int dim, Tensor indices) where T : ITensorGroup => (T)IndexSelect((ITensorGroup)me, dim, indices);

    public static void Swap(this ITensorGroup me, int dim, int index0, int index1)
    {
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
            {
                // Swap two indices along the specified dimension
                using Tensor index0Tensor = TorchSharp.torch.tensor([index0]);
                using Tensor index1Tensor = TorchSharp.torch.tensor([index1]);
                using Tensor temp = tensor.index_select(dim, index0Tensor);
                tensor.index_copy_(dim, index0Tensor, tensor.index_select(dim, index1Tensor));
                tensor.index_copy_(dim, index1Tensor, temp);
            }
            else if (value is ITensorGroup group)
            {
                group.Swap(dim, index0, index1);
            }
        }
    }

    public static void Swap<T>(this T me, int dim, int index0, int index1) where T : ITensorGroup => Swap((ITensorGroup)me, dim, index0, index1);

    public static ITensorGroup Clone(this ITensorGroup me)
    {
        ITensorGroup result = MakeNew(me.GetType());
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                field.SetValue(result, tensor.clone());
            else if (value is ITensorGroup group)
                field.SetValue(result, group.Clone());
        }
        return result;
    }

    public static T Clone<T>(this T me) where T : ITensorGroup => (T)Clone((ITensorGroup)me);

    public static ITensorGroup DetachFromDisposeScope(this ITensorGroup me)
    {
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                tensor.DetachFromDisposeScope();
            else if (value is ITensorGroup group)
                group.DetachFromDisposeScope();
        }
        return me;
    }

    public static T DetachFromDisposeScope<T>(this T me) where T : ITensorGroup => (T)DetachFromDisposeScope((ITensorGroup)me);

    public static ITensorGroup MoveToOuterDisposeScope(this ITensorGroup me)
    {
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                tensor.MoveToOuterDisposeScope();
            else if (value is ITensorGroup group)
                group.MoveToOuterDisposeScope();
        }
        return me;
    }

    public static T MoveToOuterDisposeScope<T>(this T me) where T : ITensorGroup => (T)MoveToOuterDisposeScope((ITensorGroup)me);

    public static void Dispose(this ITensorGroup me)
    {
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
                tensor.Dispose();
            if (value is ITensorGroup tensorGroup)
                tensorGroup.Dispose();
        }
    }

    static ITensorGroup MakeNew(Type type)
    {
        ConstructorInfo constructor = type.GetConstructor(Type.EmptyTypes);
        return (ITensorGroup)constructor.Invoke(null);
    }

    static FieldInfo[] GetTensorFields(Type type)
    {
        if (!_tensorFieldsByType.TryGetValue(type, out FieldInfo[] tensors))
        {
            List<FieldInfo> tList = new(5);
            foreach (FieldInfo field in type.GetFields())
            {
                if (field.FieldType == typeof(Tensor) || typeof(ITensorGroup).IsAssignableFrom(field.FieldType))
                {
                    tList.Add(field);
                }
            }
            tensors = tList.ToArray();
            _tensorFieldsByType.Add(type, tensors);
        }
        return tensors;
    }

    public static ITensorGroup ToDevice(this ITensorGroup me, Device device, bool nonBlocking = false)
    {
        foreach (FieldInfo field in GetTensorFields(me.GetType()))
        {
            object value = field.GetValue(me);
            if (value is Tensor tensor)
            {
                Tensor moved = tensor.to(device, nonBlocking);
                moved.MoveToOuterDisposeScope();
                field.SetValue(me, moved);
            }
            else if (value is ITensorGroup group)
            {
                field.SetValue(me, group.ToDevice(device, nonBlocking));
            }
        }
        return me;
    }

    public static T ToDevice<T>(this T me, Device device, bool nonBlocking = false) where T : ITensorGroup => (T)ToDevice((ITensorGroup)me, device, nonBlocking);
    
    static readonly Dictionary<Type, FieldInfo[]> _tensorFieldsByType = [];
}
