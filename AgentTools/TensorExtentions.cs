namespace AgentTools;

using System.IO;
using System.Runtime.CompilerServices;
using static TorchSharp.torch;

public static class TensorExtentions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor Size(this Tensor tensor, params ReadOnlySpan<int> expectedDims)
    {
#if DEBUG
        long[] size = tensor.size();
        bool match = expectedDims.Length == size.Length;
        if (match)
        {
            for (int i = 0; i < expectedDims.Length; ++i)
                if (expectedDims[i] == size[i])
                {
                    match = false;
                }
        }
        if (!match)
        {
            throw new InvalidDataException("Tensor dimensions did not match the expected shape.");
        }

        return tensor;
#endif
    }
}
