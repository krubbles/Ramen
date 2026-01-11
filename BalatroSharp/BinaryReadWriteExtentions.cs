namespace Ramen.Game;

using System.IO;

public static class BinaryWriterExtentions
{
    public static unsafe void StartTag(this BinaryWriter writer, string text) => writer.WriteStruct(text.GetHashCode());

    public static unsafe void EndTag(this BinaryWriter writer, string text) => writer.WriteStruct(0x12345678 + text.GetHashCode());

    public static unsafe void WriteStruct<T>(this BinaryWriter writer, T value) where T : unmanaged
    {
        ReadOnlySpan<byte> chars = new(&value, sizeof(T));
        writer.Write(chars);
    }

    public static unsafe void WriteSpan<T>(this BinaryWriter writer, ReadOnlySpan<T> values) where T : unmanaged
    {
        fixed (T* valuesPtr = values)
        {
            ReadOnlySpan<byte> bytes = new(valuesPtr, sizeof(T) * values.Length);
            writer.Write(bytes);
        }
    }

}

public static class BinaryReaderExtentions
{
    public static void StartTag(this BinaryReader reader, string text)
    {
        int tag = reader.ReadStruct<int>();
        if (tag != text.GetHashCode())
            throw new FormatException($"Failed to find start tag {text}");
    }

    public static void EndTag(this BinaryReader reader, string text)
    {
        int tag = reader.ReadStruct<int>();
        if (tag != 0x12345678 + text.GetHashCode())
            throw new FormatException($"Failed to find end tag {text}");
    }

    public static unsafe T ReadStruct<T>(this BinaryReader reader) where T : unmanaged
    {
        T value = default;
        Span<byte> bytes = new(&value, sizeof(T));
        int readCount = reader.Read(bytes);
        return value;
    }

    public static unsafe void ReadSpan<T>(this BinaryReader reader, Span<T> values) where T : unmanaged
    {
        fixed (T* valuesPtr = values)
        {
            Span<byte> bytes = new(valuesPtr, sizeof(T) * values.Length);
            reader.Read(bytes);
        }
    }

}