namespace Ramen.Game;

using System.IO;

public class GameStateSerializer
{
    public readonly GameState GameState;
    public readonly GameData GameData;
    public readonly Stream Stream;

    public GameStateSerializer(GameState gameState, Stream stream)
    {
        GameState = gameState;
        GameData = gameState.GameData;
        Stream = stream;
    }
}

public static class StreamExtentions
{
    public static unsafe void WriteStartTag(this Stream writer, string text) => writer.WriteStruct(text.GetHashCode());

    public static unsafe void WriteEndTag(this Stream writer, string text) => writer.WriteStruct(0x12345678 + text.GetHashCode());

    public static unsafe void WriteStruct<T>(this Stream writer, T value) where T : unmanaged
    {
        ReadOnlySpan<byte> chars = new(&value, sizeof(T));
        writer.Write(chars);
    }

    public static unsafe void WriteSpan<T>(this Stream writer, ReadOnlySpan<T> values) where T : unmanaged
    {
        fixed (T* valuesPtr = values)
        {
            ReadOnlySpan<byte> bytes = new(valuesPtr, sizeof(T) * values.Length);
            writer.Write(bytes);
        }
    }

    public static void ReadStartTag(this Stream reader, string text)
    {
        int tag = reader.ReadStruct<int>();
        if (tag != text.GetHashCode())
            throw new FormatException($"Failed to find start tag {text}");
    }

    public static void ReadEndTag(this Stream reader, string text)
    {
        int tag = reader.ReadStruct<int>();
        if (tag != 0x12345678 + text.GetHashCode())
            throw new FormatException($"Failed to find end tag {text}");
    }

    public static unsafe T ReadStruct<T>(this Stream reader) where T : unmanaged
    {
        T value = default;
        Span<byte> bytes = new(&value, sizeof(T));
        reader.ReadExactly(bytes);
        return value;
    }

    public static unsafe void ReadSpan<T>(this Stream reader, Span<T> values) where T : unmanaged
    {
        fixed (T* valuesPtr = values)
        {
            Span<byte> bytes = new(valuesPtr, sizeof(T) * values.Length);
            reader.ReadExactly(bytes);
        }
    }
}