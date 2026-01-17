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
    public static void WriteStartTag(this Stream writer, string text) => writer.WriteStruct(12345);

    public static void WriteEndTag(this Stream writer, string text) => writer.WriteStruct(123456);

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

    public static void WriteArray<T>(this Stream writer, T[] values) where T : unmanaged
    {
        writer.WriteStruct<int>(values.Length);
        writer.WriteSpan<T>(values);
    }

    public static void WriteArrayByteSize<T>(this Stream writer, T[] values) where T : unmanaged
    {
        if (values.Length > 255)
            throw new ArgumentOutOfRangeException($"Cannot byte size serialize an array with length > 255. Length = {values.Length}");
        writer.WriteStruct<byte>((byte)values.Length);
        writer.WriteSpan<T>(values);
    }

    public static void WriteArrayUshortSize<T>(this Stream writer, T[] values) where T : unmanaged
    {
        if (values.Length > 65535)
            throw new ArgumentOutOfRangeException($"Cannot ushort size serialize an array with length > 65535. Length = {values.Length}");
        writer.WriteStruct<ushort>((ushort)values.Length);
        writer.WriteSpan<T>(values);
    }


    public static void ReadStartTag(this Stream reader, string text)
    {
        int tag = reader.ReadStruct<int>();
        if (tag != 12345)
            throw new FormatException($"Failed to find start tag {text}");
    }

    public static void ReadEndTag(this Stream reader, string text)
    {
        int tag = reader.ReadStruct<int>();
        if (tag != 123456)
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

    public static T[] ReadArray<T>(this Stream writer) where T : unmanaged
    {
        int length = writer.ReadStruct<int>();
        T[] data = new T[length];
        writer.ReadSpan<T>(data);
        return data;
    }

    public static T[] ReadArrayByteSize<T>(this Stream writer) where T : unmanaged
    {
        byte length = writer.ReadStruct<byte>();
        T[] data = new T[length];
        writer.ReadSpan<T>(data);
        return data;
    }

    public static T[] ReadArrayUshortSize<T>(this Stream writer) where T : unmanaged
    {
        ushort length = writer.ReadStruct<ushort>();
        T[] data = new T[length];
        writer.ReadSpan<T>(data);
        return data;
    }
}