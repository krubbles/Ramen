namespace Ramen.ConsoleApp;

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using System.Xml.Linq;

public class ConsoleCommandContext
{
    readonly string[] _words;

    public ConsoleCommandContext(string command)
    {
        List<string> words = new();
        StringBuilder sb = new(); // could be optimzed by using indices instead of a SB
        int contextDepth = 0;
        bool lastQuoteOpened = false;
        for (int i = 0; i < command.Length; ++i)
        {
            char c = command[i];
            if (c == '"')
                contextDepth += lastQuoteOpened ? 1 : -1;
            else if ("([{".Contains(c))
                contextDepth += 1;
            else if ("}])".Contains(c))
                contextDepth -= 1;

            if (contextDepth <= 0)
            {
                contextDepth = 0;
                if (char.IsWhiteSpace(c))
                {
                    if (sb.Length > 0)
                    {
                        words.Add(sb.ToString());
                        sb.Clear();
                    }
                }
                else
                    sb.Append(c);
            }
            else sb.Append(c);

        }
        if (sb.Length > 0)
            words.Add(sb.ToString());
        _words = words.ToArray();
    }

    public string Name => _words.Length > 0 ? _words[0].ToLower() : "";

    public int NumberOfArguments => _words.Length - 1;

    public bool NumberOfArgumentsInRange(int min, int maxInclusive) => NumberOfArguments >= min && NumberOfArguments <= maxInclusive;

    T ThrowBadArgument<T>(string argName, int index) =>
        throw new BadCommandArgumentException(argName, typeof(T), index);

    public int GetIntArg(int index, string name = "unnamed")
    {
        if (_words.Length > index + 1)
        {
            if (int.TryParse(_words[index + 1], out int value))
                return value;
            else return ThrowBadArgument<int>(name, index);
        }
        else return ThrowBadArgument<int>(name, index);
    }

    public float GetFloatArg(int index, string name = "unnamed")
    {
        if (_words.Length > index + 1)
        {
            if (float.TryParse(_words[index + 1], out float value))
                return value;
            else return ThrowBadArgument<float>(name, index);
        }
        else return ThrowBadArgument<float>(name, index);
    }

    public bool GetBoolArg(int index, string name = "unnamed")
    {
        if (_words.Length > index + 1)
        {
            if (bool.TryParse(_words[index + 1], out bool value))
                return value;
            else return ThrowBadArgument<bool>(name, index);
        }
        else return ThrowBadArgument<bool>(name, index);
    }

    public string GetTextArg(int index, string name = "unnamed")
    {
        if (_words.Length > index + 1)
        {
            string text = _words[index + 1];
            if (text.Length >= 2 && text[0] == '"' && text[^1] == '"')
                return text[1..^1];
            return _words[index + 1];
        }
        else return ThrowBadArgument<string>(name, index);
    }

    public T GetEnumArg<T>(int index, string name = "unnamed") where T : struct
    {
        if (_words.Length > index + 1)
        {
            if (Enum.TryParse<T>(_words[index + 1], true, out T result))
                return result;
            else return ThrowBadArgument<T>(name, index);  
        }
        return ThrowBadArgument<T>(name, index);
    }

    class BadCommandArgumentException : Exception
    {
        public readonly string Name;
        public readonly Type Type;
        public readonly int Index;

        public BadCommandArgumentException(string name, Type type, int index)
        {
            Name = name;
            Type = type;
            Index = index;
        }

        public override string ToString()
        {
            return $"expected {Type.Name} argument '{Name}' at index {Index}";
        }
    }
}
