namespace Ramen.ConsoleApp;

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using System.Xml.Linq;

public class ConsoleCommandContext
{
    readonly string[] _words;
    readonly Dictionary<string, string> _optionalArgs;

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
            {
                lastQuoteOpened = !lastQuoteOpened;
                contextDepth += lastQuoteOpened ? 1 : -1;
            }
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

        _optionalArgs = new();
        List<string> positionalArgs = new();
        foreach (string word in words)
        {
            int eqIndex = word.IndexOf('=');
            if (eqIndex > 0)
            {
                string name = word[..eqIndex];
                string value = word[(eqIndex + 1)..];
                _optionalArgs[name.ToLower()] = value;
            }
            else
                positionalArgs.Add(word);
        }
        _words = positionalArgs.ToArray();
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

    public bool TryGetIntArg(string name, out int value)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (int.TryParse(strValue, out value))
                return true;
            throw new BadCommandArgumentException(name, typeof(int), -1);
        }
        value = 0;
        return false;
    }

    public bool TryGetFloatArg(string name, out float value)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (float.TryParse(strValue, out value))
                return true;
            throw new BadCommandArgumentException(name, typeof(float), -1);
        }
        value = 0;
        return false;
    }

    public bool TryGetBoolArg(string name, out bool value)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (bool.TryParse(strValue, out value))
                return true;
            throw new BadCommandArgumentException(name, typeof(bool), -1);
        }
        value = false;
        return false;
    }

    public bool TryGetTextArg(string name, out string value)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (strValue.Length >= 2 && strValue[0] == '"' && strValue[^1] == '"')
                value = strValue[1..^1];
            else
                value = strValue;
            return true;
        }
        value = "";
        return false;
    }

    public bool TryGetEnumArg<T>(string name, out T value) where T : struct
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (Enum.TryParse<T>(strValue, true, out value))
                return true;
            throw new BadCommandArgumentException(name, typeof(T), -1);
        }
        value = default!;
        return false;
    }

    public int GetIntArg(string name, int defaultValue)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (int.TryParse(strValue, out int value))
                return value;
            throw new BadCommandArgumentException(name, typeof(int), -1);
        }
        return defaultValue;
    }

    public float GetFloatArg(string name, float defaultValue)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (float.TryParse(strValue, out float value))
                return value;
            throw new BadCommandArgumentException(name, typeof(float), -1);
        }
        return defaultValue;
    }

    public bool GetBoolArg(string name, bool defaultValue)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (bool.TryParse(strValue, out bool value))
                return value;
            throw new BadCommandArgumentException(name, typeof(bool), -1);
        }
        return defaultValue;
    }

    public string GetTextArg(string name, string defaultValue)
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (strValue.Length >= 2 && strValue[0] == '"' && strValue[^1] == '"')
                return strValue[1..^1];
            return strValue;
        }
        return defaultValue;
    }

    public T GetEnumArg<T>(string name, T defaultValue) where T : struct
    {
        if (_optionalArgs.TryGetValue(name.ToLower(), out string? strValue))
        {
            if (Enum.TryParse<T>(strValue, true, out T value))
                return value;
            throw new BadCommandArgumentException(name, typeof(T), -1);
        }
        return defaultValue;
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
            string location = Index >= 0 ? $" at index {Index}" : " (optional argument)";
            return $"expected {Type.Name} argument '{Name}'{location}";
        }
    }
}
