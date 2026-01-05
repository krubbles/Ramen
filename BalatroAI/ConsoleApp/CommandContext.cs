namespace Ramen.ConsoleApp;

using System.Text;
using Ramen.Game;

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

    public int GetIntArg(int index)
    {
        return int.Parse(_words[index + 1]);
    }

    public float GetFloatArg(int index)
    {
        return float.Parse(_words[index + 1]);
    }

    public bool GetBoolArg(int index)
    {
        return bool.Parse(_words[index + 1]);
    }

    public string GetTextArg(int index)
    {
        string text = _words[index + 1];
        if (text.Length >= 2 && text[0] == '"' && text[^1] == '"')
            return text[1..^1];
        return _words[index + 1];
    }

    public T GetEnumArg<T>(int index) where T : struct
    {
        return Enum.Parse<T>(_words[index + 1], true);
    }

    public Card GetCardArg(int index)
    {
        return Card.Parse(_words[index + 1]);
    }
}
