namespace BalatroAI;

using System.Text;

public static class LoggingUtility
{
    public static string FormatArray<T>(T[] array, string seperator = ", ")
    {
        StringBuilder sb = new();
        for (int i = 0; i < array.Length; i++)
        {
            sb.Append(array[i]);
            if (i < array.Length - 1) 
                sb.Append(seperator);
        }
        return sb.ToString();
    }

    public static string FormatArray(float[] array, string seperator = ", ")
    {
        StringBuilder sb = new();
        for (int i = 0; i < array.Length; i++)
        {
            sb.Append(array[i].ToString("F3"));
            if (i < array.Length - 1)
                sb.Append(seperator);
        }
        return sb.ToString();
    }

}