namespace Ramen.AI;

using System.Globalization;
using System.Text;

/// <summary>
/// Builds CSV content with named columns and rows.
/// </summary>
public sealed class CSVBuilder
{
    readonly List<string> _columns = [];
    readonly Dictionary<string, int> _columnLookup = [];
    readonly List<string[]> _rows = [];
    string[] _currentRow;

    /// <summary>
    /// Starts a new row.
    /// </summary>
    public CSVBuilder NextRow()
    {
        _currentRow = new string[_columns.Count];
        _rows.Add(_currentRow);
        return this;
    }

    /// <summary>
    /// Adds a named column before any rows are written.
    /// </summary>
    public CSVBuilder AddColumn(string colName)
    {
        if (_columnLookup.ContainsKey(colName))
            return this;

        int columnIndex = _columns.Count;
        _columns.Add(colName);
        _columnLookup[colName] = columnIndex;
        for (int i = 0; i < _rows.Count; i++)
        {
            string[] row = _rows[i];
            Array.Resize(ref row, _columns.Count);
            _rows[i] = row;
        }

        if (_currentRow != null)
            _currentRow = _rows[^1];

        return this;
    }

    /// <summary>
    /// Sets the value for the named column in the current row.
    /// </summary>
    public CSVBuilder SetCell<T>(string colName, T value)
    {
        if (!_columnLookup.TryGetValue(colName, out int columnIndex))
        {
            AddColumn(colName);
            columnIndex = _columnLookup[colName];
        }

        if (_currentRow == null)
            NextRow();

        if (value == null)
        {
            _currentRow[columnIndex] = string.Empty;
            return this;
        }

        if (value is float floatValue)
        {
            _currentRow[columnIndex] = floatValue.ToString("F4", CultureInfo.InvariantCulture);
            return this;
        }

        _currentRow[columnIndex] = value.ToString();
        return this;
    }

    /// <summary>
    /// Returns the CSV content for the current data.
    /// </summary>
    public override string ToString()
    {
        StringBuilder sb = new();
        AppendRow(sb, [.. _columns]);
        for (int i = 0; i < _rows.Count; i++)
            AppendRow(sb, _rows[i]);
        return sb.ToString();
    }

    static void AppendRow(StringBuilder sb, string[] cells)
    {
        for (int i = 0; i < cells.Length; i++)
        {
            if (i > 0)
                sb.Append(',');

            sb.Append(EscapeCell(cells[i]));
        }

        sb.AppendLine();
    }

    static string EscapeCell(string value)
    {
        if (string.IsNullOrEmpty(value))
            return string.Empty;

        bool requiresQuotes = value.IndexOfAny([',', '"', '\n', '\r']) >= 0;
        if (!requiresQuotes)
            return value;

        StringBuilder sb = new();
        sb.Append('"');
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            if (c == '"')
                sb.Append('"');
            else
                sb.Append(c);
        }

        sb.Append('"');
        return sb.ToString();
    }
}
