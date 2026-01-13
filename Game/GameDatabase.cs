namespace Ramen.Game;

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class GameDatabase : IEnumerable<GameState>
{
    readonly string _filePath;
    int _activeEnumerators = 0;

    public GameDatabase(string name, bool load = true, bool delete = false)
    {
        string folderPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "Ramen", "GameDatabases"
        );

        Directory.CreateDirectory(folderPath);

        _filePath = Path.Combine(folderPath, $"{name}.bin");

        if (!load && File.Exists(_filePath))
        {
            if (delete)
                File.Delete(_filePath);
            else throw new ArgumentException($"Trying to create a game database where one already exists");
        }
    }

    public void AddGame(GameState gameState)
    {
        if (_activeEnumerators > 0)
        {
            throw new InvalidOperationException("Cannot add a game while the database is being enumerated.");
        }

        using FileStream file = new(_filePath, FileMode.Append, FileAccess.Write, FileShare.Read);
        using BufferedStream buffered = new(file);
        gameState.Serialize(buffered);
    }

    public IEnumerator<GameState> GetEnumerator()
    {
        _activeEnumerators++;

        try
        {
            if (!File.Exists(_filePath))
                yield break;
            using FileStream file = new(_filePath, FileMode.Open, FileAccess.Read, FileShare.None);

            while (file.Position < file.Length)
            {
                GameState game = new(GameData.Default);
                game.Deserialize(file);
                yield return game;  
            }

        }
        finally
        {
            _activeEnumerators--;
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}