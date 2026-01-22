namespace Ramen.UnitTests;

using Ramen.Game;

public class GameDatabaseManagementTests
{
    const string UtPrefix = "UT-";

    GameState CreateTestGame(int seed)
    {
        FastRandom random = new(seed);
        GameState gameState = new(GameData.Default);
        gameState.PlayRandomGame(random);
        return gameState;
    }

    void CreateTestDatabase(string name, int gameCount, int randomSeed = 1)
    {
        GameDatabase db = new(name, load: false, delete: true);
        FastRandom random = new(randomSeed);
        for (int i = 0; i < gameCount; i++)
        {
            GameState gameState = new(GameData.Default);
            gameState.PlayRandomGame(random);
            db.AddGame(gameState);
        }
    }

    int CountGamesInDatabase(string name)
    {
        GameDatabase db = new(name, load: true);
        int count = 0;
        foreach (GameState game in db)
            count++;
        return count;
    }

    [Test]
    public void DeleteSingleDatabase()
    {
        string dbName = UtPrefix + "DeleteTest";
        CreateTestDatabase(dbName, 10);
        
        string dbPath = GameDatabase.GetGameDatabasePath(dbName);
        Assert.That(File.Exists(dbPath), Is.True, "Database should exist before deletion");
        
        File.Delete(dbPath);
        Assert.That(File.Exists(dbPath), Is.False, "Database should not exist after deletion");
    }

    [Test]
    public void DeleteMultipleDatabasesWithRangePattern()
    {
        string baseName = UtPrefix + "RangeDelete";
        
        for (int i = 0; i <= 9; i++)
            CreateTestDatabase(baseName + i, 5);
        
        for (int i = 0; i <= 9; i++)
        {
            string dbPath = GameDatabase.GetGameDatabasePath(baseName + i);
            Assert.That(File.Exists(dbPath), Is.True, $"Database {baseName}{i} should exist before deletion");
        }
        
        for (int i = 0; i <= 9; i++)
        {
            string dbPath = GameDatabase.GetGameDatabasePath(baseName + i);
            File.Delete(dbPath);
        }
        
        for (int i = 0; i <= 9; i++)
        {
            string dbPath = GameDatabase.GetGameDatabasePath(baseName + i);
            Assert.That(File.Exists(dbPath), Is.False, $"Database {baseName}{i} should not exist after deletion");
        }
    }

    [Test]
    public void CombineTwoDatabases()
    {
        string db1 = UtPrefix + "Combine1";
        string db2 = UtPrefix + "Combine2";
        string target = UtPrefix + "CombineTarget";
        
        CreateTestDatabase(db1, 10, 1);
        CreateTestDatabase(db2, 15, 2);
        
        GameDatabase targetDb = new(target, load: false, delete: true);
        
        GameDatabase source1 = new(db1, load: true);
        foreach (GameState game in source1)
            targetDb.AddGame(game);
        
        GameDatabase source2 = new(db2, load: true);
        foreach (GameState game in source2)
            targetDb.AddGame(game);
        
        int totalGames = CountGamesInDatabase(target);
        Assert.That(totalGames, Is.EqualTo(25), "Combined database should have 25 games");
        
        File.Delete(GameDatabase.GetGameDatabasePath(db1));
        File.Delete(GameDatabase.GetGameDatabasePath(db2));
        File.Delete(GameDatabase.GetGameDatabasePath(target));
    }

    [Test]
    public void CombineMultipleDatabases()
    {
        List<string> sourceDbs = new();
        for (int i = 0; i < 5; i++)
        {
            string dbName = UtPrefix + "MultiCombine" + i;
            CreateTestDatabase(dbName, 10 + (i * 5), i + 1);
            sourceDbs.Add(dbName);
        }
        
        string target = UtPrefix + "MultiCombineTarget";
        GameDatabase targetDb = new(target, load: false, delete: true);
        
        int expectedTotal = 0;
        foreach (string sourceDb in sourceDbs)
        {
            GameDatabase source = new(sourceDb, load: true);
            int gamesFromSource = 0;
            foreach (GameState game in source)
            {
                targetDb.AddGame(game);
                gamesFromSource++;
            }
            expectedTotal += gamesFromSource;
        }
        
        int totalGames = CountGamesInDatabase(target);
        Assert.That(totalGames, Is.EqualTo(expectedTotal), $"Combined database should have {expectedTotal} games");
        
        foreach (string sourceDb in sourceDbs)
            File.Delete(GameDatabase.GetGameDatabasePath(sourceDb));
        File.Delete(GameDatabase.GetGameDatabasePath(target));
    }

    [Test]
    public void CombinePreservesGameData()
    {
        string db1 = UtPrefix + "PreserveTest1";
        string db2 = UtPrefix + "PreserveTest2";
        string target = UtPrefix + "PreserveTarget";
        
        List<GameState> originalGames = new();
        FastRandom random = new(42);
        
        GameDatabase source1Db = new(db1, load: false, delete: true);
        for (int i = 0; i < 10; i++)
        {
            GameState game = new(GameData.Default);
            game.PlayRandomGame(random);
            source1Db.AddGame(game);
            originalGames.Add(game);
        }
        
        GameDatabase source2Db = new(db2, load: false, delete: true);
        for (int i = 0; i < 10; i++)
        {
            GameState game = new(GameData.Default);
            game.PlayRandomGame(random);
            source2Db.AddGame(game);
            originalGames.Add(game);
        }
        
        GameDatabase targetDb = new(target, load: false, delete: true);
        GameDatabase loadSource1 = new(db1, load: true);
        foreach (GameState game in loadSource1)
            targetDb.AddGame(game);
        
        GameDatabase loadSource2 = new(db2, load: true);
        foreach (GameState game in loadSource2)
            targetDb.AddGame(game);
        
        List<GameState> combinedGames = new();
        GameDatabase loadTarget = new(target, load: true);
        foreach (GameState game in loadTarget)
            combinedGames.Add(game);
        
        Assert.That(combinedGames, Has.Count.EqualTo(20), "Combined database should have 20 games");
        
        for (int i = 0; i < originalGames.Count && i < combinedGames.Count; i++)
        {
            Assert.That(combinedGames[i].GetHashCode(), Is.EqualTo(originalGames[i].GetHashCode()),
                $"Game {i} should match after combining");
        }
        
        File.Delete(GameDatabase.GetGameDatabasePath(db1));
        File.Delete(GameDatabase.GetGameDatabasePath(db2));
        File.Delete(GameDatabase.GetGameDatabasePath(target));
    }

    [Test]
    public void GetGameDatabasePathReturnsCorrectPath()
    {
        string dbName = "TestDatabase";
        string path = GameDatabase.GetGameDatabasePath(dbName);
        
        Assert.That(path, Does.EndWith("TestDatabase.bin"), "Path should end with database name and .bin extension");
        Assert.That(path, Does.Contain("Ramen"), "Path should contain Ramen folder");
        Assert.That(path, Does.Contain("GameDatabases"), "Path should contain GameDatabases folder");
    }

    [Test]
    public void GetGameDatabasesFolderReturnsValidPath()
    {
        string folderPath = GameDatabase.GetGameDatabasesFolder();
        
        Assert.That(folderPath, Does.Contain("Ramen"), "Folder path should contain Ramen");
        Assert.That(folderPath, Does.Contain("GameDatabases"), "Folder path should contain GameDatabases");
        Assert.That(folderPath, Does.Not.EndWith(".bin"), "Folder path should not have .bin extension");
    }
}
