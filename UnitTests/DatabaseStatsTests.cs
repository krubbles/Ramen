namespace Ramen.UnitTests;

using Ramen.AI;
using System;
using System.IO;

public class DatabaseStatsTests
{
    [Test]
    public void GameDatabaseStatsFor833()
    {
        string dbName = "8-3-3";
        string filePath = GetDatabaseFilePath(dbName);
        if (!File.Exists(filePath))
            Assert.Inconclusive($"Database '{dbName}' not found at {filePath}");

        Testing.GameDatabaseStatistics stats = Testing.GetGameDatabaseStatistics(dbName);

        TestContext.Out.WriteLine($"Database '{dbName}' games: {stats.TotalGames}");
        TestContext.Out.WriteLine($"Played straight: {stats.PlayedStraightPercent:P2}");
        TestContext.Out.WriteLine($"Played flush: {stats.PlayedFlushPercent:P2}");
        TestContext.Out.WriteLine($"Played full house: {stats.PlayedFullHousePercent:P2}");
        TestContext.Out.WriteLine($"Discard same suit: {stats.DiscardSameSuitPercent:P2}");
        TestContext.Out.WriteLine($"Discard rank range <= 4: {stats.DiscardRankRangePercent:P2}");

        Assert.That(stats.TotalGames, Is.GreaterThan(0), "Database should contain games.");
        Assert.That(stats.PlayedStraightPercent, Is.GreaterThan(0.01f), "Played straight percent should be > 1%.");
        Assert.That(stats.PlayedFlushPercent, Is.GreaterThan(0.01f), "Played flush percent should be > 1%.");
        Assert.That(stats.PlayedFullHousePercent, Is.GreaterThan(0.01f), "Played full house percent should be > 1%.");
        Assert.That(stats.DiscardSameSuitPercent, Is.GreaterThan(0.01f), "Discard same suit percent should be > 1%.");
        Assert.That(stats.DiscardRankRangePercent, Is.GreaterThan(0.01f), "Discard rank range percent should be > 1%.");
    }

    static string GetDatabaseFilePath(string databaseName)
    {
        string folderPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            "Ramen",
            "GameDatabases"
        );

        return Path.Combine(folderPath, $"{databaseName}.bin");
    }
}
