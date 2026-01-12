namespace Ramen.UnitTests;

using Ramen.Game;

public class SerializationTests
{
    [Test]
    public void SerializeDoesSomething()
    {
        FastRandom random = new(10);
        for (int i = 0; i < 5; ++i)
        {
            GameState gameState = new(GameData.Default);
            gameState.PlayRandomGame(random);
            using MemoryStream stream = new();
            Assert.That(() =>
            {
                gameState.Serialize(stream);
            }, Throws.Nothing, "Serialization threw an exception.");
            Assert.That(stream.Length, Is.GreaterThan(0), "Serialization produced an empty byte buffer.");
        }
    }

    [Test]
    public void SerializationWorks()
    {
        FastRandom random = new(8);
        for (int i = 0; i < 5; ++i)
        {
            GameState gameState = new(GameData.Default);
            gameState.PlayRandomGame(random);

            using MemoryStream stream = new();
            gameState.Serialize(stream);

            using MemoryStream read = new(stream.ToArray(), false);
            GameState deserializedGameState = new(GameData.Default);
            deserializedGameState.Deserialize(read);
            Assert.That(deserializedGameState.GetHashCode(), Is.EqualTo(gameState.GetHashCode()), "GameState hash code before and after serialization don't match.");
        }
    }

    [Test]
    public void GameDatabaseWorks()
    {
        GameDatabase gdb = new("Test", false, true);
        FastRandom random = new(7);
        for (int i = 0; i < 10; ++i)
        {
            GameState gameState = new(GameData.Default);
            gameState.PlayRandomGame(random);

            gdb.AddGame(gameState);
        }
        List<GameState> games = new();
        foreach (GameState game in gdb)
            games.Add(game);
        Assert.That(games, Has.Count.EqualTo(10), "Loading from database didn't return same number of games that were saved.");
    }
}