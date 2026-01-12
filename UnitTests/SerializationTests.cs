using Ramen.Game;

namespace Ramen.UnitTests;

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
        FastRandom random = new(10);
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
}