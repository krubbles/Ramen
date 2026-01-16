namespace Ramen.UnitTests;

using Ramen.Game;

public class AnnotatingDataMoveTests
{
    [Test]
    public void AnnotatingDataMoveEmptyDataRoundTrip()
    {
        GameState originalGameState = new(GameData.Default);
        int initialMoveCount = originalGameState.MoveState.MoveHistory.Count;
        AnnotatingDataMove originalMove = new(Array.Empty<byte>());
        originalMove.Apply(originalGameState);

        using MemoryStream stream = new();
        originalGameState.Serialize(stream);

        using MemoryStream readStream = new(stream.ToArray(), false);
        GameState deserializedGameState = new(GameData.Default);
        deserializedGameState.Deserialize(readStream);

        Assert.That(deserializedGameState.MoveState.MoveHistory.Count, Is.EqualTo(initialMoveCount + 1), "Should have one additional move in history");
        Assert.That(deserializedGameState.MoveState.MoveHistory[^1], Is.TypeOf<AnnotatingDataMove>(), "Last move should be AnnotatingDataMove");
        var annotatingMove = (AnnotatingDataMove)deserializedGameState.MoveState.MoveHistory[^1];
        Assert.That(annotatingMove.Data.Length, Is.EqualTo(0), "Deserialized data should be empty");
    }

    [Test]
    public void AnnotatingDataMoveRoundTrip()
    {
        GameState originalGameState = new(GameData.Default);
        int initialMoveCount = originalGameState.MoveState.MoveHistory.Count;
        byte[] testData = [10, 20, 30, 40, 50];
        AnnotatingDataMove originalMove = new(testData);
        originalMove.Apply(originalGameState);

        using MemoryStream stream = new();
        originalGameState.Serialize(stream);

        using MemoryStream readStream = new(stream.ToArray(), false);
        GameState deserializedGameState = new(GameData.Default);
        deserializedGameState.Deserialize(readStream);

        Assert.That(deserializedGameState.MoveState.MoveHistory.Count, Is.EqualTo(initialMoveCount + 1), "Should have one additional move in history");
        Assert.That(deserializedGameState.MoveState.MoveHistory[^1], Is.TypeOf<AnnotatingDataMove>(), "Last move should be AnnotatingDataMove");
        var annotatingMove = (AnnotatingDataMove)deserializedGameState.MoveState.MoveHistory[^1];
        Assert.That(annotatingMove.Data.Length, Is.EqualTo(5), "Deserialized data should have correct length");
        Assert.That(annotatingMove.Data, Is.EqualTo(testData), "Deserialized data should match original");
    }
}