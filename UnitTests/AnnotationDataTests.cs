namespace Ramen.UnitTests;

using Ramen.AgentTools;
using Ramen.Game;

public class AnnotationDataTests
{
    [Test]
    public void PolicyAnnotationHelpersRoundTrip()
    {
        float[] policy = [0.1f, 0.2f, 0.7f];

        AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation(policy);
        float[] decodedPolicy = AnnotationDataUtils.DecodePolicyAnnotation(annotation);

        Assert.That(annotation.DataTypeID, Is.EqualTo((ushort)AnnotationDataType.Policy));
        Assert.That(decodedPolicy.Length, Is.EqualTo(policy.Length));

        for (int i = 0; i < policy.Length; i++)
        {
            float expected = AnnotatingDataMove.DecodeProb(AnnotatingDataMove.EncodeProb(policy[i]));
            Assert.That(decodedPolicy[i], Is.EqualTo(expected).Within(1e-6f));
        }
    }

    [Test]
    public void PolicyAnnotationSerializationPreservesDataType()
    {
        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();

        Move move = gameState.GetMoveOptions()[0];
        move.Apply(gameState);

        AnnotatingDataMove annotation = AnnotationDataUtils.CreatePolicyAnnotation([0.25f, 0.75f]);
        annotation.Apply(gameState);

        using MemoryStream stream = new();
        gameState.Serialize(stream);

        GameState deserializedGameState = new(GameData.Default);
        using MemoryStream readStream = new(stream.ToArray(), writable: false);
        deserializedGameState.Deserialize(readStream);

        AnnotatingDataMove deserializedAnnotation = deserializedGameState.MoveState.MoveHistory[^1] as AnnotatingDataMove;
        Assert.That(deserializedAnnotation, Is.Not.Null);
        Assert.That(deserializedAnnotation.DataTypeID, Is.EqualTo((ushort)AnnotationDataType.Policy));

        float[] decodedPolicy = AnnotationDataUtils.DecodePolicyAnnotation(deserializedAnnotation);
        Assert.That(decodedPolicy.Length, Is.EqualTo(2));
        Assert.That(decodedPolicy[0], Is.EqualTo(AnnotatingDataMove.DecodeProb(AnnotatingDataMove.EncodeProb(0.25f))).Within(1e-6f));
        Assert.That(decodedPolicy[1], Is.EqualTo(AnnotatingDataMove.DecodeProb(AnnotatingDataMove.EncodeProb(0.75f))).Within(1e-6f));
    }

    [Test]
    public void MoveRewardsAnnotationHelpersRoundTrip()
    {
        float[] moveRewards = [-1.5f, 0.25f, 2.75f, 5.5f];

        AnnotatingDataMove annotation = AnnotationDataUtils.CreateMoveRewardsAnnotation(moveRewards);
        float[] decodedRewards = AnnotationDataUtils.DecodeMoveRewardsAnnotation(annotation);

        Assert.That(annotation.DataTypeID, Is.EqualTo((ushort)AnnotationDataType.MoveRewards));
        Assert.That(decodedRewards.Length, Is.EqualTo(moveRewards.Length));

        float tolerance = (moveRewards.Max() - moveRewards.Min()) / ushort.MaxValue + 1e-6f;
        for (int i = 0; i < moveRewards.Length; ++i)
            Assert.That(decodedRewards[i], Is.EqualTo(moveRewards[i]).Within(tolerance));
    }

    [Test]
    public void MoveRewardsAnnotationSerializationPreservesDataType()
    {
        GameState gameState = new(GameData.Default);
        gameState.AdvanceToNextPlayerChoice();

        Move move = gameState.GetMoveOptions()[0];
        move.Apply(gameState);

        float[] moveRewards = [-2f, 1.5f, 4.25f];
        AnnotatingDataMove annotation = AnnotationDataUtils.CreateMoveRewardsAnnotation(moveRewards);
        annotation.Apply(gameState);

        using MemoryStream stream = new();
        gameState.Serialize(stream);

        GameState deserializedGameState = new(GameData.Default);
        using MemoryStream readStream = new(stream.ToArray(), writable: false);
        deserializedGameState.Deserialize(readStream);

        AnnotatingDataMove deserializedAnnotation = deserializedGameState.MoveState.MoveHistory[^1] as AnnotatingDataMove;
        Assert.That(deserializedAnnotation, Is.Not.Null);
        Assert.That(deserializedAnnotation.DataTypeID, Is.EqualTo((ushort)AnnotationDataType.MoveRewards));

        float[] decodedRewards = AnnotationDataUtils.DecodeMoveRewardsAnnotation(deserializedAnnotation);
        Assert.That(decodedRewards.Length, Is.EqualTo(moveRewards.Length));

        float tolerance = (moveRewards.Max() - moveRewards.Min()) / ushort.MaxValue + 1e-6f;
        for (int i = 0; i < moveRewards.Length; ++i)
            Assert.That(decodedRewards[i], Is.EqualTo(moveRewards[i]).Within(tolerance));
    }
}
