namespace Ramen.UnitTests;

using Ramen.Agents;
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
}
