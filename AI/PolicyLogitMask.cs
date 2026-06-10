namespace Ramen.AI;

public static class PolicyLogitMask
{
    public const float IllegalMoveLogit = -50000f;

    public static Tensor Apply(GameStateTensors gameStateTensors, Tensor logits)
    {
        using var dScope = NewDisposeScope();

        long batchSize = logits.size(0);
        long moveCount = logits.size(1);
        Tensor remainingHands = gameStateTensors.RemainingHands.to(logits.device).to_type(ScalarType.Int64);
        Tensor remainingDiscards = gameStateTensors.RemainingDiscards.to(logits.device).to_type(ScalarType.Int64);
        Tensor legalCaps = ones([batchSize, 1], dtype: logits.dtype, device: logits.device) * float.PositiveInfinity;
        Tensor illegalCaps = ones([batchSize, 1], dtype: logits.dtype, device: logits.device) * IllegalMoveLogit;
        Tensor playCaps = where(remainingHands.gt(0).unsqueeze(1), legalCaps, illegalCaps);
        Tensor discardCaps = where(remainingDiscards.gt(0).unsqueeze(1), legalCaps, illegalCaps);
        Tensor actionCaps = cat([playCaps, discardCaps], dim: 1);
        Tensor logitCaps = actionCaps.repeat([1, moveCount / 2]);
        Tensor cappedLogits = min(logits, logitCaps);
        Tensor maskedLogits = where(logitCaps.eq(IllegalMoveLogit), illegalCaps.expand(batchSize, moveCount), cappedLogits);

        maskedLogits.MoveToOuterDisposeScope();
        return maskedLogits;
    }

    public static Tensor Apply(GameStateTensors gameStateTensors, Tensor logits, Tensor moveIndices)
    {
        using var dScope = NewDisposeScope();

        long batchSize = logits.size(0);
        long moveCount = logits.size(1);
        Tensor selectedMoveIndices = moveIndices.to(logits.device).to_type(ScalarType.Int64);
        Tensor actionIndices = selectedMoveIndices.remainder(2);
        Tensor canPlay = gameStateTensors.RemainingHands.to(logits.device).to_type(ScalarType.Int64).gt(0).unsqueeze(1);
        Tensor canDiscard = gameStateTensors.RemainingDiscards.to(logits.device).to_type(ScalarType.Int64).gt(0).unsqueeze(1);
        Tensor legalMask = where(actionIndices.eq(1), canDiscard.expand(batchSize, moveCount), canPlay.expand(batchSize, moveCount));
        Tensor legalCaps = ones([batchSize, moveCount], dtype: logits.dtype, device: logits.device) * float.PositiveInfinity;
        Tensor illegalCaps = ones([batchSize, moveCount], dtype: logits.dtype, device: logits.device) * IllegalMoveLogit;
        Tensor logitCaps = where(legalMask, legalCaps, illegalCaps);
        Tensor cappedLogits = min(logits, logitCaps);
        Tensor maskedLogits = where(legalMask, cappedLogits, illegalCaps);

        maskedLogits.MoveToOuterDisposeScope();
        return maskedLogits;
    }
}
