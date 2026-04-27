namespace Ramen.AI;

public sealed class PpoPolicyAgent : PolicyOnlyAgent
{
    public PpoPolicyAgent(PpoPolicyValueModel model, bool ownsModel = false) : base(model, ownsModel)
    {
    }
}
