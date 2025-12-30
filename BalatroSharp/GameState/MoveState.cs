namespace BalatroAI;

public sealed class MoveState
{
    public readonly GameState GameState;
    readonly GameData _gameData;

    public readonly List<IMove> MoveHistory = new();

    public MoveState(GameState gameState)
    {
        GameState = gameState;
        _gameData = gameState.GameData;
    }

    internal void CloneFrom(MoveState other)
    {
        MoveHistory.Clear();
        MoveHistory.AddRange(other.MoveHistory);
    }

    public List<IMove> GetValidMoves()
    {
        return null;
    }

    public void MakeMove(IMove move)
    {
        move.Apply(GameState);
        MoveHistory.Add(move);
    }

}