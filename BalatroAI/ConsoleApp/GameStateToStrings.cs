namespace BalatroAI.ConsoleApp;
using System.Text;

public static class GameStateToStringExtentions
{
    public static string HandToString(this GameState gameState)
    {
        StringBuilder sb = new();
        foreach (Card card in gameState.HandState.Hand)
        {
            sb.Append(card.ToString());
            sb.Append(' ');
        }
        return sb.ToString();
    }
}