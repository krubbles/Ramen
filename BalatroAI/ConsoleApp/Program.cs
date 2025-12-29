using BalatroAI;
using BalatroAI.ConsoleApp;

GameData gameData = new();
GameState gameState = new(gameData);
gameState.StartRound();
gameState.JokerState.AddJoker(Jokers.SlyJoker);
while (true)
{

    Console.WriteLine();
    Console.WriteLine("Current Round Chips: " + gameState.ScoringState.CurrentRoundTotalChips);
    Console.WriteLine("Remaining Hands: " + gameState.HandState.RemainingHands);
    Console.WriteLine("Remaining Discards: " + gameState.HandState.RemainingDiscards);
    Console.WriteLine("Hand: " + gameState.HandToString());
    Console.WriteLine();

    ConsoleCommandContext command = new(Console.ReadLine());
    switch (command.Name)
    {
        case "play":
            Span<int> hand = new int[command.NumberOfArguments];
            for (int i = 0; i < command.NumberOfArguments; i++)
                hand[i] = command.GetIntArg(i);
            Console.WriteLine();
            Console.WriteLine("Hand Scored: " + gameState.HandState.PlayHand(hand));
            break;
        case "discard":
            Span<int> discard = new int[command.NumberOfArguments];
            for (int i = 0; i < command.NumberOfArguments; i++)
                discard[i] = command.GetIntArg(i);
            gameState.HandState.DiscardHand(discard);
            break;

    }
}