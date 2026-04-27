namespace Ramen.ConsoleApp;

using System.Globalization;
using Ramen.Game;

public sealed class ConsoleBalatroApp
{
    const ConsoleColor GeneratedTextColor = ConsoleColor.Gray;
    const ConsoleColor MoneyTextColor = ConsoleColor.Yellow;

    readonly GameState _gameState;
    readonly bool _saveGameOnExit;
    bool _gameSaved;

    public const string TempGameDatabaseName = "ConsoleAppTemp";

    public ConsoleBalatroApp(bool saveGameOnExit = true)
    {
        _gameState = new(new());
        _saveGameOnExit = saveGameOnExit;
    }

    public void Run()
    {
        try
        {
            while (true)
            {
                if (!ApplyAutomaticMoves())
                    return;

                if (_gameState.Stage == StageOfGame.InRoundPlayerChoice)
                {
                    if (_gameState.HandState.RemainingHands == 0 &&
                        _gameState.HandState.RemainingDiscards == 0 &&
                        _gameState.ScoringState.CurrentRoundTotalScore < _gameState.ScoringState.CurrentRoundThresholdScore)
                    {
                        WriteGeneratedLine("Round lost.");
                        SaveGame();
                        return;
                    }

                    if (!PromptForRoundMove())
                        return;
                }
                else if (_gameState.Stage == StageOfGame.InShop)
                {
                    if (!PromptForShopMove())
                        return;
                }
            }
        }
        finally
        {
            SaveGame();
        }
    }

    bool ApplyAutomaticMoves()
    {
        while (_gameState.Stage != StageOfGame.InRoundPlayerChoice &&
            _gameState.Stage != StageOfGame.InShop)
        {
            if (_gameState.Stage == StageOfGame.BeginRound)
            {
                if (GameData.GetAnteForRound(_gameState.Round + 1) >= _gameState.GameData.AnteScoreThresholds.Length)
                {
                    WriteGeneratedLine("No more rounds configured.");
                    return false;
                }

                StartRoundMove startRoundMove = new();
                startRoundMove.Apply(_gameState);
                WriteGeneratedLine($"Round: {_gameState.Round}");

                AfterHandUsedMove drawOpeningHandMove = new();
                drawOpeningHandMove.Apply(_gameState);
            }
            else if (_gameState.Stage == StageOfGame.InRoundAfterHandUsed)
            {
                AfterHandUsedMove afterHandUsedMove = new();
                afterHandUsedMove.Apply(_gameState);
                if (afterHandUsedMove.Cards.Length > 0)
                    WriteCardsLine("Drew cards: ", afterHandUsedMove.Cards);
            }
            else if (_gameState.Stage == StageOfGame.EndRound)
            {
                WriteRoundMoneyGained();
                EndRoundMove endRoundMove = new();
                endRoundMove.Apply(_gameState);
            }
            else if (_gameState.Stage == StageOfGame.EnterShop)
            {
                EnterShopMove enterShopMove = new();
                enterShopMove.Apply(_gameState);
                WriteGeneratedLine();
                WriteGeneratedLine($"Store: #{_gameState.Round}");
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    bool PromptForRoundMove()
    {
        while (_gameState.Stage == StageOfGame.InRoundPlayerChoice)
        {
            WriteRoundStatus();
            while (_gameState.Stage == StageOfGame.InRoundPlayerChoice)
            {
                if (!TryReadInput(out string input))
                    return false;

                string commandName = GetCommandName(input);
                bool moveSucceeded;

                if (commandName == "play")
                    moveSucceeded = HandleUseHand(input, isDiscard: false);
                else if (commandName == "discard")
                    moveSucceeded = HandleUseHand(input, isDiscard: true);
                else
                {
                    WriteGeneratedLine("Unknown command.");
                    moveSucceeded = false;
                }

                if (moveSucceeded)
                    break;
            }
        }

        return true;
    }

    bool PromptForShopMove()
    {
        while (_gameState.Stage == StageOfGame.InShop)
        {
            WriteShopStatus();
            while (_gameState.Stage == StageOfGame.InShop)
            {
                if (!TryReadInput(out string input))
                    return false;

                string commandName = GetCommandName(input);
                bool moveSucceeded;

                if (commandName == "buy")
                    moveSucceeded = HandleBuy(input);
                else if (commandName == "reroll")
                    moveSucceeded = HandleReroll();
                else if (commandName == "exit")
                    moveSucceeded = HandleExitShop();
                else
                {
                    WriteGeneratedLine("Unknown command.");
                    moveSucceeded = false;
                }

                if (moveSucceeded)
                    break;
            }
        }

        return true;
    }

    void WriteRoundStatus()
    {
        WriteGeneratedLine();
        WriteJokers();
        WriteGeneratedLine();
        WriteGeneratedLine($"Current Score: {FormatScore(_gameState.ScoringState.CurrentRoundTotalScore)}, Threshold Score: {FormatScore(_gameState.ScoringState.CurrentRoundThresholdScore)}");
        WriteGenerated("Hands: ");
        WriteGenerated(_gameState.HandState.RemainingHands.ToString(CultureInfo.InvariantCulture));
        WriteGenerated(", Discards: ");
        WriteGenerated(_gameState.HandState.RemainingDiscards.ToString(CultureInfo.InvariantCulture));
        WriteGenerated(", Money: ");
        WriteMoney(_gameState.ShopState.Money);
        WriteGeneratedLine();
        WriteGeneratedLine();
        WriteCardsLine("Hand: ", _gameState.HandState.Hand);
        WriteGeneratedLine();
        WriteGeneratedLine("Use 'play [hand]' to play a hand.");
        if (_gameState.HandState.RemainingDiscards > 0)
            WriteGeneratedLine("Use 'discard [hand]' to discard a hand.");
        WriteGeneratedLine();
    }

    void WriteShopStatus()
    {
        WriteGeneratedLine();
        WriteJokers();
        WriteGeneratedLine();
        WriteGenerated("Money: ");
        WriteMoney(_gameState.ShopState.Money);
        WriteGeneratedLine();
        WriteGeneratedLine();
        WriteGenerated("Current reroll cost: ");
        WriteMoney(_gameState.ShopState.CurrentRerollCost);
        WriteGeneratedLine();
        WriteGeneratedLine();
        WriteGeneratedLine("Shop offerings:");
        for (int offeringIndex = 0; offeringIndex < _gameState.ShopState.ShopOfferings.Count; ++offeringIndex)
        {
            JokerInstance offering = _gameState.ShopState.ShopOfferings[offeringIndex];
            if (offering == null)
                WriteGeneratedLine($"{offeringIndex + 1}. Empty");
            else
            {
                WriteGenerated($"{offeringIndex + 1}. ");
                WriteJokerName(offering.JokerData.Name);
                WriteGenerated(", ");
                WriteMoney(offering.JokerData.BasePrice);
                WriteGeneratedLine();
            }
        }

        WriteGeneratedLine();
        if (HasAffordableShopOffering())
            WriteGeneratedLine("Use 'buy [offer number]' to purchase an offering");
        if (_gameState.ShopState.CurrentRerollCost <= _gameState.ShopState.Money)
            WriteGeneratedLine("Use 'reroll' to reroll the shop");
        WriteGeneratedLine("Use 'exit' to exit the shop");
        WriteGeneratedLine();
    }

    bool HandleUseHand(string input, bool isDiscard)
    {
        string actionName = isDiscard ? "discard" : "play";
        string handText = GetCommandArguments(input);
        if (string.IsNullOrWhiteSpace(handText))
        {
            WriteGeneratedLine($"You must provide a hand when using '{actionName}'. Example: '{actionName} 9C JC'");
            return false;
        }

        string[] cardTexts = handText.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (cardTexts.Length > GameData.MaxPlayedHandSize)
        {
            WriteGeneratedLine($"You cannot {actionName} more then 5 cards");
            return false;
        }

        if (!TryParseCards(cardTexts, out Card[] cards))
            return false;

        if (!TryGetCardIndices(cards, out byte[] cardIndices, out Card missingCard))
        {
            WriteGeneratedLine($"The card '{missingCard}' is not in your hand, so you can't {actionName} it.");
            return false;
        }

        if (isDiscard && _gameState.HandState.RemainingDiscards <= 0)
        {
            WriteGeneratedLine("You have no discards remaining.");
            return false;
        }

        if (!isDiscard && _gameState.HandState.RemainingHands <= 0)
        {
            WriteGeneratedLine("You have no hands remaining.");
            return false;
        }

        float scoreBefore = (float)_gameState.ScoringState.CurrentRoundTotalScore;
        UseHandMove useHandMove = new(isDiscard, cardIndices);
        useHandMove.Apply(_gameState);

        if (!isDiscard)
        {
            float handScore = (float)_gameState.ScoringState.CurrentRoundTotalScore - scoreBefore;
            WriteGeneratedLine($"Hand scored: {FormatScore(handScore)}");
            if (_gameState.Stage == StageOfGame.EndRound)
                WriteGeneratedLine("Round won!");
        }

        return true;
    }

    bool HandleBuy(string input)
    {
        string offerText = GetCommandArguments(input);
        if (!int.TryParse(offerText, NumberStyles.None, CultureInfo.InvariantCulture, out int offerNumber))
        {
            WriteGeneratedLine("Invalid offer number.");
            return false;
        }

        int offerIndex = offerNumber - 1;
        if (offerIndex < 0 || offerIndex >= _gameState.ShopState.ShopOfferings.Count)
        {
            WriteGeneratedLine("Invalid offer number.");
            return false;
        }

        JokerInstance offering = _gameState.ShopState.ShopOfferings[offerIndex];
        if (offering == null)
        {
            WriteGeneratedLine("That shop slot is empty.");
            return false;
        }

        if (_gameState.ShopState.GetShopOfferingPrice(offerIndex) > _gameState.ShopState.Money)
        {
            WriteGeneratedLine("You can't afford that offering.");
            return false;
        }

        string jokerName = offering.JokerData.Name;
        BuyShopOfferMove buyShopOfferMove = new(offerIndex);
        buyShopOfferMove.Apply(_gameState);
        WriteGenerated("Purchased: ");
        WriteJokerName(jokerName);
        WriteGeneratedLine();
        return true;
    }

    bool HandleReroll()
    {
        if (_gameState.ShopState.CurrentRerollCost > _gameState.ShopState.Money)
        {
            WriteGeneratedLine("You can't afford to reroll the shop.");
            return false;
        }

        RerollMove rerollMove = new();
        rerollMove.Apply(_gameState);
        WriteGeneratedLine("Rerolled!");
        return true;
    }

    bool HandleExitShop()
    {
        ExitShopMove exitShopMove = new();
        exitShopMove.Apply(_gameState);
        WriteGeneratedLine("Leaving shop.");
        return true;
    }

    bool TryParseCards(string[] cardTexts, out Card[] cards)
    {
        cards = new Card[cardTexts.Length];
        for (int cardIndex = 0; cardIndex < cardTexts.Length; ++cardIndex)
        {
            string cardText = cardTexts[cardIndex].ToUpperInvariant();
            if (cardText.Length != 2)
            {
                WriteInvalidCardMessage(cardTexts[cardIndex]);
                return false;
            }

            try
            {
                cards[cardIndex] = Card.Parse(cardText);
            }
            catch (NotSupportedException)
            {
                WriteInvalidCardMessage(cardTexts[cardIndex]);
                return false;
            }
            catch (FormatException)
            {
                WriteInvalidCardMessage(cardTexts[cardIndex]);
                return false;
            }
        }

        return true;
    }

    bool TryGetCardIndices(ReadOnlySpan<Card> cards, out byte[] cardIndices, out Card missingCard)
    {
        ReadOnlySpan<Card> hand = _gameState.HandState.Hand;
        bool[] usedHandCards = new bool[hand.Length];
        cardIndices = new byte[cards.Length];
        missingCard = Card.Null;

        for (int selectedCardIndex = 0; selectedCardIndex < cards.Length; ++selectedCardIndex)
        {
            bool found = false;
            for (int handCardIndex = 0; handCardIndex < hand.Length; ++handCardIndex)
            {
                if (!usedHandCards[handCardIndex] && hand[handCardIndex] == cards[selectedCardIndex])
                {
                    usedHandCards[handCardIndex] = true;
                    cardIndices[selectedCardIndex] = (byte)handCardIndex;
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                missingCard = cards[selectedCardIndex];
                return false;
            }
        }

        Array.Sort(cardIndices);
        return true;
    }

    bool HasAffordableShopOffering()
    {
        for (int offeringIndex = 0; offeringIndex < _gameState.ShopState.ShopOfferings.Count; ++offeringIndex)
        {
            if (_gameState.ShopState.ShopOfferings[offeringIndex] != null &&
                _gameState.ShopState.GetShopOfferingPrice(offeringIndex) <= _gameState.ShopState.Money)
            {
                return true;
            }
        }

        return false;
    }

    void WriteRoundMoneyGained()
    {
        int blindRewardMoney = _gameState.GameData.GetRewardMoney(_gameState.Round);
        int remainingHandMoney = _gameState.HandState.RemainingHands;
        int interestMoney = Math.Min(_gameState.ShopState.Money / 5, 5);
        int totalMoney = blindRewardMoney + remainingHandMoney + interestMoney;

        WriteGeneratedLine();
        WriteGenerated("Money gained: ");
        WriteMoney(totalMoney);
        WriteGeneratedLine();
        WriteGenerated("Blind reward: ");
        WriteMoney(blindRewardMoney);
        WriteGeneratedLine();
        WriteGenerated("Remaining hands: ");
        WriteMoney(remainingHandMoney);
        WriteGeneratedLine();
        WriteGenerated("Interest: ");
        WriteMoney(interestMoney);
        WriteGeneratedLine();
    }

    void WriteJokers()
    {
        WriteGeneratedLine("Jokers:");
        if (_gameState.JokerState.Jokers.Count == 0)
        {
            WriteGeneratedLine("- None.");
            return;
        }

        for (int jokerIndex = 0; jokerIndex < _gameState.JokerState.Jokers.Count; ++jokerIndex)
        {
            JokerInstance joker = _gameState.JokerState.Jokers[jokerIndex];
            WriteGenerated($"{jokerIndex + 1}. ");
            WriteJokerName(joker.JokerData.Name);
            WriteGeneratedLine();
        }
    }

    void WriteCardsLine(string prefix, ReadOnlySpan<Card> cards)
    {
        WriteGenerated(prefix);
        for (int cardIndex = 0; cardIndex < cards.Length; ++cardIndex)
        {
            if (cardIndex > 0)
                WriteGenerated(" ");
            WriteCard(cards[cardIndex]);
        }
        WriteGeneratedLine();
    }

    void WriteCard(Card card)
    {
        Console.ForegroundColor = GetSuitColor(card.Suit);
        Console.Write(card.ToString());
        Console.ForegroundColor = GeneratedTextColor;
    }

    void WriteInvalidCardMessage(string cardText)
    {
        WriteGeneratedLine($"Invalid card '{cardText}'. Cards should use the format [rank][suit], for example 4H, TC.");
    }

    void SaveGame()
    {
        if (!_saveGameOnExit || _gameSaved)
            return;

        GameDatabase gameDatabase = new(TempGameDatabaseName);
        gameDatabase.AddGame(_gameState);
        _gameSaved = true;
    }

    static bool TryReadInput(out string input)
    {
        if (Console.IsInputRedirected && Console.In.Peek() < 0)
        {
            input = "";
            return false;
        }

        WriteGenerated("> ");
        Console.ResetColor();
        input = Console.ReadLine();
        if (input == null)
            return false;

        if (Console.IsInputRedirected && input.Length == 0 && Console.In.Peek() < 0)
            return false;

        return true;
    }

    static string GetCommandName(string input)
    {
        string trimmedInput = input.Trim();
        int spaceIndex = trimmedInput.IndexOf(' ');
        if (spaceIndex < 0)
            return trimmedInput.ToLowerInvariant();
        return trimmedInput[..spaceIndex].ToLowerInvariant();
    }

    static string GetCommandArguments(string input)
    {
        string trimmedInput = input.Trim();
        int spaceIndex = trimmedInput.IndexOf(' ');
        if (spaceIndex < 0)
            return "";
        return trimmedInput[(spaceIndex + 1)..].Trim();
    }

    static string FormatScore(float score)
    {
        return score.ToString("0", CultureInfo.InvariantCulture);
    }

    static string FormatScore(double score)
    {
        return score.ToString("0", CultureInfo.InvariantCulture);
    }

    static string FormatMoney(int money)
    {
        return money.ToString("$0", CultureInfo.InvariantCulture);
    }

    static ConsoleColor GetSuitColor(Suit suit)
    {
        return suit switch
        {
            Suit.Heart => ConsoleColor.Red,
            Suit.Diamond => ConsoleColor.Magenta,
            Suit.Club => ConsoleColor.Green,
            Suit.Spade => ConsoleColor.Cyan,
            _ => ConsoleColor.Gray
        };
    }

    static void WriteMoney(int money)
    {
        Console.ForegroundColor = MoneyTextColor;
        Console.Write(FormatMoney(money));
        Console.ForegroundColor = GeneratedTextColor;
    }

    static void WriteJokerName(string jokerName)
    {
        Console.ResetColor();
        Console.Write(jokerName);
        Console.ForegroundColor = GeneratedTextColor;
    }

    static void WriteGenerated(string text)
    {
        Console.ForegroundColor = GeneratedTextColor;
        Console.Write(text);
    }

    static void WriteGeneratedLine()
    {
        Console.ForegroundColor = GeneratedTextColor;
        Console.WriteLine();
        Console.ResetColor();
    }

    static void WriteGeneratedLine(string text)
    {
        Console.ForegroundColor = GeneratedTextColor;
        Console.WriteLine(text);
        Console.ResetColor();
    }
}
