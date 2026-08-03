namespace Ramen.Game;

public enum TarotCardType
{
    Null,
    Fool, // create a copy of last used tarot card
    Magician, // turn 2 cards into lucky cards
    HighPriestess, // create 2 random planet cards
    Empress, // turn 2 cards into mult cards
    Emperor, // create 2 random tarot cards
    Hierophant, // turn 2 cards into bonus cards
    Lovers, // turn 1 card into a wildcard
    Chariot, // turn 1 card into a steel card
    Justice, // turn 1 card into a glass card
    Hermit, // double money, max +$20
    WheelOfFortune, // 1/4 chance: add edition to random joker
    Strength, // increase rank of 2 cards by 1
    HangedMan, // destroy 2 cards
    Death, // turn left card into right card
    Temperance, // add money = total joker sell value, max +$50
    Devil, // turn 1 card into a gold card
    Tower, // turn 1 card into a stone card
    Star, // turn 3 cards into diamonds
    Moon, // turn 3 cards into clubs
    Sun, // turn 3 cards into hearts
    Judgement, // create 1 random joker
    World // turn 3 cards into spades
}

public enum ConsumableType : byte
{
    Tarot,
    Planet
}

public readonly struct ConsumableCard
{
    public readonly ConsumableType Type;
    public readonly TarotCardType TarotCard;
    public readonly HandType PlanetCard;

    public ConsumableCard(TarotCardType tarotCard)
    {
        Type = ConsumableType.Tarot;
        TarotCard = tarotCard;
    }

    public ConsumableCard(HandType planetCard)
    {
        Type = ConsumableType.Planet;
        PlanetCard = planetCard;
    }
}

public sealed class ConsumableState
{
    public readonly List<ConsumableCard> Cards = [];
    public int Capacity = 2;
    public TarotCardType LastUsedTarotCard { get; internal set; }
}

public static class UseTarotCardMove
{
    internal static void ApplyTarotCard(GameState gameState, TarotCardType tarotCard, params ReadOnlySpan<int> cardIndices)
    {
        Card[] hand = gameState.HandState.Hand.ToArray();
        Card[] originalCards = new Card[cardIndices.Length];
        for (int i = 0; i < cardIndices.Length; ++i)
            originalCards[i] = hand[cardIndices[i]];

        switch (tarotCard)
        {
            case TarotCardType.Null:
                return;
            case TarotCardType.Fool:
                if (gameState.ConsumableState.LastUsedTarotCard != TarotCardType.Null &&
                    gameState.ConsumableState.Cards.Count < gameState.ConsumableState.Capacity)
                {
                    gameState.ConsumableState.Cards.Add(new(gameState.ConsumableState.LastUsedTarotCard));
                }
                break;
            case TarotCardType.Magician:
            case TarotCardType.Empress:
            case TarotCardType.Hierophant:
            case TarotCardType.Lovers:
            case TarotCardType.Chariot:
            case TarotCardType.Justice:
            case TarotCardType.Devil:
            case TarotCardType.Tower:
            {
                Enhancement enhancement = tarotCard switch
                {
                    TarotCardType.Magician => Enhancement.Lucky,
                    TarotCardType.Empress => Enhancement.Mult,
                    TarotCardType.Hierophant => Enhancement.Bonus,
                    TarotCardType.Chariot => Enhancement.Steel,
                    TarotCardType.Justice => Enhancement.Glass,
                    TarotCardType.Devil => Enhancement.Gold,
                    TarotCardType.Tower => Enhancement.Stone,
                    _ => Enhancement.None
                };
                for (int i = 0; i < cardIndices.Length; ++i)
                {
                    Card card = hand[cardIndices[i]];
                    Suit suit = tarotCard == TarotCardType.Lovers ? Suit.All : card.Suit;
                    hand[cardIndices[i]] = new(card.Rank, suit, enhancement, card.Edition, card.Seal);
                }
                break;
            }
            case TarotCardType.HighPriestess:
            {
                int cardsToCreate = Math.Min(2, gameState.ConsumableState.Capacity - gameState.ConsumableState.Cards.Count);
                for (int i = 0; i < cardsToCreate; ++i)
                {
                    HandType handType = (HandType)gameState.Random.NextInRange(1, Enum.GetValues<HandType>().Length);
                    gameState.ConsumableState.Cards.Add(new(handType));
                }
                break;
            }
            case TarotCardType.Emperor:
            {
                int cardsToCreate = Math.Min(2, gameState.ConsumableState.Capacity - gameState.ConsumableState.Cards.Count);
                for (int i = 0; i < cardsToCreate; ++i)
                {
                    TarotCardType createdCard = (TarotCardType)gameState.Random.NextInRange(1, Enum.GetValues<TarotCardType>().Length);
                    gameState.ConsumableState.Cards.Add(new(createdCard));
                }
                break;
            }
            case TarotCardType.Hermit:
                gameState.ShopState.Money += Math.Min(gameState.ShopState.Money, 20);
                break;
            case TarotCardType.WheelOfFortune:
                if (gameState.Random.NextFlip(0.25f))
                {
                    JokerInstance[] eligibleJokers = gameState.JokerState.Jokers
                        .Where(joker => joker.Edition == Edition.None)
                        .ToArray();
                    if (eligibleJokers.Length > 0)
                    {
                        JokerInstance joker = eligibleJokers[gameState.Random.Next(eligibleJokers.Length)];
                        joker.Edition = (Edition)gameState.Random.NextInRange(1, Enum.GetValues<Edition>().Length);
                    }
                }
                break;
            case TarotCardType.Strength:
                for (int i = 0; i < cardIndices.Length; ++i)
                {
                    Card card = hand[cardIndices[i]];
                    byte rank = card.Rank == 14 ? (byte)2 : (byte)(card.Rank + 1);
                    hand[cardIndices[i]] = new(rank, card.Suit, card.Enhancement, card.Edition, card.Seal);
                }
                break;
            case TarotCardType.HangedMan:
            {
                bool[] destroyedHandCards = new bool[hand.Length];
                for (int i = 0; i < cardIndices.Length; ++i)
                    destroyedHandCards[cardIndices[i]] = true;
                hand = hand.Where((_, index) => !destroyedHandCards[index]).ToArray();
                break;
            }
            case TarotCardType.Death:
                hand[cardIndices[0]] = hand[cardIndices[1]];
                break;
            case TarotCardType.Temperance:
            {
                int sellValue = 0;
                foreach (JokerInstance joker in gameState.JokerState.Jokers)
                    sellValue += Math.Max(1, joker.JokerData.BasePrice / 2);
                gameState.ShopState.Money += Math.Min(sellValue, 50);
                break;
            }
            case TarotCardType.Star:
            case TarotCardType.Moon:
            case TarotCardType.Sun:
            case TarotCardType.World:
            {
                Suit suit = tarotCard switch
                {
                    TarotCardType.Star => Suit.Diamond,
                    TarotCardType.Moon => Suit.Club,
                    TarotCardType.Sun => Suit.Heart,
                    _ => Suit.Spade
                };
                for (int i = 0; i < cardIndices.Length; ++i)
                {
                    Card card = hand[cardIndices[i]];
                    hand[cardIndices[i]] = new(card.Rank, suit, card.Enhancement, card.Edition, card.Seal);
                }
                break;
            }
            case TarotCardType.Judgement:
                if (gameState.JokerState.Jokers.Count < gameState.GameData.MaxJokers)
                    gameState.JokerState.AddJoker(gameState.Random.NextPick(gameState.GameData.Jokers));
                break;
        }

        if (cardIndices.Length > 0)
        {
            if (tarotCard == TarotCardType.HangedMan)
            {
                for (int i = 0; i < originalCards.Length; ++i)
                    gameState.DeckState.RemoveCardFromFullDeck(originalCards[i]);
            }
            else
            {
                for (int i = 0; i < originalCards.Length; ++i)
                    gameState.DeckState.ReplaceCardInFullDeck(originalCards[i], hand[cardIndices[i]]);
                Array.Sort(hand, Card.RankComparer);
            }

            gameState.HandState.SetHand(hand);
        }

        if (tarotCard != TarotCardType.Fool)
            gameState.ConsumableState.LastUsedTarotCard = tarotCard;
    }
}
