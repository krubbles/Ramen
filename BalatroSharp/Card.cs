namespace BalatroAI
{
    public enum Suit : byte
    {
        None,
        Diamond,
        Club,
        Heart,
        Spade,
        All,
    }


    public struct Card
    {
        public int Rank;
        public Suit Suit;

        public Card(int rank, Suit suit) 
        {
            Rank = rank;
            Suit = suit;
        }

        public readonly bool IsNull => Rank == 0;

        public override readonly string ToString()
        {
            if (IsNull)
                return "__";
            return $"{CardParseUtils.CharForRank(Rank)}{CardParseUtils.CharForSuit(Suit)}";
        }

        public static Card Parse(ReadOnlySpan<char> text)
        {
            if (text.Length < 2)
                throw new FormatException($"Card text length {text.Length} is too short, should be >= 2");
            if (text[0] == '_')
                return Null;
            return new Card(CardParseUtils.RankForChar(text[0]), CardParseUtils.SuitForChar(text[1]));
        }

        public static readonly Card Null = new();

        public static bool operator ==(Card a, Card b) => a.Suit == b.Suit && a.Rank == b.Rank;
        public static bool operator !=(Card a, Card b) => a.Suit != b.Suit && a.Rank != b.Rank;
    }

    public static class CardUtils 
    {
        public static Suit SmearedSuit(Suit suit)
        {
            return suit switch
            {
                Suit.Diamond => Suit.Heart,
                Suit.Club => Suit.Spade,
                _ => suit
            };
        }

    }

    public static class CardParseUtils
    {
        public static char CharForSuit(Suit suit)
        {
            return suit switch
            {
                Suit.None => 'N',
                Suit.Diamond => 'D',
                Suit.Heart => 'H',
                Suit.Club => 'C',
                Suit.Spade => 'S',
                Suit.All => 'A',
                _ => throw new NotSupportedException($"Unrecognized suit {suit}")
            };
        }

        public static Suit SuitForChar(char c)
        {
            return c switch
            {
                'N' => Suit.None,
                'D' => Suit.Diamond,
                'H' => Suit.Heart,
                'C' => Suit.Club,
                'S' => Suit.Spade,
                'A' => Suit.All,
                _ => throw new NotSupportedException($"Unrecognized suit char {c}")
            };
        }


        public static char CharForRank(int rank)
        {
            return rank switch
            {
                2 => '2',
                3 => '3',
                4 => '4',
                5 => '5',
                6 => '6',
                7 => '7',
                8 => '8',
                9 => '9',
                10 => 'T',
                11 => 'J',
                12 => 'Q',
                13 => 'K',
                14 => 'A',
                _ => throw new NotSupportedException($"Unrecognized rank {rank}")
            };
        }

        public static int RankForChar(char c)
        {
            return c switch
            {
                '2' => 2,
                '3' => 3,
                '4' => 4,
                '5' => 5,
                '6' => 6,
                '7' => 7,
                '8' => 8,
                '9' => 9,
                'T' => 10,
                'J' => 11,
                'Q' => 12,
                'K' => 13,
                'A' => 14,
                _ => throw new NotSupportedException($"Unrecognized rank char {c}")
            };
        }

        public static Card[] ParseHand(string hand)
        {
            string[] cardTexts = hand.Split(' ');
            Card[] cards = new Card[cardTexts.Length];
            for (int i = 0; i < cardTexts.Length; ++i)
            {
                cards[i] = Card.Parse(cardTexts[i]);
            }
            return cards;
        }
    }
}
