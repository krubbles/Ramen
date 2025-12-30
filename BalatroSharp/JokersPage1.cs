using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BalatroAI
{
    public static class Jokers
    {
        public static readonly Joker Joker = new()
        {
            Name = "Joker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                gameState.ScoringState.AddMultToCurrentHand(4);
            }
        };

        public const int SinJokerMultBonus = 3;

        // plus mult for specific suits

        public static readonly Joker GreedyJoker = new()
        {
            Name = "GreedyJoker",
            Rarity = Rarity.Common,
            OnScoreCard = (gameState, joker, card) =>
            {
                if (gameState.PatternMatchingState.SuitsMatch(card.Suit, Suit.Diamond))
                    gameState.ScoringState.AddMultToCurrentHand(SinJokerMultBonus);
            }
        };

        public static readonly Joker LustyJoker = new()
        {
            Name = "LustyJoker",
            Rarity = Rarity.Common,
            OnScoreCard = (gameState, joker, card) =>
            {
                if (gameState.PatternMatchingState.SuitsMatch(card.Suit, Suit.Heart))
                    gameState.ScoringState.AddMultToCurrentHand(SinJokerMultBonus);
            }
        };

        public static readonly Joker WrathfulJoker = new()
        {
            Name = "WrathfulJoker",
            Rarity = Rarity.Common,
            OnScoreCard = (gameState, joker, card) =>
            {
                if (gameState.PatternMatchingState.SuitsMatch(card.Suit, Suit.Spade))
                    gameState.ScoringState.AddMultToCurrentHand(SinJokerMultBonus);
            }
        };

        public static readonly Joker GluttonousJoker = new()
        {
            Name = "GluttonousJoker",
            Rarity = Rarity.Common,
            OnScoreCard = (gameState, joker, card) =>
            {
                if (gameState.PatternMatchingState.SuitsMatch(card.Suit, Suit.Club))
                    gameState.ScoringState.AddMultToCurrentHand(SinJokerMultBonus);
            }
        };

        // plus mult for specific hands

        public static readonly Joker JollyJoker = new()
        {
            Name = "JollyJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsPair)
                    gameState.ScoringState.AddMultToCurrentHand(8);
            }
        };

        public static readonly Joker ZanyJoker = new()
        {
            Name = "ZanyJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.Contains3OAK)
                    gameState.ScoringState.AddMultToCurrentHand(12);
            }
        };

        public static readonly Joker MadJoker = new()
        {
            Name = "MadJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsTwoPair)
                    gameState.ScoringState.AddMultToCurrentHand(10);
            }
        };

        public static readonly Joker CrazyJoker = new()
        {
            Name = "CrazyJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsStraight)
                    gameState.ScoringState.AddMultToCurrentHand(12);
            }
        };

        public static readonly Joker DrollJoker = new()
        {
            Name = "DrollJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsFlush)
                    gameState.ScoringState.AddMultToCurrentHand(10);
            }
        };

        // plus chips for specific hands

        public static readonly Joker SlyJoker = new()
        {
            Name = "SlyJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsPair)
                    gameState.ScoringState.AddChipsToCurrentHand(50);
            }
        };

        public static readonly Joker WilyJoker = new()
        {
            Name = "WilyJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.Contains3OAK)
                    gameState.ScoringState.AddChipsToCurrentHand(100);
            }
        };

        public static readonly Joker CleverJoker = new()
        {
            Name = "CleverJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsTwoPair)
                    gameState.ScoringState.AddChipsToCurrentHand(80);
            }
        };

        public static readonly Joker DeviousJoker = new()
        {
            Name = "DeviousJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsStraight)
                    gameState.ScoringState.AddChipsToCurrentHand(100);
            }
        };

        public static readonly Joker CraftyJoker = new()
        {
            Name = "CraftyJoker",
            Rarity = Rarity.Common,
            OnJokerTrigger = (gameState, joker) =>
            {
                if (gameState.HandState.ActiveHandPatterns.ContainsFlush)
                    gameState.ScoringState.AddChipsToCurrentHand(80);
            }
        };
    }
}
