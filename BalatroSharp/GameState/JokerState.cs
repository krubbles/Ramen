
namespace BalatroAI
{
    /// <summary>
    /// Stores the state of jokers in the game. Handles adding, removing, and triggering joker effects.
    /// </summary>
    public class JokerState
    {
        public readonly GameState GameState;
        readonly GameData _gameData;

        public readonly List<JokerInstance> Jokers = new();

        public JokerState(GameState gameState)
        {
            GameState = gameState;
            _gameData = gameState.GameData;
        }

        public void AddJoker(Joker joker) => AddJoker(new JokerInstance(joker));

        public void AddJoker(JokerInstance joker)
        {
            Jokers.Add(joker);
            joker.JokerData.OnAdd?.Invoke(GameState, joker);
        }

        public void RemoveJoker(JokerInstance joker)
        {
            Jokers.Remove(joker);
            joker.JokerData.OnRemove?.Invoke(GameState, joker);
        }

        /// <summary>
        /// Runs all joker effects that trigger when a hand is played. Ex: Jimbo
        /// </summary>
        public void OnPlayHand()
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnJokerTrigger?.Invoke(GameState, joker);
            }
        }

        /// <summary>
        /// Runs all joker effects that trigger when a card is scored. Ex: Greedy Joker
        /// </summary>
        public void OnScoreCard(Card card)
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnScoreCard?.Invoke(GameState, joker, card);
            }
        }

        /// <summary>
        /// Runs all joker effects that trigger right before a card is scored. Mostly used for retriggering jokers. Ex: Dusk
        /// </summary>
        public void OnBeginScoringCard(Card card)
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnBeginScoringCard?.Invoke(GameState, joker, card);
            }
        }

        /// <summary>
        /// Runs all joker effects that trigger when a card is discarded. Ex: Mail-In Rebate
        /// </summary>
        public void OnDiscardCard(Card card)
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnDiscardCard?.Invoke(GameState, joker, card);
            }
        }

        /// <summary>
        /// Runs all joker effects that trigger before a hand is played. Ex: Space Joker
        /// </summary>
        public void OnBeforePlayHand()
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnPlayHand?.Invoke(GameState, joker);
            }
        }

        /// <summary>
        /// Runs all joker effects that trigger when a hand is discarded. Ex: Burnt Joker
        /// </summary>
        public void OnDiscardHand()
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnDiscardHand?.Invoke(GameState, joker);
            }
        }
    }
}
