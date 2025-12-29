
namespace BalatroAI
{
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

        public void OnJokerTriggers()
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnJokerTrigger?.Invoke(GameState, joker);
            }
        }

        public void OnScoreCard(Card card)
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnScoreCard?.Invoke(GameState, joker, card);
            }
        }

        public void OnBeginScoringCard(Card card)
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnBeginScoringCard?.Invoke(GameState, joker, card);
            }
        }

        public void OnDiscardCard(Card card)
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnDiscardCard?.Invoke(GameState, joker, card);
            }
        }


        public void OnPlayHand()
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnPlayHand?.Invoke(GameState, joker);
            }
        }

        public void OnDiscardHand()
        {
            foreach (JokerInstance joker in Jokers)
            {
                joker.JokerData.OnDiscardHand?.Invoke(GameState, joker);
            }
        }
    }
}
