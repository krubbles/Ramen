# GameState Class Info
- Represents the state of a game of Balatro.
- Maintains a history of all state changes (Move class) which are all invertable. 
- State should never change except through a move.
- RNG state is automatically reverted when moves are reverted.