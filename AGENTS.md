
# Code Architecture Overview
This solution is broken up into 4 projects:
1. Game - An implementation of the game logic of Balatro, with specialized features designed to be used by an AI agent.
2. AI - An AI agent that plays Balatro, self-play code, and training code. 
3. ConsoleApp - The entry point for the project. Supports commands for training and testing the AI. 
4. UnitTests - Unit tests.

# Game Project Class Descriptions

## GameState
The GameState is a class representing the state of a Balatro game. It contains all state needed to simulate the game, as well as the full move history for the game. 

- All freshly constructed GameStates must be in exactly the same state.
- All state changes must be performed by serializable, revertable Move objects, inheriting from Move.
- The functions moves use to modify the GameState must be internal or private, and should not be called by any code except Move objects.
- The same sequence of moves applied to a freshly constructed GameState must result in the same state

## Move
Moves are objects representing a change to the GameState that inherit from Move. 

All moves should 
- Be sealed public classes. 
- Have a constructor that takes in any parameters to the move. These should be immutable and public readonly fields. This state gets serialized so it should be as small as possible. 
- Implement GetMoveType(), which returns an enum value specific to this move type.
- Implement Apply(), which applies the move. The GameState is provided as a field named gameState. The apply function can cache any data in private fields it needs to revert the move.
- Implement Revert(), which reverts the move.
- Have an internal child class called Serializer, which inherets from IMoveSerializer. This class should serialize and deserialize all non-cache state in the move. This class should have no fields and no declared constructor. The Move.cs file has a static field containing all move serializers, which must contain a reference to an instance of this serializer. 

## GameData
This class contains config data about the rules for the game. 
- It should not be mutated.

# AI Project Class Descriptions

## Agent
This class is an AI agent that chooses moves to play on a GameState.

- The agent is a policy-only model.
- Generating synthetic gameplay data is the performance bottleneck for this project. Any code downstream of this process should be optimized for performance.

## GameEvalModel
This class is a policy network for evaluating moves.

## ITensorGroup
This is an interface for classes that contain a group of related tensors. It supports applying specific torchsharp functions on all its children using reflection. It will recursively call into child ITensorGroup objects.

## GameStateTensors
This class is a group of tensors representing the state of a GameState. It is used by the GameEvalModel to evaluate moves.

## MoveTensors
This class is a group of tensors representing the state of a Move. It is used by the GameEvalModel to evaluate moves.

# Code Guidelines
- Use modern C#. 
- Always use file-scoped namespace declarations.
- Namespace declaration goes first, then using statements. Line break between the two. 
- All fields should be declared at the top of the class.
- Private fields should start with an underscore.
- Public fields and properties should be pascal case.
- The constructors should be declared immediatley after the fields and before all other functions.
- There should be a line-break seperating each function.
- Variables declarations should always be explicitly typed, never use var. The only exception is DisposeScopes.
- Constructors should not declare the type if not required (use new() instead of new Foo())
- Public functions should always have short summary blocks. These should provide a brief description about what the function does and any unobvious details that would be important to a caller.
- Function summary blocks should not describe implementation details or obvious facts.
- When calling a function, arguments should be named if their meaning cannot be implied from the function name and type+name of the passed in expression. 
- Short functions (< 10 lines or so) do not need to be commented. 
- The code in longer functions should be broken up into a couple steps. Each step should start with a single line command, and there should be a line break between each step.
- For major, public classes, use small summary blocks on their public fields. Don't bother for minor or private classes. 
# Torch Sharp Guidelines
- C#/pytorch interop is slow. Do it using as view calls as possible (ex: use data<float>.ToArray() instead of indexing the tensor and calling .item<float>() for each item)
- Use DisposeScopes. 
