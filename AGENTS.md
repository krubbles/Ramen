# Commands
- Don't use && when running commands.

# Git
- Use short commit messages.

# Unit Tests
- Don't write tests for things that can be easily validated by looking at the code.

# Code Guidelines
- Always use file-scoped namespace declarations.
- Namespace declaration goes first, then using statements. Line break between the two. 
- All classes, functions, properties, and non-private/protected fields should be pascal case.
- Never use the private keyword.
- Never use the var keyword (except dispose scopes)
- There should be a line-break seperating each function.
- Use new() syntax whenever possible.
- Function summary blocks should NOT describe implementation details or obvious facts.
- When calling a function, arguments should be named if their meaning cannot be implied from the calling code. 
- Short functions (< 10 lines) do not need to be commented. 

# Ordering Code Inside A Class
1. Fields
2. Properties that directly return fields
3. Contructors (public, then private)
4. Other properties
5. non static functions (public then internal then protected then private)
6. static functions (public then internal then protected then private)

Note: private functions that are only called in one place should be placed directly after their calling function.

# Torch Sharp Guidelines
- If you need to access most of the data in a tensor, use .data<T>.().ToArray() instead of using .item<T>() multiple times. 
- Don't use using statements on individual tensors, instead, add a using var scope = NewDisposeScope() at the top of the function instead.
- Call .MoveToOuterDisposeScope() on tensors created in functions who's ownership should be transfered to their calling context.
