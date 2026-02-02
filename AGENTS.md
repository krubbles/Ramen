# Platform
- This is a Mac

# Git
- Use short commit messages.

# Unit Tests
- Use Assert.That()

# Commenting
- For longer functions, break them up into sections with short comments explaining what each section does. 

# Code Guidelines
- Use file-scoped namespace declarations.
- Namespace declaration goes first, then using statements. Line break between the two. 
- Classes, functions, properties, and public/internal fields: PascalCase
- Private/protected fields: _camelCase
- Tuple fields: camelCase
- Never use the private keyword.
- Never use the var keyword (except dispose scopes)
- Seperate functions with line-breaks
- Constructor syntax priority: [], then new(), then new Foo()
- Use [a, b] instead of new Foo[] { a, b }
- When calling a function, arguments should be named if their meaning cannot be implied from the calling code. 
- Game project does not include TorchSharp and should not have AI related code.
- Don't use argument validation on non-user facing functions.
- Always use float, never double. 

# Ordering Code Inside A Class
1. Fields
2. Properties that directly return fields
3. Contructors (public, then private)
4. Other properties
5. non static functions (public then internal then protected then private)
6. static functions (public then internal then protected then private)
Note: private functions with exactly 1 caller should be placed directly after their calling function.

# Torch Sharp Guidelines
- Inlude `using static TorchSharp.torch` in all files that use TorchSharp.
- If you need to access most of the data in a tensor, use .data<T>().ToArray() instead of using .item<T>() multiple times. 
- Don't use using statements on individual tensors, instead, add using var scope = NewDisposeScope() to the top of the function.
- Call .MoveToOuterDisposeScope() on tensors created in functions who's ownership should be transfered to their calling context. 
- Call .DetachFromDisposeScope() on long-life tensors.
- The first argument of amax() is a dim array. 