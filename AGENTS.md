# Commands
- Use mac commands
- ALWAYS run dotnet build with elevated sandbox permissions. 

# Git
- Use short commit messages.

# Unit Tests
- Use `Assert.That()`

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
- Separate functions with line-breaks
- Use `new()` instead of `new Foo()`
- Use `[]` instead of `Array.Empty<T>()`
- Use `[a, b]` instead of `new Foo[] { a, b }`
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
- Include `using static TorchSharp.torch` in all files that use TorchSharp.
- If you need to access most of the data in a tensor, use `.data<T>().ToArray()` instead of using `.item<T>()` multiple times. 
- Don't use using statements on individual tensors, instead, add `using var scope = NewDisposeScope()` to the top of the function.
- Call `.ToOuterScope()` on tensors created in functions whose ownership should be transferred to their calling context. 
- Never call `.ToOuterScope()` in functions that don't have a dispose scope. They are already in the outer dispose scope.
- Call `.DetachFromScope()` on long-life tensors.
- Always call `.Size()` on the output of a tensor op that changes the size.
- All code should assume the first dimension is the batch dimension.