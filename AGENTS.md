# general
- do not use special case logic and hacks to avoid changing existing code structure. just change it. 
- do not write helper functions with 1 or 2 lines of code that are used < 5 times. rare exception for seperating out logic that is frequently modified, like reward functions. 
- do not sanitize functions inputs unless they take user data.

# memory management
- If a function creates any tensor objects that should be disposed, start the function with `using var dScope = NewDisposeScope();`
- If a function returns a tensor that should be considered part of the parent's scope, call `[Tensor].ToOuterScope()`
- If a tensor should be persisted long term, call `[Tensor].DetachFromScope()`
- By default, all tensors are disposed between rollouts. If a tensor must persist between rollouts (ex: static constant) call `[Tensor].PersistForever()`
- Do not use try/finally to dispose tensors. Errors are fatal anyways.
