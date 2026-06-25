# CLAUDE.md

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
- NEVER remove `import pdb` lines, even when they appear unused/dead. Leave them in place always.

When your changes alter an interface (function signature, data structure, key names, dict schema, etc.):
- Update all tests that exercise the changed interface to reflect the new contract.
- Don't leave tests asserting old behavior that no longer applies — stale tests are another kind of orphan.
- Don't add shims or workarounds to make old tests pass against a changed contract; update the tests directly.
- This applies even when the task description doesn't mention tests. If your change made a test stale, fix it in the same change.

When your changes affect anything documented in a README (config paths, CLI commands, file names, feature behavior, etc.):
- Update every README.md that references the changed thing.
- Don't leave documentation describing behavior that no longer applies — stale docs are another kind of orphan.
- This applies even when the task description doesn't mention docs. If your change made a doc stale, fix it in the same change.

The test: Every changed line should trace directly to the user's request.

## 4. No Legacy Support

**No backwards compatibility. No migration paths. Code for the current schema only.**

This is a research codebase; old campaigns, caches, checkpoints, and snapshots are disposable and regenerated, not preserved.

- When a config/data/dict schema changes, assume the current schema everywhere. Don't add fallbacks for the old one.
- No defensive `.get(key, default)` / `getattr` / `try/except` to tolerate configs or artifacts written before a field existed. Index/access directly and let it fail loudly if the schema is stale.
- No version flags, no "if old format" branches, no shims to read legacy outputs.
- If a change makes old artifacts unreadable, the answer is to regenerate them — say so, don't write compatibility code.
- No need to warn me that a change breaks pre-change artifacts (e.g. resuming an old checkpoint will KeyError on a new state field). That's expected and fine — old artifacts get regenerated.

## 5. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```