**`elabStringAsExpr` can't elaborate types that mention the goal's universes**

`Canonical2.elabStringAsExpr` (Util.lean) elaborates in a fresh `TermElabM` with no universe names in scope, so any string mentioning the theorem's universes fails. This breaks `generate` (and the Typewriter RPCs) on universe-polymorphic goals.

```lean
import Canonical
open Lean Meta Elab Tactic Canonical2

universe u

theorem repro (α : Type u) (l : List α) : l = l := by
  run_tac do
    (← getMainGoal).withContext do
      try let e ← elabStringAsExpr "∀ (β : Type u) (f : α → β), List.map f l = List.map f l"
          logInfo m!"ok {e}"
      catch ex => logInfo m!"{ex.toMessageData}"
  rfl
```

Output: `unknown universe level 'u'`

Fix (same signature): put the local context's universe names in scope.

```lean
def elabStringAsExpr (str : String) (type : Option Expr := none) : MetaM Expr := do
  let stx ← IO.ofExcept (Parser.runParserCategory (← getEnv) `term str)
  let mut lvls : CollectLevelParams.State := {}
  if let some t := type then lvls := collectLevelParams lvls t
  for decl in ← getLCtx do lvls := collectLevelParams lvls decl.type
  TermElabM.run' do Term.withLevelNames lvls.params.toList do withoutErrToSorry do
    elabTermAndSynthesize stx type
```

With the fix the repro prints `ok ∀ (β : Type u) (f : α → β), List.map f l = List.map f l`.
