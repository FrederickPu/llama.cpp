import Lean
import Canonical.Canonical2.Util

open Lean Meta Elab Term

/-- Premises the selector ranked above the highest-ranked used premise, minus the
used ones: the false positives that outscored the truth. -/
def rankedNegatives (ranked : Array Name) (used : NameSet) : Array Name :=
  (ranked.popWhile (!used.contains ·)).filter (!used.contains ·)

/-- `Canonical2.collectConstants`, but an identifier that is not a global constant
(a hypothesis name in tactic syntax) contributes nothing instead of throwing. -/
def collectConstants' (s : Syntax) (goal : MVarId) : MetaM NameSet := goal.withContext do
  match s with
  | .ident .. => return NameSet.ofArray ((← try realizeGlobalConst s catch _ => pure []).head?.toArray)
  | .node _ k args => do
    if (`Lean.Parser.Term).isPrefixOf k then try
        let e ← TermElabM.run' (ctx := { errToSorry := false }) do elabTerm s none
        withTransparency .all do Canonical2.explicitConstants e
      catch _ => return {}
    else
      args.foldlM (init := {}) fun acc arg => do pure (acc ++ (← collectConstants' arg goal))
  | _ => return {}

def pp (e : Expr) : MetaM String := TermElabM.run' do
  return (← Meta.ppExpr e).pretty.replace "⋯" "sorry"

def printFlush {α} [ToString α] (s : α) : IO Unit := do
  let out ← IO.getStdout
  out.putStr (toString s)
  out.flush -- Flush the output
