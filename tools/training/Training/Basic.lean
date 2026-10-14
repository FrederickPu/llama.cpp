import Lean

open Lean Meta Elab Term

def pp (e : Expr) : MetaM String := TermElabM.run' do
  return (← Meta.ppExpr e).pretty.replace "⋯" "sorry"

def printFlush {α} [ToString α] (s : α) : IO Unit := do
  let out ← IO.getStdout
  out.putStr (toString s)
  out.flush -- Flush the output
