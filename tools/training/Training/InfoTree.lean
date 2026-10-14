import Lean
import Training.Repo

open Lean

/-- Extract the range of a `Syntax` expressed as lines and columns. -/
-- Extracted from the private declaration `Lean.Elab.formatStxRange`,
-- in `Lean.Elab.InfoTree.Main`.
def stxRange (fileMap : FileMap) (stx : Syntax) : Position × Position :=
  let pos    := stx.getPos?.getD 0
  let endPos := stx.getTailPos?.getD pos
  (fileMap.toPosition pos, fileMap.toPosition endPos)

open Lean Elab System

set_option maxHeartbeats 2000000  -- 10x the default maxHeartbeats.


instance : ToJson Substring.Raw where
  toJson s := toJson s.toString

instance : ToJson String.Pos.Raw where
  toJson n := toJson n.1

deriving instance ToJson for SourceInfo
deriving instance ToJson for Syntax.Preresolved
deriving instance ToJson for Syntax
deriving instance ToJson for Position

open Lean Meta Elab Tactic


partial def getSubNodeGoals (infotree : InfoTree) (rootGoals : List MVarId) (ctx? : Option ContextInfo) : (List MVarId) :=
  match infotree with
  | .context ctx t => getSubNodeGoals t rootGoals (ctx.mergeIntoOuter? ctx?)
  | .node info children =>
    match info with
    | .ofTacticInfo info =>
      if (info.goalsBefore != rootGoals) ∧ info.goalsBefore != [] then
        info.goalsBefore
      else
        (children.map (getSubNodeGoals · rootGoals ctx?)).toList.foldr (· ++ ·) []
    | _                   => []
  | .hole _ => []

/-- Find the name for the outermost `Syntax` in this `TacticInfo`. -/
def name? (t : TacticInfo) : Option Name :=
  match t.stx with
  | Syntax.node _ n _ => some n
  | _ => none

/-- The `Syntax` for a `Lean.Elab.Info`, if there is one. -/
def stx? : Info → Option Syntax
  | .ofTacticInfo         info => info.stx
  | .ofTermInfo           info => info.stx
  | .ofPartialTermInfo     info => info.stx
  | .ofCommandInfo        info => info.stx
  | .ofMacroExpansionInfo info => info.stx
  | .ofOptionInfo         info => info.stx
  | .ofFieldInfo          info => info.stx
  | .ofCompletionInfo     info => info.stx
  | .ofUserWidgetInfo     info => info.stx
  | .ofCustomInfo         info => info.stx
  | .ofFVarAliasInfo      _    => none
  | .ofFieldRedeclInfo    info => info.stx
  | .ofChoiceInfo         info => info.stx
  | .ofDelabTermInfo      info => info.stx
  | .ofErrorNameInfo      info => info.stx
  | .ofDocElabInfo        _    => none
  | .ofDocInfo            _    => none

/-- Is the `Syntax` for this `Lean.Elab.Info` original, or synthetic? -/
def isOriginal (i : Info) : Bool :=
  match stx? i with
  | none => true   -- Somewhat unclear what to do with `FVarAliasInfo`, so be conservative.
  | some stx => match stx.getHeadInfo with
    | .original .. => true
    | _ => false

/-- Decide whether a tactic is "substantive",
or is merely a tactic combinator (e.g. `by`, `;`, multiline tactics, parenthesized tactics). -/
def isSubstantive (t : TacticInfo) : Bool :=
  match name? t with
  | none => false
  | some `null => false
  | some ``cdot => false
  | some ``cdotTk => false
  | some ``Lean.Parser.Term.byTactic => false
  | some ``Lean.Parser.Tactic.tacticSeq => false
  | some ``Lean.Parser.Tactic.tacticSeq1Indented => false
  | some ``Lean.Parser.Tactic.«tactic_<;>_» => false
  | some ``Lean.Parser.Tactic.paren => false
  | _ => true

def includeUnfold (n : Name) : MetaM Bool := do
  return (← getReducibilityStatus n) == .semireducible &&
          !(← isProjectionFn n) && !(← isInstance n) &&
          !Lean.Compiler.hasInlineAttribute (← getEnv) n &&
          !Lean.Compiler.hasAlwaysInlineAttribute (← getEnv) n

/-- Harvest the `Name` payloads from any `Meta.isDefEq.delta.*` trace nodes within `msg`.
The payload was emitted as `trace[…] (fn : Name)`, so we render and re-parse. -/
partial def collectDeltaNames : MessageData → MetaM (NameSet)
  | .trace data inner children => do
    let mut init := {}
    if data.cls.getPrefix == `Meta.isDefEq.delta then
      let name := (← inner.toString).trimAscii.toString.toName
      if (← includeUnfold name) then
        if let some thm ← getConstUnfoldEqnFor? name then
          init := { thm }
    children.foldlM (init := init) fun acc c =>
      do pure (acc ++ (← collectDeltaNames c))
  | .withContext _ m => collectDeltaNames m
  | _ => return {}

/-- Run `isDefEq t s`, returning whether it succeeded along with every definition that got
delta-unfolded along the way. -/
def collectUnfolds (t s : Expr) : MetaM (NameSet) :=
  withOptions (·.setBool `trace.Meta.isDefEq.delta true) do
    let saved ← getTraceState
    resetTraceState
    if !(← isDefEqGuarded t s) then return {}
    let traces := (← getTraceState).traces
    setTraceState saved
    traces.foldlM (init := {}) fun acc c =>
      do pure (acc ++ (← collectDeltaNames c.msg))
