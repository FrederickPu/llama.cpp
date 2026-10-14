import Lean
import Canonical.Canonical2.Basic
import Canonical.Canonical2.LLM
import Training.Repo
import Training.InfoTree
import Training.Basic

open Lean Meta Elab Tactic

namespace Canonical2

def isNumericSubscript (c : Char) : Bool :=
  '₀' ≤ c && c ≤ '₉'

def subscriptToNum (c : Char) : Char :=
  match c with
  | '₀' => '0'
  | '₁' => '1'
  | '₂' => '2'
  | '₃' => '3'
  | '₄' => '4'
  | '₅' => '5'
  | '₆' => '6'
  | '₇' => '7'
  | '₈' => '8'
  | '₉' => '9'
  | _   => c

def increment (s : String) : String :=
  let pre := s.dropEndWhile (fun c => isNumericSubscript c)
  let num := s.takeEndWhile (fun c => isNumericSubscript c)
  pre.toString ++ ((num.toString.map (subscriptToNum)).toNat?.getD 0 + 1).toSubscriptString

partial def disambiguate (s : String) : MetaM String := do
  if ((← getLCtx).findFromUserName? s.toName).isSome then disambiguate (increment s) else return s

def desiredName (userName : Name) (tactic : String) : String :=
  match userName.getRoot with
  | .str _ s => s
  | _ => ((tactic.splitOn).firstM (fun s : String =>
    if !s.isEmpty && isIdFirst ((String.Pos.Raw.get s) 0) && (s.drop 1).all isIdRest then some s else none
  )).getD "this"

inductive Kind
| hypothesis : Kind
| goal : Kind -- imp case
deriving ToJson

structure Premise where
  goal : String
  tactic : String
  premises : Array Name
deriving ToJson

-- structure Automation where
--   goal : String
--   tactic : String
--   premises : Array Name
--   preprocess : Bool
--   preprocessOutput : String
--   grind : Bool
--   grindOutput : String
--   canonical : Bool
--   canonicalOutput : String
--   librarySearch : Bool
--   librarySearchOutput : String
-- deriving ToJson

/-
 Given a current goal tactic state string the next relevant sublemmas/hypothesis to be "have drafted"
-/
structure HaveDraft where
  tactic : String
  goal : String
  kind : Kind
  name : String
  type : String
  removals : Array String
  messages : Array Message
  tools : Json
deriving ToJson

def mkCurriedImplication (hyps : Array LocalDecl) (goal : Expr) : MetaM Expr := do
  -- Extract the `Expr.fvarId` for each hypothesis
  let fvars ← hyps.mapM fun hyp => return mkFVar hyp.fvarId

  -- Use mkForallFVars to abstract the goal over the fvars
  mkForallFVars fvars goal

def extractHypsFromGoal (mvarId : MVarId) : MetaM (Array LocalDecl) := do
  mvarId.withContext do
    pure <| (← getLCtx).foldl (init := #[]) fun acc decl =>
      if decl.isAuxDecl || decl.isImplementationDetail then acc else acc.push decl

def hypsDiff (goalBefore : MVarId) (goalAfter : MVarId) : MetaM (Array LocalDecl × Array LocalDecl) := do
  let beforeDecls ← extractHypsFromGoal goalBefore
  let afterDecls ← extractHypsFromGoal goalAfter
  let beforeFVars := beforeDecls.map (·.fvarId)
  let afterFVars := afterDecls.map (·.fvarId)
  let newHyps := afterDecls.filter fun h => !(beforeFVars.contains h.fvarId)
  let removals := beforeDecls.filter fun h => !(afterFVars.contains h.fvarId)
  return (newHyps, removals)

-- mvar with same type as the goal (but doesn't have the metadata/name)
def goalCopy (goal : MVarId) : MetaM MVarId := do
  return (← goal.withContext do mkFreshExprMVar (← goal.getType)).mvarId!

def isDuplicate (mvar : MVarId) (type : Expr) : MetaM Bool := do
  withTransparency .none do
    let goalType ← mvar.getType
    let any ← (← mvar.getDecl).lctx.anyM fun decl => isDefEqGuarded type decl.type
    return any || (← isDefEqGuarded type goalType)

def goalPremisesMessage (goal premises : String) : Message :=
  { role := "user", content := s!"Goal:\n```lean4\n{goal}\n```\n\nLibrary theorems:\n```lean4\n{premises}\n```" }

def subgoalPrompt (goal : String) (premises : String) : Array Message :=
  #[{ role := "system", content := "Propose one intermediate `have` toward the goal." }, goalPremisesMessage goal premises]

def haveSchema (hyps : Array String) : Json := Json.mkObj [
    ("type", "object"),
    ("properties", Json.mkObj [
      ("type", Json.mkObj [("type", "string")]),
      ("name", Json.mkObj [("type", "string")]),
      ("clear", Json.mkObj [
        ("type", "array"),
        ("items", Json.mkObj [("type", "string"), ("enum", Json.arr (hyps.map Json.str))])
      ])
    ]),
    ("required", Json.arr #["type", "name", "clear"])
  ]

def subgoalTools (hyps : Array String) : Json :=
  Json.arr #[Json.mkObj [
    ("type", "function"),
    ("function", Json.mkObj [
      ("name", "have"),
      ("parameters", haveSchema hyps)
    ])
  ]]

def premise (range : Position × Position) (filepath : System.FilePath) (prevEnv : Environment) (tactic : String) (original : MVarId) (premises : Array Name) : MetaM Premise := do
  let original ← (← original.clone).exposeNames
  let result := {
    goal := (← Meta.ppGoal original).pretty,
    tactic, premises
  }

  if (← IO.rand 0 99) == 0 then withEnv prevEnv try
    IO.print s!"{filepath} {range}: {tactic} "
    original.withContext do preprocess original
    if let #[goal] := (← getMVarsNoDelayed (← instantiateMVars (.mvar original))) then
      let test (x : MetaM NameSet) : MetaM Bool := withoutModifyingMCtx do try
        let _ ← x; pure true
        catch _ => pure false
      goal.withContext do
        if ← test (automateGrind goal premises) then
          IO.println "(grind)"
        else if ← test (automateCanonical goal premises (timeout := 1)) then
          IO.println "(canonical)"
        else if ← test (automateLibrarySearch goal) then
          IO.println "(librarySearch)"
        else
          IO.println "(failure)"
    else
      IO.println "(preprocess)"
    catch _ =>
      IO.println "(error)"
      pure ()


  return result

def draft (tacticText : String) (name : String) (goal : MVarId) (type : Expr)
  (kind : Kind) (removals : Array FVarId) : MetaM (Option HaveDraft) := do
  let goal ← (← goal.clone).exposeNames
  goal.withContext do
    if ← isDuplicate goal type then return none
    let goalString := (← Meta.ppGoal goal).pretty
    let typeString ← pp type
    let removalsStrs ← removals.mapM fun id => do pure (← id.getUserName).toString
    let hyps : Array String := (← getLCtx).foldr (init := #[]) fun decl acc =>
      if decl.isImplementationDetail then acc else acc.push decl.userName.toString
    let toolCall : ToolCall := {
      id := "have"
      function := {
        name := "have"
        arguments := (Json.mkObj [
          ("name", Json.str name),
          ("type", Json.str typeString),
          ("clear", Json.arr (removalsStrs.map Json.str))
        ]).compress
      }
    }
    let messages := (subgoalPrompt goalString "").push {
      role := "assistant", content := "", tool_calls := #[toolCall]
    }
    let tools := subgoalTools hyps
    return some {
      tactic := tacticText, name, goal := goalString,
      kind, type := typeString, removals := removalsStrs,
      messages, tools
    }

/-
 returns tuple and Array of (goal, name, type)
 `automation` shouldn't modify the `MetaM` state
-/
def haveDrafts (range : Position × Position) (filepath : System.FilePath) (prevEnv : Environment)
  (goalsBefore : List MVarId)
  (goalsAfter : List MVarId) (tacticText : String) (tac : Syntax) : MetaM (Array HaveDraft × Array Premise) := do
  let intersection := goalsBefore.inter goalsAfter
  let goalsBefore := goalsBefore.removeAll intersection
  let goalsAfter := goalsAfter.removeAll intersection
  let [gBefore] := goalsBefore | return (#[], #[])
  let constants ← collectConstants tac gBefore
  if let [gAfter'] := goalsAfter then
    if ← isDefEqGuarded (← gAfter'.getType) (← gBefore.getType) then -- isDefEq used instead of == for mdata
      let (newHyps, removals) ← hypsDiff gBefore gAfter'
      let fvarNewHyps := newHyps.filter (!·.isLet)
      let letNewHyps := newHyps.filter (·.isLet)
      if ← gAfter'.withContext do fvarNewHyps.allM (isProp ·.type) then
        let mut goal ← goalCopy gBefore
        let mut drafts : Array HaveDraft := #[]
        let mut premises : Array Premise := #[]
        let mut fvars := #[]
        for (h, idx) in fvarNewHyps.zipIdx do
          let subseq := (fvarNewHyps.take idx).map (Expr.fvar ·.fvarId)
          let replaced ← gAfter'.withContext do pure ((← mkCurriedImplication letNewHyps h.type).replaceFVars subseq fvars)

          let (draft, fvar, goal', premise) ← goal.withContext do
            let name ← disambiguate (desiredName h.userName tacticText)
            let draft ← draft tacticText name goal replaced .hypothesis (removals.map (·.fvarId))
            let (fvar, goal') ← goal.note name.toName (← Meta.mkSorry replaced false) replaced
            let premise ← premise range filepath prevEnv tacticText (← mkFreshExprMVar replaced).mvarId! constants.toArray
            pure (draft, fvar, goal', premise)

          if let .some draft := draft then
            drafts := drafts.push draft
          premises := premises.push premise
          fvars := fvars.push (.fvar fvar)
          goal := goal'
        return (drafts, premises)

  let mut drafts : Array HaveDraft := #[]
  let mut goal ← goalCopy gBefore

  -- IO.println s!"{← gBefore.withContext do ppExpr (← gBefore.getType)} num goals after: {goalsAfter.length}"
  for gAfter in goalsAfter do
    let (newHyps, removals) ← hypsDiff gBefore gAfter
    let imp ← gAfter.withContext do mkCurriedImplication newHyps (← gAfter.getType)

    let (draft, goal') ← goal.withContext do
      let gAfterName := ((← getMCtx).getDecl gAfter).userName
      let name ← disambiguate (desiredName gAfterName tacticText)
      let goal' := (← goal.note name.toName (← Meta.mkSorry imp false) imp).2
      let draft ← draft tacticText name goal imp .goal (removals.map (·.fvarId))
      pure (draft, goal')

    if let .some draft := draft then
      drafts := drafts.push draft
    goal := goal'
  let premise ← premise range filepath prevEnv tacticText goal constants.toArray
  return (drafts, #[premise])

/-
  `automation` should not change the `MetaM` state
-/
def tacticToHaveDraft (range' : Position × Position) (filepath : System.FilePath) (prevEnv : Environment) (ti : TacticInfo) (ctx : ContextInfo) : MetaM (Array HaveDraft × Array Premise) := do
  IO.setNumHeartbeats 0
  let range := ti.stx.getRange?.get!
  -- IO.println ti.stx.formatStx
  let tacticText := String.Pos.Raw.extract ctx.fileMap.source range.start range.stop
  if ["intro", "intros", "rintro", "expose_names"].any (·.isPrefixOf tacticText) then
    return (#[], #[])
  haveDrafts range' filepath prevEnv ti.goalsBefore ti.goalsAfter tacticText ti.stx

instance : Append (Array HaveDraft × Array Premise) :=
  ⟨fun (x₁, y₁) (x₂, y₂) => (x₁ ++ x₂, y₁ ++ y₂)⟩

/-- Traverse an `InfoTree`, invoking extraction inline at each relevant node.
For `ofTacticInfo` nodes we drive `tacticToHaveDraft`; for `ofTermInfo` nodes we
extract premises directly from `i.expr` and call `automation`, avoiding the
re-elaboration of the term syntax (which fails on typeclass synthesis when the
expected type is unknown). -/
partial def processInfoTree (filepath : System.FilePath) (prevEnv : Environment) (t : InfoTree)
    (ctx? : Option ContextInfo := none) (allowTerm := true) : IO ((Array HaveDraft × Array Premise) × Bool) := do
  match t with
  | .context ctx t => processInfoTree filepath prevEnv t (ctx.mergeIntoOuter? ctx?)
  | .node info ts =>
    let mut result : Array HaveDraft × Array Premise := (#[], #[])
    let mut hasTacticInfo := false
    let allowTermRec :=
      match info with
      | .ofTacticInfo _ => true
      | .ofTermInfo _ => false
      | _ => allowTerm
    for child in ts do
      let (arr, b) ← processInfoTree filepath prevEnv child ctx? allowTermRec
      hasTacticInfo := hasTacticInfo || b
      result := result ++ arr
    -- let mut allowTerm := allowTerm
    if isOriginal info then if let some ctx := ctx? then
      -- let ctx := { ctx with env := prevEnv }
      match info with
      | .ofTacticInfo i =>
        if isSubstantive i then
          let ti : TacticInfo := { i with
            goalsAfter :=
              if !i.goalsAfter.isEmpty then i.goalsAfter
              else getSubNodeGoals t i.goalsBefore ctx? }
          result := result ++ (← ctx.runMetaM {} do withOptions roundtrip do
            tryCatchRuntimeEx (handler := fun e => do
              IO.println s!"ExtractData: Failed to process {stxRange ctx.fileMap ti.stx}: {← e.toMessageData.format}"
              pure (#[], #[])
            ) do tacticToHaveDraft (stxRange ctx.fileMap ti.stx) filepath prevEnv ti ctx)
        hasTacticInfo := true
      | .ofTermInfo i =>
        if !i.isBinder && allowTerm && !hasTacticInfo then if let some expectedType := i.expectedType? then
          result := result ++ (← ctx.runMetaM i.lctx do withOptions roundtrip do
            tryCatchRuntimeEx (handler := fun e => do
              IO.println s!"ExtractData: Failed to process {stxRange ctx.fileMap i.stx}: {← e.toMessageData.format}"
              pure (#[], #[])
            ) do
              if (← isProp expectedType) then
                IO.setNumHeartbeats 0
                let constants ← withTransparency .all do pure ((← explicitConstants i.expr) ++ (← collectUnfolds (expectedType) (← inferType i.expr)))
                let constants := constants.toList.eraseDups.toArray
                let range := i.stx.getRange?.get!
                let tacticText := String.Pos.Raw.extract ctx.fileMap.source range.start range.stop
                let goal := (← mkFreshExprMVar expectedType).mvarId!
                let goal ← goal.tryClearMany ((← goal.getDecl).lctx.auxDeclToFullName.keys.toArray)
                pure (#[], #[← premise (stxRange ctx.fileMap i.stx) filepath prevEnv tacticText goal constants])
              else pure (#[], #[]))
        | _ => pure ()
    return (result, hasTacticInfo)
  | _ => return ((#[], #[]), false)

/-- One progress-log entry, keyed by `const` so it shares `readSkip` with the
rollout path. For extraction the key is a command's ordinal in the file. -/
structure ExtractEntry where
  const : String
  done : Bool
deriving ToJson

structure ExtractCtx where
  fileName : System.FilePath
  draftsFile : System.FilePath
  premisesFile : System.FilePath
  progressFile : System.FilePath
  skip : Std.HashSet String

/-- The output paths derived from a source file's base path (its path without the
`.lean` suffix). Drafts/premises are appended to the `.jsonl` files as extraction
proceeds — these are what downstream reads, so partial runs are still picked up.
The progress file records per-command markers and is renamed to the `done` marker
on clean completion (mirroring the rollout artifacts). -/
structure OutPaths where
  drafts : System.FilePath
  premises : System.FilePath
  progress : System.FilePath
  done : System.FilePath

def outPaths (basePath : String) : OutPaths where
  drafts := basePath ++ ".drafts.jsonl"
  premises := basePath ++ ".premises.jsonl"
  progress := basePath ++ ".extract.jsonl.tmp"
  done := basePath ++ ".extract.jsonl"

/-- Walk the command snapshots, extracting per command. Each command is keyed by its
ordinal `idx` (stable across re-parses): before processing we append an `inProgress`
marker to the progress file, and after we append the command's drafts/premises to the
`.jsonl` data files and finalize the marker. On resume, commands already in `ctx.skip`
- including an unfinished crasher whose marker was never finalized - are skipped, so a
deterministic crash no longer stalls the file. -/
partial def processSnapshot (ctx : ExtractCtx)
  (snap : Language.Lean.CommandParsedSnapshot) (prevEnv : Environment) (idx : Nat := 0)
  : IO Unit := do
  let key := toString idx
  unless ctx.skip.contains key do
    appendJsonl ctx.progressFile (ExtractEntry.mk key false)
    let trees := snap.elabSnap.infoTreeSnap.get.infoTree?.toList
    let (drafts, premises) ← trees.foldlM (init := (#[], #[])) fun acc t => do
      return acc ++ (← processInfoTree ctx.fileName prevEnv t).1
    for draft in drafts do appendJsonl ctx.draftsFile draft
    for premise in premises do appendJsonl ctx.premisesFile premise
    removeLastLine ctx.progressFile
    appendJsonl ctx.progressFile (ExtractEntry.mk key true)
  if let some next := snap.nextCmdSnap? then
    processSnapshot ctx next.task.get snap.elabSnap.resultSnap.get.cmdState.env (idx + 1)

unsafe def main (args : List String) : IO Unit := do
  let [entry] := args |
    IO.println "Usage: extract_data <entry_directory>"
    return

  Lean.initSearchPath (← Lean.findSysroot)
  enableInitializersExecution
  let mut state := {}
  let mut tasks := []
  let mut i := 0
  let files ← getLeanFiles entry

  while h : i < files.size do
    if h : tasks.length >= 8 then
      let (_, tasks') ← IO.waitAny' tasks (by grind)
      tasks := tasks'
    else
      let filepath := files[i]
      i := i + 1
      let path ← IO.FS.realPath filepath
      let p := outPaths (path.toString.dropSuffix ".lean").toString

      if ← shouldSkip path p.done then
        continue

      enableInitializersExecution
      IO.println s!"{filepath} ({i}/{files.size})"

      let (state', inputCtx, parserState, commandState₀, _) ← mkFileStateCached path state
        #[{ module := `Canonical.Symbols }]
      state := state'
      let st ← IO.processCommandsIncrementally inputCtx parserState
        { commandState₀ with infoState.enabled := true } none

      tasks := (← IO.asTask do
        let skip ← readSkip p.progress
        let ctx : ExtractCtx :=
          { fileName := filepath, draftsFile := p.drafts, premisesFile := p.premises,
            progressFile := p.progress, skip }
        processSnapshot ctx st.initialSnap commandState₀.env
        -- Done: promote the per-command progress log to the completion marker (empty
        -- if the file had no commands). The drafts/premises jsonl are already on disk
        -- (appended as we went), so a crash before this just leaves them for resume.
        if ← p.progress.pathExists then IO.FS.rename p.progress p.done
        else IO.FS.writeFile p.done ""
      ) :: tasks

  for task in tasks do
    let _ ← task.get

  IO.println "Done processing all files"
where
  shouldSkip (leanPath donePath : System.FilePath) : IO Bool := do
  if ← donePath.pathExists then
    let leanModified := (← leanPath.metadata).modified
    let doneModified := (← donePath.metadata).modified
    pure (leanModified ≤ doneModified)
  else
    pure false
