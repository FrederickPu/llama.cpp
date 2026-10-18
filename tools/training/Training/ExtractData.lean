import Lean
import Lake
import Canonical
import Training.Repo
import Training.InfoTree
import Training.Basic

open Lean Meta Elab Tactic
open Canonical2 hiding Premise getKind

def EXTRACT_AUTOMATION : Bool := false

/-- Set by `--drafts`: write `.drafts.jsonl` training data. -/
initialize EXTRACT_DRAFTS : IO.Ref Bool ← IO.mkRef false
/-- Set by `--checkAccuracy`: ask the drafter at `daemonPort` for each have and write `.checks.jsonl`. -/
initialize CHECK_ACCURACY : IO.Ref Bool ← IO.mkRef false
/-- Set by `--premises`: run premise selection and write `.premises.jsonl`. -/
initialize EXTRACT_PREMISES : IO.Ref Bool ← IO.mkRef false

/-- Set by `--checkKeys=<file>`: only check drafts whose `[goal, type]` is a line of the file. -/
initialize CHECK_KEYS : IO.Ref (Option (Std.HashSet String)) ← IO.mkRef none

def checkKey (goal type : String) : String := (Json.arr #[Json.str goal, Json.str type]).compress

initialize draftStats : IO.Ref (Nat × Nat) ← IO.mkRef (0, 0)

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

/-- A goal with its used premises and the selector's false positives (those it
ranked above every used premise), all rendered as the selector sees them. -/
structure Premise where
  goal : String
  tactic : String
  premises : Array String
  negativePremises : Array String
deriving ToJson

private def getKind (cinfo : ConstantInfo) : MetaM String := do
  let env ← getEnv
  return match cinfo with
    | .axiomInfo _  => "axiom"
    | .thmInfo _    => "theorem"
    | .opaqueInfo _ => "opaque"
    | .defnInfo i   => if isInstanceCore env i.name then "instance" else "def"
    | .inductInfo i =>
      if isClass env i.name then "class"
      else if isStructure env i.name then "structure"
      else "inductive"
    | .ctorInfo _ | .recInfo _ | .quotInfo _ => "def"

/-- Copy of `Canonical.Canonical2.PremiseSelection.toPremise`'s rendering (docstring,
kind, signature; `none` when denied or unprintable). Training data must match the
text the premise server embeds exactly, so keep this in sync with it. -/
def renderPremise (name : Name) : MetaM (Option String) := do
  if LibrarySuggestions.isDeniedPremise (← getEnv) name then return none
  withCurrHeartbeats do try
    let kind ← getKind (← getConstInfo name)
    let ⟨fmt, _⟩ ← PrettyPrinter.ppSignature name
    let docPrefix := match ← findSimpleDocString? (← getEnv) name with
      | some doc => "/-- " ++ doc.dropSuffix " " ++ " -/\n"
      | none => ""
    return some (docPrefix ++ kind ++ " " ++ fmt.pretty 1000000000)
  catch _ => return none

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
  messages : Array Canonical2.Message
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


def premise (range : Position × Position) (filepath : System.FilePath) (prevEnv : Environment) (tactic : String) (original : MVarId) (premises : Array Name) : MetaM Premise := do
  unless ← EXTRACT_PREMISES.get do return { goal := "", tactic, premises := #[], negativePremises := #[] }
  let original ← (← original.clone).exposeNames
  let ranked := (← LibrarySuggestions.select original { maxSuggestions := 30 }).map (·.name)
  let result := {
    goal := (← Meta.ppGoal original).pretty,
    tactic,
    premises := ← premises.filterMapM renderPremise,
    negativePremises := ← (rankedNegatives ranked (.ofArray premises)).filterMapM renderPremise
  }

  if EXTRACT_AUTOMATION && (← IO.rand 0 99) == 0 then withEnv prevEnv try
    IO.print s!"{filepath} {range}: {tactic} "
    original.withContext do preprocess original
    if let #[goal] := (← getMVarsNoDelayed (← instantiateMVars (.mvar original))) then
      let test (x : MetaM NameSet) : MetaM Bool := withoutModifyingMCtx do try
        let _ ← x; pure true
        catch _ => pure false
      goal.withContext do
        if ← test (automateGrind goal premises) then
          IO.println "(grind)"
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

structure DraftCheck where
  tactic : String
  goal : String
  type : String
  generated : String
  premises : Array String
  defeq : Bool
deriving ToJson

def isDuplicate (goal : MVarId) (type : Expr) : MetaM Bool := goal.withContext do
  try checkDuplicate goal type; return false catch _ => return true

/-- TEMPORARY: `Canonical2.elabStringAsExpr`, but with the goal's universe names in scope, so
types like `Type u_1` elaborate. Remove (with `generate'`) once CanonicalLean does this. -/
def elabStringAsExpr' (mvar : MVarId) (str : String) (type : Option Expr := none) : MetaM Expr := do
  let stx ← IO.ofExcept (Parser.runParserCategory (← getEnv) `term str)
  let mut lvls : CollectLevelParams.State := collectLevelParams {} (← mvar.getType)
  for decl in ← getLCtx do lvls := collectLevelParams lvls decl.type
  Term.TermElabM.run' do Term.withLevelNames lvls.params.toList do Term.withoutErrToSorry do
    Term.elabTermAndSynthesize stx type

/-- TEMPORARY: verbatim copy of `Canonical2.generate`, except it calls `elabStringAsExpr'`. -/
def generate' (log : String → String → IO Unit) (mvar : MVarId) (premises : Array Name)
  (provider : Provider) (config : LLMConfig := defaultConfig provider)
  (retries := if config.temperature == 0.0 then 1 else 3) (wait : UInt32 := 0) (stall := 0)
  (prompt : String := "") : MetaM (MetaTask RolloutSubgoal) := mvar.withContext do
  let llmRef : IO.Ref (Option (CancelableTask (Except IO.Error Response))) ← IO.mkRef none
  MetaM.toTask (cancel := do if let some t ← llmRef.get then t.cancel) do
    log "generate" ""
    IO.sleep wait
    if ← IO.checkCanceled then throwError "canceled."
    let start ← IO.monoMsNow
    let goal ← Meta.ppGoal mvar
    let premisesString ← premisesString premises[:12]
    let hyps : Array String := (← getLCtx).foldr (init := #[]) fun decl acc =>
      if decl.isImplementationDetail then acc else acc.push decl.userName.toString
    let tools := subgoalTools hyps provider
    let constrained := !provider matches .Llama _ && !provider matches .Vllm _ _
    let mut messages := subgoalPrompt goal.pretty premisesString provider
    unless prompt.isEmpty do messages := messages.push { role := "user", content := prompt }
    let mut session : Option String := none
    let mut error := ""
    for _ in [0:retries] do
      log "prefill" goal.pretty
      log "error" error
      let task ← llm log provider config messages tools constrained session
      llmRef.set (some task)
      let response ← IO.ofExcept task.task.get
      llmRef.set none
      if ← IO.checkCanceled then throwError "canceled."
      messages := messages.push response.message
      session := session <|> response.session
      let some tc := response.message.tool_calls[0]? |
        error := s!"No tool call. Retrying…"
        messages := messages.push { role := "user", content := "You must use the \"have\" tool." }
        continue
      try
        let args : Have ← IO.ofExcept (Json.parse tc.function.arguments >>= fromJson?)
        let type ← elabStringAsExpr' mvar args.type
        checkDuplicate mvar type
        let h ← addSubgoal mvar args.name.toName type (args.clear.map (·.toName))
        IO.sleep (stall + start - (← IO.monoMsNow)).toUInt32
        return ⟨mvar, ← getMVarsNoDelayed (← instantiateMVars (.mvar mvar)), h, args.name.toName, type, args.type, messages, tools⟩
      catch e =>
        error ← e.toMessageData.toString
        messages := messages.push { role := "tool", tool_call_id := tc.id, content := error }
    throwError "generate failed."

/-- Ask the drafter at `daemonPort` for a subgoal with Canonical's `generate` (via `generate'`; greedy, one attempt)
and record whether it is defeq to the extracted one. Uses the same goal and premises as `draft`. -/
def checkDraft (filepath : System.FilePath) (range : Position × Position) (tactic : String)
    (goal : MVarId) (desired : Expr) (premises : Array Name) : MetaM Unit := do
  unless ← CHECK_ACCURACY.get do return
  let goal ← (← goal.clone).exposeNames
  -- `draft` drops duplicates of a hypothesis, so they have no training examples.
  if ← isDuplicate goal desired then return
  let goalString := (← Meta.ppGoal goal).pretty
  let typeString ← pp desired
  if let some keys ← CHECK_KEYS.get then
    unless keys.contains (checkKey goalString typeString) do return
  let rendered ← premises.filterMapM renderPremise
  let checksFile := ((← IO.FS.realPath filepath).toString.dropSuffix ".lean").toString ++ ".checks.jsonl"
  let provider : Provider := .Llama s!"http://{daemonHost}:{daemonPort}/chat/completions"
  let lastError ← IO.mkRef ""
  let log := fun kind msg => if kind == "error" && !msg.isEmpty then lastError.set msg else pure ()
  let (hit, generated, tag) ← try
    let task ← generate' log goal premises provider { defaultConfig provider with temperature := 0.0 }
    match task.task.get with
    | .error e => pure (false, s!"{e}: {← lastError.get}", "(error)")
    | .ok (subgoal, _, sMeta) =>
      let eq ← goal.withContext do withMCtx sMeta.mctx do isDefEqGuarded subgoal.type desired
      pure (eq, subgoal.typeString, if eq then "(defeq)" else "(mismatch)")
  catch e =>
    pure (false, s!"{← e.toMessageData.toString}", "(error)")
  draftStats.modify fun (h, n) => (h + if hit then 1 else 0, n + 1)
  appendJsonl checksFile ({
    tactic, goal := goalString, type := typeString, generated,
    premises := rendered, defeq := hit
  } : DraftCheck)
  printFlush s!"{range}: {tactic} {tag}
"

/-- A training example in `generate`'s format: its prompt and tools, answered by a `have` call. -/
def draft (tacticText : String) (name : String) (goal : MVarId) (type : Expr)
  (kind : Kind) (removals : Array FVarId) (premises : Array Name) : MetaM (Option HaveDraft) := do
  unless ← EXTRACT_DRAFTS.get do return none
  let goal ← (← goal.clone).exposeNames
  goal.withContext do try
    let _ ← checkDuplicate goal type
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
    let messages := (subgoalPrompt goalString (← premisesString premises[:12]) (Provider.Llama "")).push {
      role := "assistant", content := "", tool_calls := #[toolCall]
    }
    let tools := subgoalTools hyps (Provider.Llama "")
    return some {
      tactic := tacticText, name, goal := goalString,
      kind, type := typeString, removals := removalsStrs,
      messages, tools
    }
  catch _ => return none

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
  let constants ← collectConstants' tac gBefore
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
            checkDraft filepath range tacticText goal replaced constants.toArray
            let name ← disambiguate (desiredName h.userName tacticText)
            let draft ← draft tacticText name goal replaced .hypothesis (removals.map (·.fvarId)) constants.toArray
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
      checkDraft filepath range tacticText goal imp constants.toArray
      let gAfterName := ((← getMCtx).getDecl gAfter).userName
      let name ← disambiguate (desiredName gAfterName tacticText)
      let goal' := (← goal.note name.toName (← Meta.mkSorry imp false) imp).2
      let draft ← draft tacticText name goal imp .goal (removals.map (·.fvarId)) constants.toArray
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
  /-- Selection quality: per record with any positives, `log (pos / (pos + neg))`;
  summed with the record count for the file's geometric mean. -/
  quality : IO.Ref (Nat × Float)

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
    if ← EXTRACT_DRAFTS.get then for draft in drafts do appendJsonl ctx.draftsFile draft
    if ← EXTRACT_PREMISES.get then for premise in premises do
      appendJsonl ctx.premisesFile premise
      let pos := premise.premises.size
      unless pos == 0 do
        ctx.quality.modify fun (n, s) => (n + 1, s + Float.log (pos.toFloat / (pos + premise.negativePremises.size).toFloat))
    removeLastLine ctx.progressFile
    appendJsonl ctx.progressFile (ExtractEntry.mk key true)
  if let some next := snap.nextCmdSnap? then
    processSnapshot ctx next.task.get snap.elabSnap.resultSnap.get.cmdState.env (idx + 1)

unsafe def main (args : List String) : IO Unit := do
  let flags := args.filter (·.startsWith "--")
  let [entry] := args.filter (!·.startsWith "--") |
    IO.println "Usage: extract_data [--premises] [--drafts] [--checkAccuracy [--checkKeys=<file>]] <entry>"
    return
  EXTRACT_PREMISES.set (flags.contains "--premises")
  EXTRACT_DRAFTS.set (flags.contains "--drafts")
  CHECK_ACCURACY.set (flags.contains "--checkAccuracy")
  if let some path := flags.findSome? (·.dropPrefix? "--checkKeys=") then
    -- Lines are JSON `[goal, type]`; re-encode so the keys don't depend on the writer's escaping.
    let lines := (← IO.FS.readFile path.toString).splitOn "
" |>.filter (!·.trimAscii.isEmpty)
    let keys ← lines.mapM fun line => IO.ofExcept do
      let (g, t) ← fromJson? (α := String × String) (← Json.parse line)
      return checkKey g t
    CHECK_KEYS.set (some (Std.HashSet.ofList keys))

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
        #[{ module := `Canonical.Canonical2.PremiseSelection }, { module := `Canonical.Symbols }]
      state := state'
      let st ← IO.processCommandsIncrementally inputCtx parserState
        { commandState₀ with infoState.enabled := true } none

      tasks := (← IO.asTask do
        let skip ← readSkip p.progress
        let ctx : ExtractCtx :=
          { fileName := filepath, draftsFile := p.drafts, premisesFile := p.premises,
            progressFile := p.progress, skip, quality := ← IO.mkRef (0, 0.0) }
        processSnapshot ctx st.initialSnap commandState₀.env
        let (n, s) ← ctx.quality.get
        IO.println s!"{filepath}: premise selection quality {if n == 0 then "n/a" else toString (Float.exp (s / n.toFloat))} over {n} goals"
        -- Done: promote the per-command progress log to the completion marker (empty
        -- if the file had no commands). The drafts/premises jsonl are already on disk
        -- (appended as we went), so a crash before this just leaves them for resume.
        if ← p.progress.pathExists then IO.FS.rename p.progress p.done
        else IO.FS.writeFile p.done ""
      ) :: tasks

  for task in tasks do
    let _ ← task.get

  if ← CHECK_ACCURACY.get then
    let (hit, n) ← draftStats.get
    IO.println s!"draft accuracy {hit}/{n}"
  IO.println "Done processing all files"
where
  shouldSkip (leanPath donePath : System.FilePath) : IO Bool := do
  if ← donePath.pathExists then
    let leanModified := (← leanPath.metadata).modified
    let doneModified := (← donePath.metadata).modified
    pure (leanModified ≤ doneModified)
  else
    pure false
