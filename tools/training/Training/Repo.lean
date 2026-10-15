import Lean
import Batteries.Tactic.OpenPrivate

namespace Lean

open private ImportState.mk from Lean.Environment
open private ImportState.moduleNameMap from Lean.Environment
open private ImportState.moduleNames from Lean.Environment
open private ImportedModule from Lean.Environment
open private ImportedModule.mk from Lean.Environment
open private ImportedModule.toEffectiveImport from Lean.Environment
open private ImportedModule.mainModule? from Lean.Environment
open private ImportedModule.needsIRTrans from Lean.Environment
open private ImportedModule.parts from Lean.Environment
open private ImportedModule.irParts from Lean.Environment
open private readModuleDataPartsOfMod from Lean.Environment
open private readIRPartsOfMod from Lean.Environment
open Lean Elab System

partial def importModulesCore'
    (imports : Array Import) (globalLevel : OLeanLevel := .private)
    (arts : NameMap ImportArtifacts := {}) (isExported : Bool := globalLevel < .private) :
    ImportStateM Unit := do
  go imports (importAll := true) (isExported := isExported) (needsData := true) (needsIRTrans := false)
  if globalLevel < .private then
    for i in imports do
      if let some mod := (ImportState.moduleNameMap (← get))[i.module]?.bind (ImportedModule.mainModule? ·) then
        if !mod.isModule then
          throw <| IO.userError s!"cannot import non-`module` {i.module} from `module`"
where
  go (imports : Array Import) (importAll isExported needsData needsIRTrans : Bool) := do
    for i in imports do
      -- `B > none`?
      let needsData := needsData && (i.isExported || importAll)
      -- `B ≥ privateAll`?
      let importAll := globalLevel == .private || importAll && i.importAll
      -- `B ≥ public`?
      let isExported := isExported && i.isExported
      let needsIRTrans := needsIRTrans || needsData && i.isMeta
      let needsIR := needsIRTrans || importAll || globalLevel > .exported
      if !needsData && !needsIR then
        continue

      let irPhases : IRPhases :=
        if importAll then .all
        else if needsIRTrans then .comptime  -- `globalLevel` should *not* be considered here
        else .runtime

      let goRec mod := do
        if let some mod := ImportedModule.mainModule? mod then
          go (importAll := importAll) (isExported := isExported) (needsData := needsData) (needsIRTrans := needsIRTrans) mod.imports

      let alreadyNamed := (ImportState.moduleNames (← get)).contains i.module
      let some mod ← makeModule i needsData needsIR importAll isExported irPhases needsIRTrans alreadyNamed | continue
      goRec mod
      if !alreadyNamed then
        modify fun s => ImportState.mk
          ((ImportState.moduleNameMap s).insert i.module mod)
          ((ImportState.moduleNames s).push i.module)
      else
        modify fun s => ImportState.mk ((ImportState.moduleNameMap s).insert i.module mod) (ImportState.moduleNames s)
  makeModule (i : Import) (needsData needsIR importAll isExported : Bool) (irPhases : IRPhases) (needsIRTrans alreadyNamed : Bool) : ImportStateM (Option ImportedModule) := do
    match (ImportState.moduleNameMap (← get))[i.module]? with
    | none =>
      -- newly discovered module
      let parts  ← if needsData then loadData i else pure #[]
      let irParts ← if needsIR then loadIR i else pure #[]
      return some (ImportedModule.mk ({ i with importAll, isExported, irPhases, hasData := needsData }) parts irParts needsIRTrans)
    | some mod =>
      -- cached module: merge flags
      let importAll'    := importAll    || mod.importAll
      let isExported'   := isExported   || mod.isExported
      let needsData'    := needsData    || mod.hasData
      let needsIRTrans' := needsIRTrans || ImportedModule.needsIRTrans mod
      let needsIR'      := needsIRTrans' || importAll'
      let irPhases'     := if irPhases == mod.irPhases then irPhases else .all
      if alreadyNamed && importAll' == mod.importAll && isExported' == mod.isExported &&
          needsData' == mod.hasData && needsIRTrans' == ImportedModule.needsIRTrans mod &&
          irPhases' == mod.irPhases then
        return none  -- nothing changed
      let parts  ← if needsData' && (ImportedModule.parts mod).isEmpty then loadData i else pure (ImportedModule.parts mod)
      let irParts ← if needsIR' && (ImportedModule.irParts mod).isEmpty then loadIR i else pure (ImportedModule.irParts mod)
      let imp := (ImportedModule.toEffectiveImport mod).toImport
      return some (ImportedModule.mk (EffectiveImport.mk (Import.mk imp.module importAll' isExported' imp.isMeta) irPhases' needsData') parts irParts needsIRTrans')
  loadData i := do
    if let some arts := arts.find? i.module then
      -- Opportunistically load all available parts.
      -- Producer (e.g., Lake) should limit parts to the proper import level.
      let fnames := arts.oleanParts (inServer := globalLevel ≥ .server)
      readModuleDataParts fnames
    else
      readModuleDataPartsOfMod i.module
  -- .ir.sig + .ir (optional)
  loadIR i := do
    if let some arts := arts.find? i.module then
      -- Opportunistically load all available parts.
      -- Producer (e.g., Lake) should limit parts to the proper import level.
      readModuleDataParts arts.irParts
    else
      readIRPartsOfMod i.module

unsafe def mkFileStateCached (fileName : FilePath) (s : Std.HashMap Name ImportedModule)
  (extraImports : Array Import := #[]) : IO (Std.HashMap Name ImportedModule × Parser.InputContext × Parser.ModuleParserState × Command.State × FileMap) := do
  enableInitializersExecution
  let input ← IO.FS.readFile fileName
  let inputCtx := Parser.mkInputContext input fileName.toString
  let (header, parserState, messages) ← Parser.parseHeader inputCtx
  let imports := (HeaderSyntax.imports header) ++ extraImports

  let level := if HeaderSyntax.isModule header then
    if Elab.inServer.get {} then .server else .exported else .private
  withImporting do
    let ((), s) ← importModulesCore' (globalLevel := level) imports |>.run (s := ImportState.mk s #[])
    let env ← finalizeImport (leakEnv := false) (loadExts := true) (level := level) s imports {}
    return (ImportState.moduleNameMap s, inputCtx, parserState, Command.mkState env messages, FileMap.ofString input)

/-- Resolve `entry` to the list of `.lean` files to process: a directory is
walked recursively, a `.txt` file is read as newline-delimited paths (used to
hand a SLURM job its own disjoint slice), and any other path is taken verbatim. -/
def getLeanFiles (entry : FilePath) : IO (Array FilePath) := do
  if !(← entry.isDir) then
    if entry.extension == some "txt" then
      let content ← IO.FS.readFile entry
      return (content.splitOn "\n" |>.filterMap fun line =>
        let line := line.trimAscii.toString
        if line.isEmpty then none else some (FilePath.mk line)).toArray
    return #[entry]
  let enter := fun x => return (x.fileName.map (!·.startsWith ".")).getD true
  return (← System.FilePath.walkDir entry (enter := enter)).filter fun x => x.extension == some "lean"

/-- Append `x` as one compact JSON line to `path` (creating it if needed). -/
def appendJsonl {α : Type} [ToJson α] (path : System.FilePath) (x : α) : IO Unit :=
  IO.FS.withFile path .append fun h => h.putStr ((ToJson.toJson x).compress ++ "\n")

/-- Remove the final line from `path` (no-op if the file is missing). Used to drop an
entry's `inProgress` placeholder just before its finalized entry is appended. -/
def removeLastLine (path : System.FilePath) : IO Unit := do
  if !(← path.pathExists) then return
  let content ← IO.FS.readFile path
  let trimmed := if content.endsWith "\n" then (content.dropEnd 1).toString else content
  let kept := (trimmed.splitOn "\n").dropLast
  IO.FS.writeFile path (String.join (kept.map (· ++ "\n")))

/-- Keys already attempted in a prior (interrupted) run, read from a progress file's
`const` field. Both in-progress and finalized entries are included, so an
attempted-but-unfinished key (e.g. a crasher) is skipped on resume. -/
def readSkip (path : System.FilePath) : IO (Std.HashSet String) := do
  if !(← path.pathExists) then return ∅
  let content ← IO.FS.readFile path
  return content.splitOn "\n" |>.foldl (init := ∅) fun acc line =>
    match Json.parse line >>= (·.getObjVal? "const") >>= (·.getStr?) with
    | .ok name => acc.insert name
    | .error _ => acc
