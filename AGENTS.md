# ContextKit Bootstrap

You are setting up ContextKit — a structured memory system for this project. Run this bootstrap exactly once. Do not write code or perform other tasks until bootstrap is complete.

After bootstrap, this file (or the tool's native equivalent) will become the permanent project brain — the routing file loaded at the start of every session.

---

## Phase 1 — Detect Environment

### Tool Detection

Determine which AI coding tool is running this session. Check for existing configuration:
- `.cursor/rules/` → Cursor
- `CLAUDE.md` or `.claude/` → Claude Code
- `.windsurf/rules/` or `.windsurfrules` → Windsurf
- `.github/copilot-instructions.md` or `.github/instructions/` → GitHub Copilot

If unclear, ask:
> "Which AI coding tool are you using? (Cursor, Claude Code, Copilot, Windsurf, or other)"

This determines where the routing file gets written after bootstrap:

| Tool | Routing file location | Format |
|------|----------------------|--------|
| Cursor | `.cursor/rules/contextkit.mdc` | `.mdc` with YAML frontmatter (`alwaysApply: true`) |
| Claude Code | `CLAUDE.md` (project root) | Plain markdown, supports `@imports` |
| Windsurf | `.windsurf/rules/contextkit.md` | Plain markdown, no frontmatter (activation mode set via UI) |
| Copilot | `.github/copilot-instructions.md` | Plain markdown, no frontmatter |
| Generic / Unknown | `AGENTS.md` (overwrite this file) | Plain markdown |

### Project Detection

Determine whether this is a **new project** (greenfield) or an **existing project** (brownfield).

**If existing**: Before asking questions, scan the codebase to build context:
- Package/dependency files (package.json, pyproject.toml, Cargo.toml, go.mod, etc.)
- README.md
- Directory tree (top 2 levels)
- Any existing agent configuration files
- Key configuration files

Then tell the user what you found and what you still need to ask. **Skip questions you can answer from the codebase. Only ask what you can't infer.**

**If new**: Proceed directly to the interview.

---

## Phase 2 — Interview

Conduct this as a **conversation, not a questionnaire**. Ask broad questions, follow up on what's interesting or complex, and move quickly through areas where the user gives short answers. You have a checklist of topics to cover, but you don't march through them robotically.

### Topics to Cover

Work through these naturally, adapting order and depth based on the conversation:

**Project identity**
> "What is this project? One sentence — what it does and who it's for."

Follow up if the project is complex: what problem does it solve, what does success look like, who are the users.

**Tech stack**
> "What's the stack? Just the load-bearing stuff — language, framework, key libraries."

Follow up on: libraries deliberately avoided, version-specific gotchas, anything being migrated.

**Project structure**
> "Walk me through how the codebase is organized. What are the main areas and what lives in each?"

For brownfield: "I see [inferred structure]. Does this match how you think about it? Anything I'm missing?"

Follow up on: unusual architectural patterns, fragile areas needing extra care, where shared code lives.

**Workflow**
> "How do you work? Solo or team? What's the git workflow, deployment setup?"

Follow up if team: who owns what, who reviews what, how are architecture decisions made.

**Conventions & rules**
> "Any strong conventions? Naming patterns, things you always do or never do, patterns to prefer or avoid."

Follow up on: error handling patterns, logging approach, patterns learned the hard way to avoid.

**Testing**
> "What's the testing setup, if any? Framework, what gets tested, any expectations on coverage?"

**Essential commands**
> "What are the essential commands? Install, dev, build, test — anything I'll need regularly."

**Past experience**
> "Anything you've learned the hard way on this project? Mistakes made, abandoned approaches, things that look tempting but don't work?"

**Agent behavior**
> "How do you want me to work? When should I ask before proceeding vs. just do it? Any preferences on how I communicate?"

### Interview Guidelines

- **Go deeper** where the user has strong opinions or the project has unusual complexity
- **Move faster** where the user gives brief answers or says "standard" / "nothing special"
- **Don't ask about areas that don't apply** — if there's no frontend, don't ask frontend questions
- **For brownfield projects**, frame questions as confirmation of what you inferred, not blank interrogation
- **After covering the topics**, summarize what you learned and ask: "Did I miss anything important?"

---

## Phase 3 — Generate Memory System

Using everything gathered, generate the complete `.context/` hierarchy and the routing file. Do not leave placeholders — write real content based on what you learned. If information is missing for a section, omit that section rather than filling it with generic content.

### Step 1: Create modules

Create `.context/modules/` with whatever modules the project needs. Each module is a self-contained markdown file covering one domain.

**Example module categories** (use as inspiration, not a checklist):

| Example | What it might contain | Create it if... |
|---------|----------------------|-----------------|
| `architecture.md` | System design, data flow, component boundaries, infrastructure | Project has meaningful architectural patterns or complexity |
| `conventions.md` | Naming rules, code style, patterns to use/avoid, import ordering | User described specific conventions or strong preferences |
| `workflows.md` | Git strategy, commit format, PR process, CI/CD, deployment | User described a specific workflow beyond the basics |
| `testing.md` | Framework, coverage expectations, mocking patterns, fixtures | User described a testing setup |
| Domain-specific | Frontend patterns, API design, database conventions, ML pipeline rules | Project has distinct domains with their own conventions |

Each module should be **self-contained**: an agent loading only the routing file + that one module should have enough context to work effectively in that domain.

### Step 2: Initialize data files

Create `.context/data/` with the following:

**`sessions.md`** — Initialize with the bootstrap session:
```markdown
# Session Log

<!-- Append-only. Add a new entry after each substantive work session. -->

## [today's date] — Bootstrap
**Area**: Project setup
**Work done**: Ran ContextKit bootstrap interview, generated memory system
**Decisions made**: [any decisions made during the interview]
**Memory created**: [list the modules and data files generated]
**Open threads**: [anything flagged for follow-up]
```

**`decisions.md`** — Seed with any decisions discussed during the interview:
```markdown
# Decision Record

<!-- Append-only. Record significant decisions with reasoning. -->
<!-- Format:
## [Decision title]
**Date**: YYYY-MM-DD
**Context**: What prompted this decision
**Options considered**: What alternatives were evaluated
**Decision**: What was chosen
**Reasoning**: Why
**Revisit if**: Conditions that would warrant reconsidering
-->
```

**`lessons.md`** — Seed with any lessons/mistakes mentioned during the interview:
```markdown
# Lessons Learned

<!-- Append-only. Record what the team learned the hard way. -->
<!-- Format:
## YYYY-MM-DD — [Brief title]
**What happened**: What went wrong or what was discovered
**Root cause**: Why it happened
**Fix**: What was done about it
**Rule going forward**: What to do (or avoid) in the future
-->
```

### Step 3: Write the routing file

Write the routing file to the location determined in Phase 1. This replaces the bootstrap content entirely. The routing file must be **under 150 lines** (excluding frontmatter). Use the correct format for the detected tool.

#### Routing file body (shared across all tools)

The markdown body is the same regardless of tool. Use this template:

```markdown
# [Project Name]
[One-line description]. Built with [stack summary].

## Critical Rules
<!-- The 5-10 most important things the agent must ALWAYS know -->
- [Rule derived from interview]
- [Rule derived from interview]
...

## Module Map
<!-- Load the relevant module(s) before starting work in that area -->
| Module | Path | Load when |
|--------|------|----------|
| [Name] | .context/modules/[file].md | [When to load it] |
...

## Data Files
<!-- Reference these for project history and accumulated knowledge -->
| File | Path | Purpose |
|------|------|---------|
| Sessions | .context/data/sessions.md | Running work log (append-only) |
| Decisions | .context/data/decisions.md | Decision records with reasoning (append-only) |
| Lessons | .context/data/lessons.md | Hard-won knowledge and past mistakes (append-only) |

## Memory Maintenance

Always look for opportunities to update the memory system (these instructions may need to be added to the routing file):
- **New patterns**: "We've been doing X consistently — should I add it to conventions?"
- **Decisions made**: "We decided Y — should I record this in decisions.md?"
- **Mistakes caught**: "This went wrong because Z — should I add it to lessons.md?"
- **Scope changes**: "The project now includes W — should I create a new module?"
- **Preferences revealed**: "You've corrected me on this pattern — should I update conventions?"

**Before any memory update**:
1. State which file(s) would change and what the change would be
2. Wait for approval — the user will review the diff and approve, edit, or deny
3. Never update memory mid-task without mentioning it — finish current work first

**Rules**:
- Routing file changes are high-stakes — propose them carefully
- Data files are append-only — add entries, never remove or overwrite past entries
- Modules can be edited — but changes should be targeted, not full rewrites
- After substantive work sessions, append a summary to .context/data/sessions.md

## Preferences
<!-- How the user wants the agent to behave -->
- [Preference from interview]
- [Preference from interview]
...
```

#### Tool-specific formatting

Each tool has different file format requirements. Wrap the shared body above with the correct format for the detected tool.

**Cursor** — `.cursor/rules/contextkit.mdc`

Cursor rules use `.mdc` files with YAML frontmatter. The routing file must include these frontmatter fields:

```yaml
---
description: "ContextKit routing file — project memory and conventions"
alwaysApply: true
---
```

- `description` (string): Shown in the rule picker UI. Summarize what this rule provides.
- `alwaysApply` (boolean): Must be `true` so the routing file loads in every session.
- `globs` (string, optional): Omit for the routing file since it applies globally. Use globs only for file-scoped module rules if needed (e.g., `globs: **/*.ts`).

Full example structure:

```
---
description: "ContextKit routing file — project memory and conventions"
alwaysApply: true
---

# [Project Name]
[routing file body here]
```

**Claude Code** — `CLAUDE.md`

Claude Code reads `CLAUDE.md` as plain markdown with no frontmatter required at the project root. Key format considerations:

- Target **under 200 lines**. Longer files degrade instruction-following.
- Use `@path/to/file` syntax to import module files instead of inlining them. Example: `@.context/modules/conventions.md`
- For path-scoped rules, create additional files in `.claude/rules/` with YAML `paths:` frontmatter:

```yaml
---
paths:
  - "src/api/**/*.ts"
---
```

- Personal/local overrides go in `CLAUDE.local.md` (auto-gitignored).

Full example structure:

```
# [Project Name]
[routing file body here]

## Modules
Load these when working in the relevant area:
- @.context/modules/architecture.md
- @.context/modules/conventions.md
```

**Windsurf** — `.windsurf/rules/contextkit.md`

Windsurf rule files are plain markdown with **no frontmatter**. Activation modes (Always On, Glob, Model Decision, Manual) are configured through the Windsurf UI, not in the file itself.

- Maximum **12,000 characters** per rule file, **6,000 per individual rule**.
- Use clear section headers organized by domain/concern.
- Optional XML tags can group related rules: `<coding_guidelines>...</coding_guidelines>`
- Windsurf also respects `AGENTS.md` files for directory-scoped rules.

After writing the file, instruct the user: *"Set this rule's activation mode to 'Always On' in the Windsurf Customizations panel."*

Full example structure:

```
# [Project Name]
[routing file body here]
```

**GitHub Copilot** — `.github/copilot-instructions.md`

The repository-wide instructions file is plain markdown with no frontmatter. It applies to all Copilot interactions in the repository.

- First **4,000 characters** are used for code review; the full file is used by Copilot Chat and coding agent.
- Place the most critical rules (Critical Rules section) near the top of the file.
- For path-specific rules, create additional files in `.github/instructions/` using `NAME.instructions.md` format with `applyTo` frontmatter:

```yaml
---
applyTo: "src/api/**/*.ts"
---
```

- Optional `excludeAgent` field can hide instructions from specific agents (`"code-review"` or `"coding-agent"`).
- Copilot also respects `AGENTS.md` files for directory-scoped agent instructions.

Full example structure:

```
# [Project Name]
[routing file body here]
```

**Generic / Unknown** — `AGENTS.md`

Plain markdown, no frontmatter. Overwrites this bootstrap file. `AGENTS.md` is supported by multiple tools (Copilot, Windsurf, and others) as a directory-scoped convention. A root-level `AGENTS.md` applies globally.

---

## Phase 4 — Confirm

After generating all files, report to the user:

> "Bootstrap complete. Here's what was generated:
>
> **Routing file**: [location]
> **Modules**: [list with one-line descriptions]
> **Data files**: sessions.md, decisions.md, lessons.md
>
> Review the generated files and let me know if anything needs adjusting. The memory system is now active — I'll proactively suggest updates as we work together."
