# Session Wrap-Up

The user is about to clear the context window and/or change topics. Before they do, wrap up the current session by updating the project memory system.

## Steps

1. **Summarize the session** — Review everything that happened in this conversation: what was worked on, what changed, what was decided, and what's still open.

2. **Append to `.context/data/sessions.md`** — Add a new entry with today's date and a descriptive title. Include:
   - **Area**: Which part of the project was touched
   - **Work done**: Concrete summary of changes made
   - **Decisions made**: Any choices that were resolved (also record these in decisions.md)
   - **Open threads**: Anything unfinished or flagged for next time

3. **Update other memory files if needed**:
   - **`.context/data/decisions.md`** — If any significant decisions were made during this session, append them with context, reasoning, and revisit conditions.
   - **`.context/data/lessons.md`** — If anything went wrong or a hard-won insight was gained, append it.
   - **`.context/modules/`** — If the architecture, conventions, or any domain knowledge changed, propose targeted edits to the relevant module.
   - **`.cursor/rules/contextkit.mdc`** — Only if a new critical rule emerged. Propose carefully.

4. **Update `README.md` if needed** — If the session introduced changes that affect the project's public-facing documentation (new files, changed commands, altered pipeline, updated results, new dependencies), update the README to match. Skip if nothing user-facing changed.

5. **Report to the user** — Show a brief summary of:
   - What was logged to sessions.md
   - Any other memory files that were updated
   - Whether README was updated (and what changed)
   - Key open threads for next session

## Rules

- Data files (sessions, decisions, lessons) are **append-only** — never remove or overwrite past entries.
- Always **show the user what you're about to write** before writing it.
- If nothing meaningful happened in the session (e.g., just Q&A with no code changes), say so and skip the update.
- Keep session entries concise — a few lines per field, not paragraphs.
