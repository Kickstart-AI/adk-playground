---
name: get-started
description: Help the developer to get started with their project by asking targeted questions, creating a plan and setting up the scaffolding for the project.
---

# Get Started

The developer starts from a blank project and needs help to get started. IMPORTANT: The developer may or may not be a technical person, so you have to adapt your language and style accordingly.
The goal is not to build a complete project, but to set up the basic structure of the repo, setup documentation, guidelines, coding standards, verification steps, etc.

## 1. Investigation
Ask targeted questions interview-style, one question at a time, to find out what the developer wants to build.
- What do they want to build?
- Did they already create a new repo for the project, or is this repo still pointing to the remote of the template repo?
- What is the skill-level of the developer?
- What programming language do they want to use?
- Do they have already an idea about the architecture / stack of the project?
- Are there any coding conventions they want to follow?
- What autonomy level do they want the agent to follow? Do they want the agent to be fully independent and autonomous, or do they want the agent to check in frequently with the developer?
- ... (any other question that helps you create an implementation plan and learn about the developer and the project)

## 2. Plan
Make a plan for the execution of the initial project. Be sure to keep the developer involved in writing the plan.
Present the plan to the developer. Finally, save the plan in `docs/plans/active/YYYY-MM-DD-get-started.md`.
Keep open follow-up items in a `## TODOs` section at the bottom of that plan file using markdown checkboxes.

## 3. Customize Agent Rules
Review `AGENTS.md` with the developer and customize it to their preferences before scaffolding implementation details.
Language, coding and architecture guidelines should be in line with the plan.
Communication and approach should be in line with the developer / team preference.
Keep it minimal and specific to this repo.

## 4. Execution
Only start this step after explicit developer approval of Step 2 (Plan) and Step 3 (Customize Agent Rules).
If necessary, configure the remote URL of this repo to point to away from the template repo.
Create the minimum docs structure needed by this repo if it does not exist yet:
- `docs/plans/active/`
- `docs/plans/pending/`
- `docs/plans/completed/`
- `docs/decisions/`
Handle template scaffolding based on the chosen project language:
- If the project is **Python**:
  - Move `templates/py/.pre-commit-config.yaml`, `templates/py/.gitignore`, and `templates/py/pyproject.toml` to the repo root.
  - Delete `templates/ts/` and any TS-only template files.
- If the project is **TypeScript**:
  - Move `templates/ts/package.json`, `templates/ts/.gitignore`, `templates/ts/tsconfig.json`, `templates/ts/eslint.config.mjs`, `templates/ts/secretlint.config.mjs`, and `templates/ts/.husky/` to the repo root.
  - Delete `templates/py/` and any Python-only template files.
- If the project is **neither Python nor TypeScript**:
  - Create equivalent starter files for the selected stack in the repo root.
  - Remove both `templates/py/` and `templates/ts/` so only relevant files remain.
Execute the plan. Make sure to keep the developer in the loop and explain what you are doing.
Before handoff, validate that the configured repo works:
- Run the smallest relevant install/setup command for the chosen stack.
- Run the configured lint, typecheck, test, and validation commands.
- If a validation command fails because the project has no source files yet, either add the minimal expected placeholder source/test structure or adjust the command so a fresh scaffold passes.
- Report each validation command and whether it passed.

## 5. Evaluation & Reflection
Look at your work and reflect on what you did. If there were interventions from the developer, make sure to document them in `AGENTS.md` or `README.md`, to avoid making the same mistakes again.
Definition of done checklist:
- `AGENTS.md` instructions are customized to the user.
- Initial project files/scaffolding are created.
- Architecture/stack choice is documented.
- Only the minimum necessary files and documents are present. Avoid overengineering / scope creep.
- Test/lint command is defined and recorded.
- Open TODOs are tracked in the relevant plan file under `docs/plans/` in a `## TODOs` section.
- Decisions should be logged in `docs/decisions/YYYY-MM-DD.md`.
- Open decisions or deferred items are listed explicitly.

## 6. Wrap-up
Confirm with the developer that the project is ready; then you can remove the `get-started` skill from the project.
