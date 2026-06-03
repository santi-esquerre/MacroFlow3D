# MCP setup guide

Minimal MCP (Model Context Protocol) configuration for MacroFlow3D.

---

## Scope

This project uses three MCP servers (configured in `.mcp.json`):

1. **GitHub** — issue/PR operations, code search
2. **Fetch** — web page fetching
3. **Context7** — up-to-date library/framework documentation (CUDA, PETSc, SLEPc, CMake, etc.)

The project `.mcp.json` is committed and shared. Auth tokens are provided via environment variables — never hardcoded in tracked files.

---

## Project-level config (`.mcp.json`)

The repo ships with `.mcp.json` pre-configured for all three servers. No manual editing needed unless adding a new server.

---

## Required environment variables

Set these in your shell profile (e.g., `~/.bashrc` or `~/.zshrc`):

```bash
# GitHub — required for gh MCP (issues, PRs, code search)
export GITHUB_TOKEN="<your-github-pat>"

# Context7 — required for library documentation lookups
export CONTEXT7_API_KEY="<your-context7-key>"
```

### GitHub token

- Scopes: `repo` (read/write for issues, PRs, code)
- Generate at: <https://github.com/settings/tokens>

### Context7 API key

- Get a key at: <https://context7.com>
- Enables documentation lookups for CUDA, PETSc, SLEPc, CMake, MPI, and other HPC libraries

---

## What each server enables

### GitHub MCP

- Create/read/update issues and PRs
- Search code in the repository
- Read PR review comments
- Fetch CI status

### Fetch MCP

- Fetch and read web pages
- Useful for looking up external references, documentation, error messages

### Context7 MCP

- Query up-to-date documentation for libraries and frameworks
- Particularly useful for CUDA runtime API, PETSc/SLEPc API, CMake modules
- `DEFAULT_MINIMUM_TOKENS` is set to `10000` for sufficient context

---

## VS Code Copilot

VS Code Copilot indexes the workspace directly and has its own MCP configuration surface (`.vscode/mcp.json`, gitignored). The project `.mcp.json` is for Claude Code CLI; VS Code users configure separately if needed.

---

## What NOT to do

- Do not add MCP secrets to tracked repository files.
- Do not add MCP configs that require specific machine paths.
- Do not add MCPs for databases, browsers, or devtools — this is a scientific HPC project.
- Do not add more servers unless there is a clear, justified need.

---

## Additional MCPs considered and rejected

| MCP | Reason not included |
|-----|-------------------|
| Brave Search | Fetch MCP + Context7 cover documentation lookup needs |
| Filesystem | Claude Code has native file access; redundant |
| Memory | Claude Code has native memory; redundant |
| Sequential Thinking | Adds complexity without clear benefit over plan mode |
| Docker/K8s | Not relevant — direct WSL + CUDA environment |

---

## Codex App

The `.codex/config.toml` handles web search via the `web_search` setting:

- `"cached"` for normal work
- `"live"` for research profiles

No additional MCP configuration is needed for Codex.

---

## Related

- `.codex/config.toml` — Codex profiles
- `.vscode/` is gitignored — MCP config stays local
