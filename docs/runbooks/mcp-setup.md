# MCP setup guide

Minimal MCP (Model Context Protocol) configuration for MacroFlow3D.

---

## Scope

This project uses at most three MCP servers:

1. **GitHub** — issue/PR operations, code search
2. **Technical docs** — project documentation context
3. **Technical search** (optional) — web search for CUDA/PETSc/HPC references

MCP config is **not** stored in the repository because it requires per-user auth tokens and machine-specific paths.

---

## 1. GitHub MCP

### VS Code (`.vscode/mcp.json`)

Create `.vscode/mcp.json` in your local workspace (gitignored):

```json
{
  "servers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
```

Token requirements:

- `repo` scope (read/write for issues, PRs, code)
- Generate at: <https://github.com/settings/tokens>

### What it enables

- Create/read/update issues and PRs
- Search code in the repository
- Read PR review comments
- Fetch CI status

---

## 2. Technical docs MCP

For providing project documentation as context to LLMs.

### Option A: filesystem server

```json
{
  "servers": {
    "docs": {
      "command": "npx",
      "args": [
        "-y", "@modelcontextprotocol/server-filesystem",
        "/path/to/MacroFlow3D/docs",
        "/path/to/MacroFlow3D/AGENTS.md",
        "/path/to/MacroFlow3D/ARCHITECTURE.md"
      ]
    }
  }
}
```

### Option B: rely on workspace context

VS Code Copilot already indexes the workspace. The filesystem MCP is only useful if you need to expose docs to external tools.

---

## 3. Technical search MCP (optional)

For looking up CUDA, PETSc, SLEPc, or HPC documentation:

```json
{
  "servers": {
    "search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "<your-key>"
      }
    }
  }
}
```

Alternative: use the Codex `web_search` setting (already configured in `.codex/config.toml`).

---

## What NOT to do

- Do not add MCP secrets to the repository.
- Do not add MCP configs that require specific machine paths.
- Do not add MCPs for databases, browsers, or devtools — this is a scientific HPC project.
- Do not add more than the three MCPs listed above unless there is a clear need.

---

## Codex App

The `.codex/config.toml` already handles web search via the `web_search` setting:

- `"cached"` for normal work
- `"live"` for research profiles

No additional MCP configuration is needed in `.codex/config.toml` for the current workflow.

---

## Related

- `.codex/config.toml` — Codex profiles
- `.vscode/` is gitignored — MCP config stays local
