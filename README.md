# MCP-Deep-Dive

> **Production-Grade MCP (Model Context Protocol) Developer Guide**

A comprehensive guide to building MCP servers and clients with real-time streaming, structured output, and hybrid architectures.

---

## What is MCP?

**MCP (Model Context Protocol)** is an open-source standard for connecting AI applications to external systems. It enables AI applications like Claude or ChatGPT to connect to:

- **Data Sources** - Local files, databases, APIs
- **Tools** - Search engines, calculators, external services
- **Workflows** - Specialized prompts and instruction templates

## Repository Structure

```
MCP-Deep-Dive/
├── docs/
│   └── MCP-Developer-Guide.md    # Comprehensive developer guide
├── examples/
│   └── python/
│       ├── mcp_server_example.py # Production-ready MCP server template
│       ├── hybrid_client.py      # Hybrid LLM + Direct MCP client
│       ├── requirements.txt      # Python dependencies
│       └── .env.example          # Environment variables template
├── MCP_Checklist.docx            # Implementation checklist (source notes)
└── README.md                     # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd examples/python
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the MCP Server

```bash
python mcp_server_example.py
```

Server starts at `http://localhost:8000/sse`

### 4. Run the Hybrid Client

```bash
python hybrid_client.py
```

## Key Concepts

### The Hybrid Architecture

The recommended approach separates the **BRAIN** (LLM decisions) from the **HANDS** (tool execution):

```
┌─────────────────┐              ┌─────────────────┐
│     LLM         │              │   MCP Server    │
│    (BRAIN)      │              │    (HANDS)      │
│  "What tool?"   │              │  Executes work  │
└────────┬────────┘              └────────▲────────┘
         │                                │
         │ Tool decision                  │ Tool execution
         ▼                                │
┌─────────────────────────────────────────┴───────┐
│               YOUR CLIENT                        │
│            (CONTROL CENTER)                      │
│   • Routes LLM decisions to MCP                 │
│   • Captures real-time progress                  │
│   • Full visibility into everything              │
└──────────────────────────────────────────────────┘
```

### Why Hybrid?

| Approach | LLM Decides | Tool Execution | Progress Visibility |
|----------|-------------|----------------|---------------------|
| Built-in MCP | ✅ | ❌ Black box | ❌ None |
| Direct MCP | ❌ Hardcoded | ✅ You control | ✅ Full |
| **Hybrid** | ✅ | ✅ You control | ✅ Full |

## Documentation

See the full [MCP Developer Guide](docs/MCP-Developer-Guide.md) for:

- Core concepts (Tools, Resources, Prompts)
- Structured output with Pydantic
- Progress reporting and streaming
- Complete architecture diagrams
- Best practices and patterns

## Example Features

### MCP Server (`mcp_server_example.py`)

- **Tools**: `web_search`, `calculate`
- **Resources**: `customer/{id}`, `metrics/current`
- **Prompts**: `professional_writing_style`, `code_review_template`, `debugging_assistant`
- **Progress Reporting**: Real-time updates via Context

### Hybrid Client (`hybrid_client.py`)

- Real-time progress bars during tool execution
- Colored terminal UI
- Conversation history management
- Commands: `/help`, `/history`, `/clear`, `/tools`

## Resources

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://modelcontextprotocol.io)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector) - Interactive debugging tool

## Status

> **Work in Progress** - This guide is actively being developed.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Author

**Aditya Bhatt** - 2025
