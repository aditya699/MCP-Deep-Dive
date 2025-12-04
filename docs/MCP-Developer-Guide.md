# MCP Server Developer Guide

> **Status:** Work in Progress
> **Author:** Aditya Bhatt
> **Last Updated:** December 2025

---

## Table of Contents

1. [Introduction to MCP](#1-introduction-to-mcp)
2. [Core Concepts](#2-core-concepts)
3. [Why Structured Output Matters](#3-why-structured-output-matters)
4. [Server Implementation](#4-server-implementation)
5. [Client Implementation](#5-client-implementation)
6. [Streaming & Progress Reporting](#6-streaming--progress-reporting)
7. [Hybrid Architecture](#7-hybrid-architecture-the-recommended-approach)
8. [Best Practices](#8-best-practices)
9. [Quick Reference](#9-quick-reference)

---

## 1. Introduction to MCP

**MCP (Model Context Protocol)** is an open-source standard for connecting AI applications to external systems. Using MCP, AI applications like Claude or ChatGPT can connect to:

| Component | Description | Analogy |
|-----------|-------------|---------|
| **Data Sources** | Local files, databases | GET endpoints |
| **Tools** | Search engines, calculators, APIs | POST endpoints |
| **Workflows** | Specialized prompts, templates | Instruction sets |

### What MCP Enables

The Model Context Protocol separates the concerns of **providing context** from the **actual LLM interaction**. The Python SDK lets you:

- Build MCP **clients** that can connect to any MCP server
- Create MCP **servers** that expose resources, prompts, and tools
- Use standard **transports**: stdio, SSE, and Streamable HTTP
- Handle all MCP **protocol messages** and lifecycle events

### Official Resources

- **Python SDK:** https://github.com/modelcontextprotocol/python-sdk
- **MCP Inspector:** Interactive developer tool for testing/debugging MCP servers

---

## 2. Core Concepts

### 2.1 Tools

**Tools** are functions the LLM can call to perform actions. Think of them as POST endpoints—they execute code or produce side effects.

```python
@mcp.tool()
async def web_search(query: str, ctx: Context) -> SearchResponse:
    """Search the web for information."""
    results = await search_api(query)
    return SearchResponse(results=results)
```

**Use Tools When:**
- Performing API calls
- Running calculations
- Executing any action with side effects

### 2.2 Resources

**Resources** expose data that the LLM can read. Think of them as GET endpoints—they load information into the LLM's context.

```python
@mcp.resource("customer/{customer_id}")
async def get_customer(customer_id: str) -> Customer:
    """Retrieve customer information."""
    return await database.get_customer(customer_id)
```

**Use Resources When:**
- Exposing customer records
- Providing file contents
- Sharing current metrics or state

### 2.3 Prompts

**Prompts** are reusable templates that guide how the LLM responds. They define instruction patterns.

```python
@mcp.prompt()
def novatech_style(content: str) -> str:
    """Rewrite content in NovaTech corporate style."""
    return f"""Please rewrite in NovaTech style:

GUIDELINES:
- Professional and innovative tone
- Active voice, concise sentences
- Focus on value and outcomes

AVOID:
- "synergy", "leverage", "circle back"
- Passive voice
- Jargon without explanation

CONTENT TO REWRITE:
{content}"""
```

### Tools vs Prompts Decision Guide

| Scenario | Use |
|----------|-----|
| Execute an action (API call, calculation) | **Tool** |
| Return guidelines/instructions | **Prompt** |
| LLM decides when to call | **Tool** |
| User explicitly selects | **Prompt** |
| Has side effects | **Tool** |
| Pure template/instructions | **Prompt** |

**Example:** "Writing Style Guidelines" should be a **Prompt**, not a Tool:
- It's a template, not an action
- User-controlled selection
- Better UX: direct one-step workflow

---

## 3. Why Structured Output Matters

### The Problem with Unstructured Data

Without structure, you have uncertainty about what's in a dict—clients guess, LLMs misunderstand, and bugs hide until production.

```python
# BAD: Unstructured output
def search(query: str) -> dict:
    return {"results": [...], "count": 10}  # What fields? What types?

# GOOD: Structured output with Pydantic
class SearchResponse(BaseModel):
    results: list[SearchResult]
    count: int
    query: str
    search_time_ms: float

def search(query: str) -> SearchResponse:
    return SearchResponse(results=[...], count=10, query=query, search_time_ms=45.2)
```

### Benefits of Structured Output

| Benefit | Description |
|---------|-------------|
| **Catch bugs early** | Pydantic validates immediately instead of failing at runtime |
| **LLMs understand better** | Auto-generated JSON schemas tell LLMs exactly what to expect |
| **Type safety** | IDE autocomplete prevents typos; errors caught before runtime |
| **Self-documenting** | Schema serves as living documentation that can't get outdated |
| **Easy versioning** | Optional fields with defaults let you evolve APIs without breaking clients |

### Implementation Pattern

Every MCP tool should have a corresponding schema:

```
your_mcp_server/
├── server.py          # Tool definitions
├── schemas.py         # Pydantic models
├── utils.py           # Business logic
└── __init__.py
```

**schemas.py:**
```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class SearchResult(BaseModel):
    """A single search result."""
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Text snippet")
    score: float = Field(..., ge=0, le=1, description="Relevance score")

class SearchResponse(BaseModel):
    """Complete search response with metadata."""
    results: list[SearchResult]
    total_count: int
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

> **Important:** Classes without type hints cannot be serialized for structured output. Only classes with properly annotated attributes will be converted to Pydantic models for schema generation and validation.

---

## 4. Server Implementation

### 4.1 Basic Server Setup

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.context import Context

# Initialize server
mcp = FastMCP("MyMCPServer")

@mcp.tool()
async def my_tool(param: str, ctx: Context) -> MyResponse:
    """Tool description for LLM."""
    # Implementation
    return MyResponse(...)

# Run with SSE transport
if __name__ == "__main__":
    mcp.run(transport="sse", port=8000)
```

### 4.2 Transport Options

| Transport | Use Case | Endpoint |
|-----------|----------|----------|
| **stdio** | Local CLI tools, subprocess communication | N/A (stdin/stdout) |
| **SSE** | Web-based, real-time streaming | `http://host:port/sse` |
| **Streamable HTTP** | REST-like, request/response | `http://host:port/` |

### 4.3 Server Lifecycle

```
1. Client opens SSE connection    → GET /sse (200 OK)
2. Client sends initialization    → POST /messages/?session_id=xxx (202 Accepted)
3. Server assigns session ID      → Client is ready
4. Client requests tool list      → Server returns available tools
5. Client calls tools             → Server executes and returns results
```

---

## 5. Client Implementation

### 5.1 Basic Client Connection

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    server_url = "http://localhost:8000/sse"

    async with sse_client(server_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize connection
            await session.initialize()

            # List available tools
            tools_response = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools_response.tools]}")

            # Call a tool
            result = await session.call_tool("web_search", {"query": "AI news"})
            print(f"Result: {result.content}")

asyncio.run(main())
```

### 5.2 Understanding ClientSession

**ClientSession** is the core object that manages the conversation with your MCP server. It handles:

- Protocol handshake (initialization)
- Sending requests (like "call this tool")
- Receiving responses (tool results, progress updates)

Think of it as a "phone call" object—it keeps the connection alive and manages the back-and-forth.

---

## 6. Streaming & Progress Reporting

### 6.1 The Context Object

**Context** is a special messenger object that FastMCP automatically gives your tool so it can send progress updates back to the user while the tool is running.

```python
@mcp.tool()
async def web_search(objective: str, ctx: Context) -> SearchResponse:
    await ctx.report_progress(0.0, 1.0, "Starting search...")

    results = await search_api(objective)
    await ctx.report_progress(0.5, 1.0, "Processing results...")

    processed = process_results(results)
    await ctx.report_progress(1.0, 1.0, "Complete!")

    return SearchResponse(results=processed)
```

### 6.2 Progress Reporting API

```python
await ctx.report_progress(progress, total=None, message=None)
```

| Parameter | Description |
|-----------|-------------|
| `progress` | Current progress value (e.g., 0.5) |
| `total` | Maximum value (usually 1.0) |
| `message` | Status text visible to user |

### 6.3 The Pub/Sub Pattern

Context follows a publish/subscribe pattern:

```
Your tool (Publisher):           User/Client (Subscriber):
  ctx.info("Step 1 done")    →     Receives "Step 1 done"
  ctx.info("Step 2...")      →     Receives "Step 2..."
  ctx.info("Complete!")      →     Receives "Complete!"
```

---

## 7. Hybrid Architecture (The Recommended Approach)

### 7.1 Core Idea: Separate the BRAIN from the HANDS

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐              ┌─────────────────┐          │
│   │     LLM         │              │   MCP Server    │          │
│   │    (BRAIN)      │              │    (HANDS)      │          │
│   │                 │              │                 │          │
│   │  "What tool     │              │  Executes the   │          │
│   │   should I      │              │  actual work    │          │
│   │   call?"        │              │                 │          │
│   └────────┬────────┘              └────────▲────────┘          │
│            │                                │                    │
│            │ Tool decision                  │ Tool execution     │
│            ▼                                │                    │
│   ┌─────────────────────────────────────────┴───────┐           │
│   │                  YOUR CLIENT                     │           │
│   │               (CONTROL CENTER)                   │           │
│   │                                                  │           │
│   │   • Routes LLM decisions to MCP                 │           │
│   │   • Captures real-time progress                  │           │
│   │   • Full visibility into everything              │           │
│   └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Why Hybrid?

| Approach | LLM Decides | Tool Execution | Progress Visibility |
|----------|-------------|----------------|---------------------|
| OpenAI Built-in MCP | ✅ OpenAI | ❌ OpenAI (black box) | ❌ None |
| Direct MCP Only | ❌ You hardcode | ✅ You control | ✅ Full |
| **Hybrid (recommended)** | ✅ LLM | ✅ You control | ✅ Full |

**Best of both worlds:**
- LLM makes smart decisions about WHICH tools
- YOU execute tools and see EVERYTHING

### 7.3 The 5 Key Pieces for Real SSE Streaming

#### Piece 1: Persistent SSE Connection

```python
async with sse_client(server_url) as (read_stream, write_stream):
    # Connection stays open
    # read_stream  → Server pushes data anytime
    # write_stream → You send requests
```

**Why it matters:**
- No polling ("are you done yet?")
- Server pushes TO you when it has updates
- True real-time, not simulated

#### Piece 2: ClientSession Protocol Handler

```python
async with ClientSession(read_stream, write_stream) as session:
    await session.initialize()
```

**Why it matters:**
- Wraps raw streams in MCP protocol logic
- Handles JSON-RPC serialization
- Routes incoming messages to correct handlers

#### Piece 3: Progress Token + Callback Registration

```python
result = await session.call_tool(
    tool_name,
    tool_args,
    meta={"progressToken": f"progress-tool-{idx}"},  # Unique ID
    progress_callback=progress_handler                # Your function
)
```

**Why it matters:**
- `progressToken` tells server WHERE to send updates
- `progress_callback` is YOUR function that fires on each update
- Multiple concurrent calls don't mix up

#### Piece 4: Server-Side Progress Reporting

```python
# In your MCP server
async def search_web(..., ctx: Context):
    await ctx.report_progress(0.3, 1.0, "Calling API...")   # PUSH
    # ... do work ...
    await ctx.report_progress(0.7, 1.0, "Processing...")    # PUSH
    return results
```

**Why it matters:**
- Progress is REAL (based on actual work)
- Not fake (arbitrary timers)
- User sees what's actually happening

#### Piece 5: Async Event Loop Magic

```python
result = await session.call_tool(...)  # "Paused" here
# But event loop keeps running!
# Incoming SSE messages still processed
# Callbacks fire WHILE you wait
```

**Timeline visualization:**
```
await call_tool()     ████████████████████████████████
                           ▲           ▲           ▲
Progress callbacks:       30%        70%        Done!
```

You're "waiting" but still receiving. That's the magic.

### 7.4 Complete Data Flow

```
USER: "What's the latest AI news?"
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Send to LLM                                          │
│         "Here are available tools: [web_search, ...]"        │
│         "User wants: latest AI news"                         │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: LLM responds (streamed)                              │
│         "I'll call web_search with query='AI news 2025'"     │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: YOUR CODE intercepts, calls MCP directly             │
│         session.call_tool("web_search", {"query": "AI news"})│
└──────────────────────────────────────────────────────────────┘
            │
            ├─────────────────────────────────────┐
            ▼                                     ▼
┌────────────────────────┐    ┌─────────────────────────────────┐
│ MCP Server executes    │    │ Progress streams back           │
│                        │    │                                 │
│ report_progress(30%) ──┼───►│ [████░░░░░░░░░░░░░░░░░░░] 30%  │
│ report_progress(70%) ──┼───►│ [██████████████░░░░░░░░░] 70%  │
│                        │    │                                 │
│ return results ────────┼───►│ ✓ Tool complete                 │
└────────────────────────┘    └─────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Send results back to LLM                             │
│         "Here are the search results: [...]"                 │
│         "Please synthesize a final answer"                   │
└──────────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: LLM responds with final answer (streamed to user)    │
│         "Based on recent news, here are the top AI..."       │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Best Practices

### 8.1 Schema Design

```python
# Always use Pydantic models for tool returns
class SearchResponse(BaseModel):
    results: list[SearchResult]
    metadata: SearchMetadata

# Add Field descriptions for LLM understanding
class SearchResult(BaseModel):
    title: str = Field(..., description="The title of the search result")
    url: str = Field(..., description="Direct link to the result")
```

### 8.2 Error Handling

```python
@mcp.tool()
async def risky_operation(param: str, ctx: Context) -> OperationResult:
    try:
        await ctx.report_progress(0.0, 1.0, "Starting...")
        result = await perform_operation(param)
        return OperationResult(success=True, data=result)
    except ExternalAPIError as e:
        return OperationResult(success=False, error=str(e))
```

### 8.3 Progress Reporting Guidelines

| Progress Point | When to Report |
|----------------|----------------|
| 0% | Starting the operation |
| 25-50% | After initial API call |
| 50-75% | During processing |
| 100% | Before returning |

### 8.4 Session Management

- Each tool call may create a new session (OpenAI behavior)
- Design tools to be stateless
- Don't rely on session persistence between calls

---

## 9. Quick Reference

### Key Code Patterns

**Pattern 1: SSE Connection Wrapper**
```python
async with sse_client(url) as (read, write):
    async with ClientSession(read, write) as session:
        # Everything happens here
```

**Pattern 2: Progress Callback**
```python
async def handler(progress, total, message):
    print(f"{progress}/{total}: {message}")

await session.call_tool(..., progress_callback=handler)
```

**Pattern 3: Tool Format Conversion (MCP → OpenAI)**
```python
def mcp_to_openai(tool):
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema
    }
```

**Pattern 4: Hybrid Loop**
```python
# 1. Ask LLM what to do
response = openai.create(tools=openai_tools, ...)

# 2. If LLM wants tools, YOU execute them
for tool_call in detected_tools:
    result = await session.call_tool(...)  # Your MCP connection

# 3. Send results back to LLM for final answer
final = openai.create(input=[...results...])
```

### Debugging Tools

**MCP Inspector:**
```bash
npx @modelcontextprotocol/inspector node path/to/server/index.js
```

Provides:
- Visual interface to browse tools, resources, prompts
- Invoke tools with parameters
- View responses and server logs in real-time

---

## One-Line Summary

> "Let the LLM decide WHAT to do, but YOU control HOW it's done—with full streaming visibility through SSE."

---

*This guide is a work in progress. Contributions welcome!*
