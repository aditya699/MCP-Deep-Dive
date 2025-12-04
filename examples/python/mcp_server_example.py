"""
MCP Server Example - Production-Ready Template

This example demonstrates:
- Structured output with Pydantic schemas
- Progress reporting via Context
- Tools, Resources, and Prompts
- SSE transport configuration

Author: Aditya Bhatt
"""

import asyncio
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.context import Context


# =============================================================================
# SCHEMAS - Structured Output Definitions
# =============================================================================

class SearchResult(BaseModel):
    """A single search result with metadata."""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL to the resource")
    snippet: str = Field(..., description="Brief description or excerpt")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score 0-1")


class SearchResponse(BaseModel):
    """Complete search response with results and metadata."""
    query: str = Field(..., description="The original search query")
    results: list[SearchResult] = Field(default_factory=list, description="List of search results")
    total_count: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Time taken to search in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the search was performed")


class CalculationResult(BaseModel):
    """Result of a mathematical calculation."""
    expression: str = Field(..., description="The original expression")
    result: float = Field(..., description="The calculated result")
    steps: list[str] = Field(default_factory=list, description="Steps taken to solve")


class CustomerInfo(BaseModel):
    """Customer information resource."""
    customer_id: str
    name: str
    email: str
    plan: str
    created_at: datetime


# =============================================================================
# SERVER INITIALIZATION
# =============================================================================

mcp = FastMCP(
    name="ExampleMCPServer",
    version="1.0.0",
    description="A production-ready MCP server example"
)


# =============================================================================
# TOOLS - Actions the LLM can call
# =============================================================================

@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 10,
    ctx: Context = None
) -> SearchResponse:
    """
    Search the web for information on any topic.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: Context for progress reporting

    Returns:
        SearchResponse with results and metadata
    """
    start_time = datetime.utcnow()

    # Report progress: Starting
    if ctx:
        await ctx.report_progress(0.0, 1.0, "Initializing search...")

    # Simulate API call delay
    await asyncio.sleep(0.5)

    # Report progress: API called
    if ctx:
        await ctx.report_progress(0.3, 1.0, "Querying search API...")

    # Simulate processing
    await asyncio.sleep(0.5)

    # Report progress: Processing
    if ctx:
        await ctx.report_progress(0.7, 1.0, "Processing results...")

    # Generate mock results (replace with actual API call)
    results = [
        SearchResult(
            title=f"Result {i+1} for: {query}",
            url=f"https://example.com/result-{i+1}",
            snippet=f"This is a snippet about {query}. Contains relevant information...",
            relevance_score=1.0 - (i * 0.1)
        )
        for i in range(min(max_results, 5))
    ]

    # Calculate search time
    end_time = datetime.utcnow()
    search_time = (end_time - start_time).total_seconds() * 1000

    # Report progress: Complete
    if ctx:
        await ctx.report_progress(1.0, 1.0, "Search complete!")

    return SearchResponse(
        query=query,
        results=results,
        total_count=len(results),
        search_time_ms=search_time,
        timestamp=end_time
    )


@mcp.tool()
async def calculate(
    expression: str,
    show_steps: bool = False,
    ctx: Context = None
) -> CalculationResult:
    """
    Perform mathematical calculations safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
        show_steps: Whether to show calculation steps
        ctx: Context for progress reporting

    Returns:
        CalculationResult with the answer and optional steps
    """
    if ctx:
        await ctx.report_progress(0.0, 1.0, "Parsing expression...")

    # Safe evaluation (in production, use a proper math parser)
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression")

    if ctx:
        await ctx.report_progress(0.5, 1.0, "Calculating...")

    try:
        result = eval(expression)  # Note: Use safer evaluation in production
    except Exception as e:
        raise ValueError(f"Could not evaluate expression: {e}")

    steps = []
    if show_steps:
        steps = [
            f"Original: {expression}",
            f"Evaluated: {result}"
        ]

    if ctx:
        await ctx.report_progress(1.0, 1.0, "Calculation complete!")

    return CalculationResult(
        expression=expression,
        result=float(result),
        steps=steps
    )


# =============================================================================
# RESOURCES - Data the LLM can read
# =============================================================================

@mcp.resource("customer/{customer_id}")
async def get_customer(customer_id: str) -> CustomerInfo:
    """
    Retrieve customer information by ID.

    Args:
        customer_id: The unique customer identifier

    Returns:
        CustomerInfo with customer details
    """
    # In production, fetch from database
    return CustomerInfo(
        customer_id=customer_id,
        name="John Doe",
        email="john.doe@example.com",
        plan="Premium",
        created_at=datetime.utcnow()
    )


@mcp.resource("metrics/current")
async def get_current_metrics() -> dict:
    """
    Get current system metrics.

    Returns:
        Dictionary with current metrics
    """
    return {
        "active_users": 1234,
        "requests_per_minute": 567,
        "average_response_time_ms": 45.2,
        "error_rate": 0.02,
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# PROMPTS - Reusable instruction templates
# =============================================================================

@mcp.prompt()
def professional_writing_style(content: str) -> str:
    """
    Rewrite content in a professional corporate style.

    Args:
        content: The content to rewrite

    Returns:
        Prompt template for professional rewriting
    """
    return f"""Please rewrite the following content in a professional corporate style.

GUIDELINES:
- Use professional, clear language
- Active voice preferred
- Concise sentences (under 25 words when possible)
- Focus on value and outcomes
- Be specific and action-oriented

AVOID:
- Buzzwords: "synergy", "leverage", "circle back", "low-hanging fruit"
- Passive voice constructions
- Vague statements
- Unnecessary jargon

CONTENT TO REWRITE:
{content}

Please provide the rewritten version below:"""


@mcp.prompt()
def code_review_template(code: str, language: str = "python") -> str:
    """
    Generate a code review prompt.

    Args:
        code: The code to review
        language: Programming language (default: python)

    Returns:
        Prompt template for code review
    """
    return f"""Please review the following {language} code:

```{language}
{code}
```

Analyze for:
1. **Correctness**: Does it work as intended?
2. **Security**: Any vulnerabilities (injection, XSS, etc.)?
3. **Performance**: Any obvious inefficiencies?
4. **Readability**: Is it clear and maintainable?
5. **Best Practices**: Does it follow {language} conventions?

For each issue found, provide:
- Line number (if applicable)
- Issue description
- Suggested fix with code example

End with an overall assessment and priority recommendations."""


@mcp.prompt()
def debugging_assistant(error_message: str, context: str = "") -> str:
    """
    Help debug an error.

    Args:
        error_message: The error message or stack trace
        context: Additional context about what was happening

    Returns:
        Prompt template for debugging assistance
    """
    context_section = f"\nCONTEXT:\n{context}\n" if context else ""

    return f"""I need help debugging this error:

ERROR:
```
{error_message}
```
{context_section}
Please help me:
1. Understand what this error means
2. Identify the likely root cause
3. Suggest specific steps to fix it
4. Recommend how to prevent similar errors

Be specific and provide code examples where helpful."""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Starting MCP Server...")
    print("Endpoint: http://0.0.0.0:8000/sse")
    print("Tools: web_search, calculate")
    print("Resources: customer/{id}, metrics/current")
    print("Prompts: professional_writing_style, code_review_template, debugging_assistant")
    print("-" * 50)

    # Run with SSE transport
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
