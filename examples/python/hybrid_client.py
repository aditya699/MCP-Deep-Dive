"""
Hybrid MCP Client - LLM API + Direct MCP Control

Architecture:
- OpenAI API (or any LLM) for intelligent tool selection decisions
- Direct MCP connection for tool execution with REAL-TIME progress visibility
- YOU control everything - no black box

Key Features:
- Real-time progress streaming via SSE
- Full visibility into tool execution
- Colored terminal UI with progress bars
- Conversation history management

Author: Aditya Bhatt
"""

import asyncio
import json
import os
from typing import Optional
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MCP Server configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")

# LLM Model (change as needed)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")

# Conversation history
conversation_history: list[dict] = []


# =============================================================================
# TERMINAL UI - Colors and Formatting
# =============================================================================

class Colors:
    """ANSI escape codes for terminal colors."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Print the startup banner."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("  ╔═══════════════════════════════════════════════════════════╗")
    print("  ║                                                           ║")
    print("  ║   █░█ █▄█ █▄▄ █▀█ █ █▀▄   █▀▄▀█ █▀▀ █▀█                  ║")
    print("  ║   █▀█ ░█░ █▄█ █▀▄ █ █▄▀   █░▀░█ █▄▄ █▀▀                  ║")
    print("  ║                                                           ║")
    print("  ║        Hybrid MCP Client - LLM + Direct MCP Control       ║")
    print("  ╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")


def print_divider(char: str = "─", width: int = 60):
    """Print a divider line."""
    print(f"{Colors.GRAY}{char * width}{Colors.END}")


def print_status(msg: str):
    """Print a status message in yellow."""
    print(f"{Colors.YELLOW}  ● {msg}{Colors.END}")


def print_success(msg: str):
    """Print a success message in green."""
    print(f"{Colors.GREEN}  ✓ {msg}{Colors.END}")


def print_error(msg: str):
    """Print an error message in red."""
    print(f"{Colors.RED}  ✗ {msg}{Colors.END}")


def print_tool_event(msg: str):
    """Print a tool-related message in blue."""
    print(f"{Colors.BLUE}  ⚡ {msg}{Colors.END}")


def print_assistant_prefix():
    """Print the assistant response prefix."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}  Assistant:{Colors.END} ", end="", flush=True)


def print_user_prompt() -> str:
    """Print the user input prompt and get input."""
    return input(f"\n{Colors.BOLD}  You:{Colors.END} ").strip()


def format_progress_bar(
    progress: float,
    total: float,
    message: Optional[str] = None
) -> str:
    """Format a colored progress bar."""
    percentage = int((progress / total) * 100) if total > 0 else 0
    bar_length = 25
    filled = int(bar_length * progress / total) if total > 0 else 0

    # Color gradient based on progress
    if percentage < 33:
        bar_color = Colors.RED
    elif percentage < 66:
        bar_color = Colors.YELLOW
    else:
        bar_color = Colors.GREEN

    bar = f"{bar_color}{'█' * filled}{Colors.GRAY}{'░' * (bar_length - filled)}{Colors.END}"
    status_msg = f" {Colors.DIM}{message}{Colors.END}" if message else ""

    return f"  {Colors.CYAN}Progress:{Colors.END} [{bar}] {percentage:3d}%{status_msg}"


# =============================================================================
# COMMANDS
# =============================================================================

def show_help():
    """Display available commands."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}  Available Commands:{Colors.END}")
    print_divider()
    print(f"  {Colors.BOLD}/help{Colors.END}     {Colors.GRAY}or{Colors.END} {Colors.BOLD}/?{Colors.END}      Show this help message")
    print(f"  {Colors.BOLD}/history{Colors.END}  {Colors.GRAY}or{Colors.END} {Colors.BOLD}/h{Colors.END}      Show conversation history")
    print(f"  {Colors.BOLD}/clear{Colors.END}    {Colors.GRAY}or{Colors.END} {Colors.BOLD}/c{Colors.END}      Clear conversation history")
    print(f"  {Colors.BOLD}/tools{Colors.END}    {Colors.GRAY}or{Colors.END} {Colors.BOLD}/t{Colors.END}      Show loaded MCP tools")
    print(f"  {Colors.BOLD}exit{Colors.END}      {Colors.GRAY}or{Colors.END} {Colors.BOLD}quit{Colors.END}    Exit the chatbot")
    print_divider()
    print()


def show_history():
    """Display conversation history."""
    if not conversation_history:
        print_status("No conversation history yet.")
        return

    print(f"\n{Colors.CYAN}{Colors.BOLD}  Conversation History ({len(conversation_history)} exchanges):{Colors.END}")
    print_divider()

    for i, exchange in enumerate(conversation_history, 1):
        user_msg = exchange.get("user", "")[:100]
        assistant_msg = exchange.get("assistant", "")[:100]

        print(f"  {Colors.GRAY}[{i}]{Colors.END}")
        print(f"  {Colors.BOLD}You:{Colors.END} {user_msg}{'...' if len(exchange.get('user', '')) > 100 else ''}")
        print(f"  {Colors.GREEN}Assistant:{Colors.END} {assistant_msg}{'...' if len(exchange.get('assistant', '')) > 100 else ''}")
        print()

    print_divider()


def clear_history():
    """Clear conversation history."""
    global conversation_history
    conversation_history = []
    print_success("Conversation history cleared.")


def show_tools(tools: list):
    """Display loaded MCP tools."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}  Loaded MCP Tools ({len(tools)}):{Colors.END}")
    print_divider()

    for tool in tools:
        print(f"  {Colors.BLUE}●{Colors.END} {Colors.BOLD}{tool.name}{Colors.END}")
        if tool.description:
            desc = tool.description[:80]
            print(f"    {Colors.GRAY}{desc}{'...' if len(tool.description) > 80 else ''}{Colors.END}")

    print_divider()
    print()


# =============================================================================
# MCP <-> OPENAI CONVERSION
# =============================================================================

def mcp_to_openai_tool(tool) -> dict:
    """Convert MCP tool definition to OpenAI function format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}}
        }
    }


# =============================================================================
# CORE QUERY PROCESSING
# =============================================================================

async def process_query(
    query: str,
    session: ClientSession,
    openai_tools: list[dict]
) -> str:
    """
    Process a user query using Hybrid approach:
    1. Ask LLM what tools to use
    2. Execute tools via direct MCP connection (with progress)
    3. Send results back to LLM for synthesis

    Args:
        query: User's query string
        session: Active MCP ClientSession
        openai_tools: Tools in OpenAI format

    Returns:
        Final assistant response
    """
    print_status("Processing your request...")

    # Build messages for OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
        {"role": "user", "content": query}
    ]

    # Step 1: Ask LLM what to do
    try:
        response = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
            stream=True
        )
    except Exception as e:
        print_error(f"OpenAI API Error: {e}")
        return ""

    # Collect streamed response
    tool_calls_data: dict[int, dict] = {}  # index -> {id, name, arguments}
    assistant_response = ""
    assistant_started = False

    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue

        # Collect tool calls
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_data:
                    tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}

                if tc.id:
                    tool_calls_data[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_data[idx]["name"] = tc.function.name
                        print_tool_event(f"LLM requesting tool: {Colors.BOLD}{tc.function.name}{Colors.END}")
                    if tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments

        # Stream text content
        if delta.content:
            if not assistant_started:
                print_assistant_prefix()
                assistant_started = True
            print(delta.content, end="", flush=True)
            assistant_response += delta.content

    # Step 2: If tools were requested, execute them with progress
    if tool_calls_data:
        print(f"\n\n{Colors.CYAN}  ╭{'─' * 56}╮{Colors.END}")
        print(f"{Colors.CYAN}  │{Colors.END}  {Colors.BOLD}Tool Execution{Colors.END} - {len(tool_calls_data)} tool(s) to run{' ' * 23}{Colors.CYAN}│{Colors.END}")
        print(f"{Colors.CYAN}  ╰{'─' * 56}╯{Colors.END}\n")

        all_tool_results = []

        for idx, tc_data in sorted(tool_calls_data.items()):
            tool_name = tc_data["name"]
            tool_id = tc_data["id"]

            try:
                tool_args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                tool_args = {}

            print(f"  {Colors.BLUE}[{idx+1}/{len(tool_calls_data)}]{Colors.END} {Colors.BOLD}{tool_name}{Colors.END}")

            # Format arguments
            if tool_args:
                args_str = json.dumps(tool_args, indent=2)
                for line in args_str.split('\n'):
                    print(f"      {Colors.GRAY}{line}{Colors.END}")
            print()

            # Progress callback - this is where the magic happens!
            async def progress_handler(
                progress: float,
                total: Optional[float],
                message: Optional[str]
            ):
                """Handle REAL-TIME progress notifications from MCP server."""
                if total and total > 0:
                    progress_line = format_progress_bar(progress, total, message)
                    print(f"\r{' ' * 80}\r{progress_line}", end="", flush=True)
                elif message:
                    print(f"\r{' ' * 80}\r  {Colors.CYAN}Progress:{Colors.END} {message}", end="", flush=True)

            # Execute via OUR MCP connection with streaming progress!
            try:
                result = await session.call_tool(
                    tool_name,
                    tool_args,
                    meta={"progressToken": f"progress-{tool_name}-{idx}"},
                    progress_callback=progress_handler
                )
                print(f"\r{' ' * 80}\r", end="")  # Clear progress line
                print_success(f"Tool complete: {tool_name}")
                print()

                # Extract result text
                result_text = ""
                if result.content:
                    if hasattr(result.content[0], 'text'):
                        result_text = result.content[0].text
                    else:
                        result_text = str(result.content[0])

                all_tool_results.append({
                    "tool_call_id": tool_id,
                    "tool": tool_name,
                    "result": result_text
                })

            except Exception as e:
                print(f"\r{' ' * 80}\r", end="")
                print_error(f"Tool error: {e}")
                all_tool_results.append({
                    "tool_call_id": tool_id,
                    "tool": tool_name,
                    "result": f"Error: {str(e)}"
                })

        # Step 3: Send results back to LLM for synthesis
        print_divider()
        print_status("Synthesizing final response...")

        # Build follow-up messages with tool results
        follow_up_messages = messages.copy()

        # Add assistant's tool call message
        follow_up_messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc_data["id"],
                    "type": "function",
                    "function": {
                        "name": tc_data["name"],
                        "arguments": tc_data["arguments"]
                    }
                }
                for tc_data in tool_calls_data.values()
            ]
        })

        # Add tool results
        for tr in all_tool_results:
            follow_up_messages.append({
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["result"]
            })

        # Get final answer from LLM
        try:
            final_response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=follow_up_messages,
                stream=True
            )

            print_assistant_prefix()
            final_text = ""
            for chunk in final_response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
                    final_text += delta.content

            print("\n")
            assistant_response = final_text

        except Exception as e:
            print_error(f"Error getting final response: {e}")

    else:
        # No tools called, just text response
        if assistant_response:
            print("\n")

    # Store in conversation history
    if assistant_response:
        conversation_history.append({
            "user": query,
            "assistant": assistant_response
        })

    return assistant_response


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main hybrid client - connects to MCP server directly."""

    print_banner()
    print_status(f"Connecting to MCP server at {MCP_SERVER_URL}...")

    try:
        async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize connection
                await session.initialize()
                print_success("Connected to MCP server!")

                # Get tools from OUR server
                tools_response = await session.list_tools()
                tools = tools_response.tools
                tool_names = ', '.join([t.name for t in tools])
                print_success(f"Loaded {len(tools)} tools: {Colors.CYAN}{tool_names}{Colors.END}")

                # Convert to OpenAI format
                openai_tools = [mcp_to_openai_tool(t) for t in tools]

                print()
                print_divider()
                print(f"  {Colors.GRAY}Type {Colors.BOLD}/help{Colors.END}{Colors.GRAY} for commands or start chatting{Colors.END}")
                print_divider()

                # Interactive loop
                while True:
                    try:
                        user_input = print_user_prompt()

                        # Handle commands
                        if user_input.lower() in ['quit', 'exit', 'q', '/exit']:
                            print(f"\n{Colors.CYAN}  Goodbye!{Colors.END}\n")
                            break

                        elif user_input.lower() in ['/help', '/?']:
                            show_help()
                            continue

                        elif user_input.lower() in ['/history', '/h']:
                            show_history()
                            continue

                        elif user_input.lower() in ['/clear', '/c']:
                            clear_history()
                            continue

                        elif user_input.lower() in ['/tools', '/t']:
                            show_tools(tools)
                            continue

                        # Skip empty inputs
                        if not user_input:
                            continue

                        # Process the query
                        await process_query(user_input, session, openai_tools)

                    except KeyboardInterrupt:
                        print(f"\n\n{Colors.CYAN}  Goodbye!{Colors.END}\n")
                        break

    except ConnectionRefusedError:
        print_error(f"Could not connect to MCP server at {MCP_SERVER_URL}")
        print(f"  {Colors.GRAY}Make sure the server is running first.{Colors.END}")
    except Exception as e:
        print_error(f"Connection error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
