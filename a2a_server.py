"""A2A Server setup for the Analytical Chatbot.

This module configures and creates both HTTP and gRPC A2A servers
for the analytical chatbot agent.
"""

import os
from typing import Optional

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
)

from agent_executor import AnalyticalChatbotExecutor


def create_agent_card(
    host: str = "localhost",
    http_port: int = 8000,
    grpc_port: int = 50051,
) -> AgentCard:
    """Create the A2A AgentCard describing this agent's capabilities.

    Args:
        host: The hostname where the agent is running.
        http_port: The port for the HTTP/JSON-RPC interface.
        grpc_port: The port for the gRPC interface.

    Returns:
        An AgentCard describing the analytical chatbot agent.
    """
    base_url = os.getenv("A2A_BASE_URL", f"http://{host}:{http_port}")
    grpc_url = os.getenv("A2A_GRPC_URL", f"grpc://{host}:{grpc_port}")

    return AgentCard(
        name="Analytical Chatbot",
        description=(
            "An AI-powered analytical assistant that can analyze data, "
            "generate Python code for calculations and visualizations, "
            "execute code in a sandboxed environment, and query databases. "
            "Supports CSV/JSON file uploads and creates charts using Altair."
        ),
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        skills=[
            AgentSkill(
                id="data-analysis",
                name="Data Analysis",
                description=(
                    "Analyze uploaded CSV or JSON files. Compute statistics, "
                    "find patterns, and summarize data."
                ),
                tags=["data", "analysis", "statistics", "csv", "json"],
                examples=[
                    "What is the average salary in the dataset?",
                    "Show me the distribution of ages",
                    "Find the top 10 customers by revenue",
                ],
            ),
            AgentSkill(
                id="code-generation",
                name="Code Generation",
                description=(
                    "Generate Python code for data analysis tasks using "
                    "Polars, Pandas, NumPy, and Altair for visualizations."
                ),
                tags=["python", "code", "polars", "pandas", "numpy"],
                examples=[
                    "Write code to calculate monthly trends",
                    "Generate a scatter plot of price vs quantity",
                ],
            ),
            AgentSkill(
                id="visualization",
                name="Data Visualization",
                description=(
                    "Create charts and visualizations using Altair. "
                    "Supports bar charts, line charts, scatter plots, histograms, and more."
                ),
                tags=["charts", "plots", "visualization", "altair"],
                examples=[
                    "Create a bar chart of sales by region",
                    "Plot the trend of revenue over time",
                    "Make a histogram of customer ages",
                ],
            ),
            AgentSkill(
                id="database-query",
                name="Database Querying",
                description=(
                    "Query the built-in DuckDB database with SQL. "
                    "Available tables: employees, products, sales, customers."
                ),
                tags=["sql", "database", "duckdb", "query"],
                examples=[
                    "Show all employees with salary over 80000",
                    "What are the total sales by product category?",
                    "List customers who made purchases last month",
                ],
            ),
            AgentSkill(
                id="conversation",
                name="Conversation",
                description=(
                    "Have friendly conversations, answer questions about capabilities, "
                    "and provide help with using the chatbot."
                ),
                tags=["chat", "help", "conversation"],
                examples=[
                    "What can you do?",
                    "How do I upload a file?",
                    "Hello!",
                ],
            ),
        ],
        default_input_modes=["text"],
        default_output_modes=["text"],
    )


def create_request_handler() -> DefaultRequestHandler:
    """Create the A2A request handler with the chatbot executor.

    Returns:
        A configured DefaultRequestHandler instance.
    """
    return DefaultRequestHandler(
        agent_executor=AnalyticalChatbotExecutor(),
        task_store=InMemoryTaskStore(),
    )


def create_http_app(
    host: str = "localhost",
    http_port: int = 8000,
    grpc_port: int = 50051,
):
    """Create the HTTP/JSON-RPC A2A application.

    Args:
        host: The hostname where the agent is running.
        http_port: The port for the HTTP interface.
        grpc_port: The port for the gRPC interface.

    Returns:
        A Starlette application with A2A protocol support.
    """
    agent_card = create_agent_card(host, http_port, grpc_port)
    request_handler = create_request_handler()

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Build the base Starlette app
    return a2a_app.build()


def create_grpc_server(
    host: str = "localhost",
    http_port: int = 8000,
    grpc_port: int = 50051,
):
    """Create the gRPC A2A server.

    Args:
        host: The hostname where the agent is running.
        http_port: The port for the HTTP interface (for agent card).
        grpc_port: The port for the gRPC interface.

    Returns:
        A configured gRPC server instance, or None if gRPC is not available.
    """
    try:
        from a2a.server.grpc import A2AGrpcServer
    except ImportError:
        # Fallback implementation if SDK doesn't provide it
        try:
            import grpc.aio
            import a2a.grpc.a2a_pb2_grpc as a2a_pb2_grpc
            from a2a.server.request_handlers import GrpcHandler
            
            class A2AGrpcServer:
                def __init__(self, agent_card, handler, host, port):
                    self.server = grpc.aio.server()
                    servicer = GrpcHandler(agent_card=agent_card, request_handler=handler)
                    a2a_pb2_grpc.add_A2AServiceServicer_to_server(servicer, self.server)
                    self.server.add_insecure_port(f"{host}:{port}")
                
                async def serve(self):
                    await self.server.start()
                    await self.server.wait_for_termination()
        except ImportError:
            print("Warning: gRPC support not available. Install with: pip install a2a-sdk[grpc]")
            return None

    agent_card = create_agent_card(host, http_port, grpc_port)
    request_handler = create_request_handler()

    return A2AGrpcServer(
        agent_card=agent_card,
        handler=request_handler,
        host=host,
        port=grpc_port,
    )


async def run_servers(
    host: str = "0.0.0.0",
    http_port: int = 8000,
    grpc_port: int = 50051,
    enable_grpc: bool = True,
):
    """Run both HTTP and gRPC servers concurrently.

    Args:
        host: The hostname to bind to.
        http_port: The port for the HTTP interface.
        grpc_port: The port for the gRPC interface.
        enable_grpc: Whether to start the gRPC server.
    """
    import asyncio
    import uvicorn

    # Create HTTP app (A2A only)
    http_app = create_http_app(host, http_port, grpc_port)

    # Start HTTP server
    config = uvicorn.Config(
        app=http_app,
        host=host,
        port=http_port,
        log_level="info",
    )
    http_server = uvicorn.Server(config)

    tasks = [http_server.serve()]

    # Optionally start gRPC server
    if enable_grpc:
        print("Initializing gRPC server...", flush=True)
        grpc_server = create_grpc_server(host, http_port, grpc_port)
        if grpc_server:
            tasks.append(grpc_server.serve())
            print(f"gRPC server starting on {host}:{grpc_port}", flush=True)

    print(f"HTTP server starting on {host}:{http_port}", flush=True)
    print(f"Agent card available at: http://{host}:{http_port}/.well-known/agent-card.json", flush=True)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    import asyncio

    host = os.getenv("HOST", "0.0.0.0")
    http_port = int(os.getenv("PORT", "8000"))
    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    enable_grpc = os.getenv("ENABLE_GRPC", "true").lower() in ("1", "true", "yes")

    asyncio.run(run_servers(host, http_port, grpc_port, enable_grpc))
