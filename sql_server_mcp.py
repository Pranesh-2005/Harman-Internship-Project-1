import os
import logging
import pyodbc
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
from datetime import datetime
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("mcp-sqlserver")

load_dotenv()

# SQL Server connection parameters
SQL_SERVER = os.getenv("SQL_SERVER", "DESKTOP-3C8U3VT\\SQLEXPRESS")
SQL_DATABASE = os.getenv("SQL_DATABASE", "AdventureWorksLT2022")
SQL_USERNAME = os.getenv("SQL_USERNAME")  # Optional for SQL Auth
SQL_PASSWORD = os.getenv("SQL_PASSWORD")  # Optional for SQL Auth

mcp = FastMCP("SQL Server Explorer")

def connect():
    """Create SQL Server connection"""
    if SQL_USERNAME and SQL_PASSWORD:
        # SQL Server Authentication
        conn_str = (
            r"DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            f"UID={SQL_USERNAME};"
            f"PWD={SQL_PASSWORD};"
            r"TrustServerCertificate=yes"
        )
    else:
        # Windows Authentication
        conn_str = (
            r"DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            r"Trusted_Connection=yes;"
            r"TrustServerCertificate=yes"
        )
    
    return pyodbc.connect(conn_str, timeout=30)

DANGEROUS_SQL = {
    "DROP", "DELETE", "TRUNCATE", "ALTER",
    "UPDATE", "INSERT", "CREATE",
    "GRANT", "REVOKE", "BACKUP", "RESTORE"
}

MAX_ROWS = 1000

def validate_query(query: str):
    """Validate that query is read-only"""
    q = query.strip().upper()
    
    if not q.startswith(("SELECT", "WITH", "SHOW", "EXPLAIN")):
        raise ValueError("Only read-only queries allowed")
    
    for word in DANGEROUS_SQL:
        if word in q:
            raise ValueError(f"Blocked keyword: {word}")

def enforce_limit(query: str):
    """Add TOP clause if not present"""
    q_upper = query.upper()
    if "TOP" not in q_upper and "SELECT" in q_upper:
        # Insert TOP after SELECT
        query = query.replace("SELECT", f"SELECT TOP {MAX_ROWS}", 1)
    return query

@mcp.prompt(name="generate_sql")
def generate_sql(natural_query: str, schema_info: str) -> str:
    return f"""
You are a senior SQL Server expert.

Convert the following natural language request into a SQL Server (T-SQL) query.

Rules:
- Use SQL Server syntax.
- Use TOP instead of LIMIT.
- Do not generate INSERT, UPDATE, DELETE, DROP.
- Only generate SELECT queries.
- Use schema-qualified table names (e.g., Sales.SalesOrderHeader).
- No explanations, only SQL.

Database schema:
{schema_info}

User request:
{natural_query}
"""

@mcp.tool(
    name="list_databases",
    description="List all SQL Server databases"
)
def list_databases() -> str:
    """List all databases on the SQL Server instance"""
    try:
        conn = connect()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT name
            FROM sys.databases
            WHERE database_id > 4  -- Exclude system databases
            ORDER BY name
        """)
        
        rows = cur.fetchall()
        
        return "\n".join(r[0] for r in rows) or "No user databases found"
    
    except Exception as e:
        logger.exception("list_databases failed")
        return f"Internal error: {e}"
    
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()

@mcp.tool(
    name="list_tables",
    description="List all tables in current database"
)
def list_tables() -> str:
    """List all tables in the current database"""
    try:
        conn = connect()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT TABLE_SCHEMA + '.' + TABLE_NAME as full_table_name
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_SCHEMA, TABLE_NAME
        """)
        
        rows = cur.fetchall()
        
        return "\n".join(r[0] for r in rows) or "No tables found"
    
    except Exception as e:
        logger.exception("list_tables failed")
        return f"Internal error: {e}"
    
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()
@mcp.tool(
    name="table_schema",
    description="Get table schema information"
)
def table_schema(table: str) -> str:
    """Get schema information for a specific table"""
    try:
        conn = connect()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """, (table,))
        
        rows = cur.fetchall()
        
        if not rows:
            return f"No schema found for table: {table}"
        
        result = []
        for row in rows:
            col_name, data_type, nullable, default, max_length = row
            col_info = f"{col_name}: {data_type}"
            if max_length:
                col_info += f"({max_length})"
            if nullable == "NO":
                col_info += " NOT NULL"
            if default:
                col_info += f" DEFAULT {default}"
            result.append(col_info)
        
        return "\n".join(result)
    
    except Exception as e:
        logger.exception("table_schema failed")
        return f"Internal error: {e}"
    
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()

@mcp.tool(
    name="view_table",
    description="View first 10 rows of a table"
)
def view_table(table: str) -> str:
    """View the first 10 rows of a table"""
    try:
        conn = connect()
        cur = conn.cursor()
        
        # Handle schema.table format or add SalesLT schema if not specified
        if '.' not in table:
            # Check if table exists in SalesLT schema first
            cur.execute("""
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'SalesLT' AND TABLE_NAME = ?
            """, (table,))
            
            if cur.fetchone()[0] > 0:
                table = f"SalesLT.{table}"
        
        # Use proper schema.table format with brackets
        if '.' in table:
            schema, table_name = table.split('.', 1)
            query = f"SELECT TOP 10 * FROM [{schema}].[{table_name}]"
        else:
            query = f"SELECT TOP 10 * FROM [{table}]"
            
        cur.execute(query)
        
        rows = cur.fetchall()
        
        if not rows:
            return "No rows found"
        
        # Get column names
        cols = [column[0] for column in cur.description]
        
        result = []
        for row in rows:
            row_dict = dict(zip(cols, row))
            result.append(str(row_dict))
        
        return "\n".join(result)
    
    except Exception as e:
        logger.exception("view_table failed")
        return f"Internal error: {e}"
    
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()

@mcp.tool(
    name="execute_query",
    description="Execute safe SQL query"
)
def execute_query(query: str) -> str:
    """Execute a safe, read-only SQL query"""
    try:
        validate_query(query)
        query = enforce_limit(query)
        
        conn = connect()
        cur = conn.cursor()
        
        # Set query timeout
        cur.execute("SET LOCK_TIMEOUT 30000")  # 30 seconds
        
        cur.execute(query)
        
        rows = cur.fetchall()
        
        if not rows:
            return "No results"
        
        # Get column names
        cols = [column[0] for column in cur.description]
        
        result = []
        for row in rows:
            row_dict = dict(zip(cols, row))
            result.append(str(row_dict))
        
        return "\n".join(result)
    
    except ValueError as e:
        return f"Blocked: {e}"
    
    except Exception as e:
        logger.exception("execute_query failed")
        return f"Internal error: {e}"
    
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()

@mcp.tool(
    name="hello_sqlserver",
    description="Test tool for SQL Server connection"
)
def hello_sqlserver(name: str = "World") -> str:
    """Test tool to verify SQL Server connectivity"""
    logger.info(f"Hello called: {name}")
    
    try:
        conn = connect()
        cur = conn.cursor()
        cur.execute("SELECT @@VERSION")
        version = cur.fetchone()[0]
        
        return f"Hello from SQL Server Explorer, {name}! Connected to: {version[:50]}..."
    
    except Exception as e:
        return f"Hello {name}! Connection failed: {e}"
    
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()

# Visualisation Tools

@mcp.tool(
    name="visualize_data",
    description="Create matplotlib visualization from data"
)
def visualize_data(
    data_json: str,
    chart_type: str = "line",
    title: str = "Data Visualization",
    xlabel: str = "X Axis",
    ylabel: str = "Y Axis",
    figsize: str = "12,6"
) -> str:
    """
    Create a matplotlib visualization from JSON data.
    Saves as PNG file and returns file path.
    
    Args:
        data_json: JSON string with data
        chart_type: Type of chart - line, bar, scatter, pie, histogram
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size as "width,height"
    
    Returns:
        File path to saved PNG chart
    """
    try:
        # Parse data
        data = json.loads(data_json)
        fig_w, fig_h = map(float, figsize.split(','))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        
        # Extract data
        x = data.get("x", [])
        y = data.get("y", [])
        labels = data.get("labels", None)
        
        # Create chart based on type
        if chart_type.lower() == "line":
            ax.plot(x, y, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            
        elif chart_type.lower() == "bar":
            ax.bar(x, y, color='steelblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
        elif chart_type.lower() == "scatter":
            ax.scatter(x, y, s=100, alpha=0.6, edgecolors='navy')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.grid(True, alpha=0.3)
            
        elif chart_type.lower() == "pie":
            colors = plt.cm.Set3(range(len(y)))
            ax.pie(y, labels=labels or x, autopct='%1.1f%%', colors=colors)
            
        elif chart_type.lower() == "histogram":
            ax.hist(y, bins=20, color='steelblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
        else:
            return f"Unknown chart type: {chart_type}"
        
        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chart_type}_{timestamp}.png"
        filepath = CHARTS_DIR / filename
        
        # Save chart
        plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Chart saved: {filepath}")
        return str(filepath)
    
    except json.JSONDecodeError:
        return f"Invalid JSON format"
    except Exception as e:
        logger.exception("visualize_data failed")
        return f"Visualization error: {str(e)}"


@mcp.tool(
    name="plot_query_results",
    description="Execute query and create visualization"
)
def plot_query_results(
    query: str,
    chart_type: str = "bar",
    title: str = "Query Results",
    x_column: str = None,
    y_column: str = None
) -> str:
    """
    Execute a SQL query and create a visualization.
    Returns file path to saved chart.
    """
    try:
        validate_query(query)
        query = enforce_limit(query)
        
        conn = connect()
        cur = conn.cursor()
        cur.execute(query)
        
        rows = cur.fetchall()
        
        if not rows:
            return "No data to visualize"
        
        # Get column names
        cols = [column[0] for column in cur.description]
        
        # Use first two columns if not specified
        if not x_column or x_column not in cols:
            x_column = cols[0]
        if not y_column or y_column not in cols:
            y_column = cols[1] if len(cols) > 1 else cols[0]
        
        # Extract data
        x_data = []
        y_data = []
        
        for row in rows:
            row_dict = dict(zip(cols, row))
            x_val = row_dict.get(x_column)
            y_val = row_dict.get(y_column)
            
            # Convert to numeric
            try:
                if isinstance(y_val, str):
                    y_val = float(y_val)
            except (ValueError, TypeError):
                y_val = 0
            
            x_data.append(str(x_val) if x_val else "")
            y_data.append(y_val)
        
        # Create visualization
        viz_data = {"x": x_data, "y": y_data}
        
        result = visualize_data(
            json.dumps(viz_data),
            chart_type=chart_type,
            title=title,
            xlabel=x_column,
            ylabel=y_column
        )
        
        cur.close()
        conn.close()
        
        return result
    
    except ValueError as e:
        return f"Blocked: {e}"
    except Exception as e:
        logger.exception("plot_query_results failed")
        return f"Error: {str(e)}"


@mcp.tool(
    name="multi_series_plot",
    description="Create multi-series visualization"
)
def multi_series_plot(
    data_json: str,
    chart_type: str = "line",
    title: str = "Multi-Series Plot"
) -> str:
    """
    Create visualization with multiple data series.
    Returns file path to saved chart.
    """
    try:
        data = json.loads(data_json)
        x = data.get("x", [])
        series_dict = data.get("series", {})
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = plt.cm.tab10(range(len(series_dict)))
        
        for (series_name, y_values), color in zip(series_dict.items(), colors):
            if chart_type.lower() == "line":
                ax.plot(x, y_values, marker='o', label=series_name, linewidth=2, color=color)
            elif chart_type.lower() == "bar":
                offset = len(series_dict) // 2
                positions = [i + offset for i in range(len(x))]
                ax.bar(positions, y_values, label=series_name, alpha=0.8, color=color)
            elif chart_type.lower() == "scatter":
                ax.scatter(x, y_values, label=series_name, s=100, alpha=0.6, color=color)
        
        ax.set_xlabel("X Axis", fontsize=12)
        ax.set_ylabel("Y Axis", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_{chart_type}_{timestamp}.png"
        filepath = CHARTS_DIR / filename
        
        plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Multi-series chart saved: {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.exception("multi_series_plot failed")
        return f"Error: {str(e)}"


@mcp.tool(
    name="statistical_plot",
    description="Create statistical plots (boxplot, violin, histogram)"
)
def statistical_plot(
    data_json: str,
    plot_type: str = "boxplot",
    title: str = "Statistical Plot"
) -> str:
    """
    Create statistical visualizations.
    Returns file path to saved chart.
    """
    try:
        import numpy as np
        
        data = json.loads(data_json)
        values = data.get("values", [])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type.lower() == "boxplot":
            ax.boxplot(values, labels=data.get("labels", ["Data"]))
            ax.set_ylabel("Values", fontsize=12)
            
        elif plot_type.lower() == "violin":
            parts = ax.violinplot(values, showmeans=True, showmedians=True)
            ax.set_ylabel("Values", fontsize=12)
            
        elif plot_type.lower() == "histogram":
            ax.hist(values, bins=20, color='steelblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel("Values", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats_{plot_type}_{timestamp}.png"
        filepath = CHARTS_DIR / filename
        
        plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Statistical plot saved: {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.exception("statistical_plot failed")
        return f"Error: {str(e)}"


@mcp.tool(
    name="get_chart_list",
    description="List all generated charts"
)
def get_chart_list() -> str:
    """List all generated charts in the charts directory"""
    try:
        charts = sorted(CHARTS_DIR.glob("*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not charts:
            return "No charts generated yet"
        
        result = ["📊 Generated Charts (Most Recent First):\n"]
        for i, chart in enumerate(charts[:20], 1):  # Show last 20
            size_kb = chart.stat().st_size / 1024
            result.append(f"{i}. {chart.name} ({size_kb:.1f} KB)")
        
        return "\n".join(result)
    
    except Exception as e:
        logger.exception("get_chart_list failed")
        return f"Error: {str(e)}"


@mcp.tool(
    name="open_chart",
    description="Open a chart file in default viewer"
)
def open_chart(filename: str) -> str:
    """Open a chart file"""
    try:
        filepath = CHARTS_DIR / filename
        
        if not filepath.exists():
            return f"Chart not found: {filename}"
        
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            os.startfile(filepath)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(filepath)])
        else:  # Linux
            subprocess.run(["xdg-open", str(filepath)])
        
        return f"✅ Opening chart: {filename}"
    
    except Exception as e:
        logger.exception("open_chart failed")
        return f"Error: {str(e)}"


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the provided mcp server with SSE."""
    sse = SseServerTransport("/messages/")
    
    async def handle_sse(request: Request):
        try:
            logger.info("SSE connection initiated")
            async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
            ) as (read_stream, write_stream):
                logger.info("SSE connection established, starting MCP server")
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )
        except Exception as e:
            logger.error(f"Error in SSE handler: {e}")
            raise
    
    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    # Get the MCP server instance
    mcp_server = mcp._mcp_server
    
    import argparse
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()
    
    
    starlette_app = create_starlette_app(mcp_server, debug=True)
    
    # Run the server
    try:
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise