import os
import logging
import pyodbc
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

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

if __name__ == "__main__":
    print("Starting SQL Server MCP server...")
    mcp.serve()