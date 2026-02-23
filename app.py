import os
import logging
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("mcp-postgres")


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")


mcp = FastMCP("Postgres Explorer")

def connect():

    return psycopg2.connect(
        dsn=DATABASE_URL,
        connect_timeout=30
    )


DANGEROUS_SQL = {
    "DROP", "DELETE", "TRUNCATE", "ALTER",
    "UPDATE", "INSERT", "CREATE",
    "GRANT", "REVOKE", "VACUUM"
}

MAX_ROWS = 1000


def validate_query(query: str):

    q = query.strip().upper()

    if not q.startswith(("SELECT", "WITH", "SHOW", "EXPLAIN")):
        raise ValueError("Only read-only queries allowed")

    for word in DANGEROUS_SQL:
        if word in q:
            raise ValueError(f"Blocked keyword: {word}")


def enforce_limit(query: str):

    if "LIMIT" not in query.upper():
        query = query.rstrip(";") + f" LIMIT {MAX_ROWS}"

    return query


@mcp.tool(
    name="list_databases",
    description="List all PostgreSQL databases"
)
def list_databases() -> str:

    try:
        conn = connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT datname
            FROM pg_database
            WHERE datistemplate = false;
        """)

        rows = cur.fetchall()

        return "\n".join(r[0] for r in rows) or "No databases found"

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
    description="List all public tables in database"
)
def list_tables(db_name: str) -> str:

    try:
        conn = connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_catalog = %s;
        """, (db_name,))

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
    description="Get table schema"
)
def table_schema(db_name: str, table: str) -> str:

    try:
        conn = connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_catalog = %s
            AND table_name = %s;
        """, (db_name, table))

        rows = cur.fetchall()

        if not rows:
            return f"No schema for {table}"

        return "\n".join(
            f"{c}: {t}" for c, t in rows
        )

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
def view_table(db_name: str, table: str) -> str:

    try:
        conn = connect()
        cur = conn.cursor()

        query = sql.SQL("SELECT * FROM {} LIMIT 10").format(
            sql.Identifier(table)
        )

        cur.execute(query)

        rows = cur.fetchall()

        cols = [d[0] for d in cur.description]

        if not rows:
            return "No rows found"

        return "\n".join(
            str(dict(zip(cols, r)))
            for r in rows
        )

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
def execute_query(db_name: str, query: str) -> str:

    try:
        validate_query(query)

        query = enforce_limit(query)

        conn = connect()
        cur = conn.cursor()

        cur.execute("SET statement_timeout = 3000")

        cur.execute(query)

        rows = cur.fetchall()

        if not rows:
            return "No results"

        cols = [d[0] for d in cur.description]

        return "\n".join(
            str(dict(zip(cols, r)))
            for r in rows
        )

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


# -------------------------------------------------


@mcp.tool(
    name="hello_postgres",
    description="Test tool for Postgres server"
)
def hello_postgres(name: str = "World") -> str:

    logger.info(f"Hello called: {name}")

    return f"Hello from Postgres Explorer, {name}!"


if __name__ == "__main__":

    print("Starting Postgres MCP server...")

    mcp.serve()