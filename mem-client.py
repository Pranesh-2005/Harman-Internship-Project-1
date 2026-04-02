import asyncio
import json
import os
import re
import sys
import sqlite3
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# CONFIG
MEMORY_DB        = "memory-new.db"
MAX_WORKING      = 800          # chars of working-memory injected per turn
RECENT_EPISODES  = 4            # last N episodes recalled in working memory
MAX_FACTS        = 20           # top-N semantic facts per recall
LLM_MODEL        = "openai/gpt-oss-20b"
EXTRACT_MODEL    = LLM_MODEL
MAX_TOOL_ROUNDS  = 20

# Carry the last assistant response into the next turn so follow-ups like
# "make a chart for those" have the data they reference.
CARRY_LAST_RESPONSE = True


def safe_extract_json_array(text: str) -> list:
    if not text:
        return []

    # Remove markdown fences
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    # Find outermost array bounds
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Last-resort: replace unescaped newlines/tabs inside strings
    candidate_fixed = re.sub(r'(?<!\\)\n', ' ', candidate)
    candidate_fixed = re.sub(r'(?<!\\)\t', ' ', candidate_fixed)
    try:
        return json.loads(candidate_fixed)
    except json.JSONDecodeError:
        return []


# TOOL NAME SANITIZER
_CHANNEL_RE = re.compile(r"<\|.*?\|>.*$")   # strips <|channel|>commentary etc.

def sanitize_tool_name(name: str) -> str:
    """Strip <|channel|>... corruption that some model checkpoints emit."""
    return _CHANNEL_RE.sub("", name).strip()


# DATABASE LAYER
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryDB:
    def __init__(self, path: str = MEMORY_DB):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.cur = self.con.cursor()
        self._ensure_schema()

    def _ensure_schema(self):
        self.cur.executescript("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            user_query   TEXT,
            response     TEXT,
            tools_used   TEXT
        );

        CREATE TABLE IF NOT EXISTS memory_facts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            entity        TEXT    NOT NULL,
            fact          TEXT    NOT NULL,
            source_query  TEXT,
            confidence    REAL    DEFAULT 1.0,
            created_at    TEXT    NOT NULL,
            last_seen     TEXT    NOT NULL,
            recall_count  INTEGER DEFAULT 0,
            UNIQUE(entity, fact)
        );

        CREATE INDEX IF NOT EXISTS idx_facts_entity
            ON memory_facts(entity);
        CREATE INDEX IF NOT EXISTS idx_facts_last_seen
            ON memory_facts(last_seen DESC);
        """)
        self.con.commit()

    def log_episode(self, query: str, response: str, tools: list[str]):
        self.cur.execute(
            "INSERT INTO query_logs (timestamp, user_query, response, tools_used) "
            "VALUES (?, ?, ?, ?)",
            (_now(), query, response, json.dumps(tools))
        )
        self.con.commit()

    def recent_episodes(self, n: int = RECENT_EPISODES) -> list[dict]:
        self.cur.execute(
            "SELECT timestamp, user_query, response FROM query_logs "
            "ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = self.cur.fetchall()
        return [
            {"ts": r[0], "query": r[1], "response": r[2]}
            for r in reversed(rows)
        ]

    def upsert_facts(self, facts: list[dict]):
        ts = _now()
        for f in facts:
            entity    = (f.get("entity") or "").strip()
            fact_text = (f.get("fact")   or "").strip()
            conf      = float(f.get("confidence", 1.0))
            if not entity or not fact_text:
                continue
            self.cur.execute("""
                INSERT INTO memory_facts
                    (entity, fact, source_query, confidence, created_at, last_seen, recall_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(entity, fact) DO UPDATE SET
                    confidence   = MAX(confidence, excluded.confidence),
                    last_seen    = excluded.last_seen,
                    recall_count = recall_count + 1
            """, (entity, fact_text, f.get("source_query", "")[:200], conf, ts, ts))
        self.con.commit()

    def top_facts(self, n: int = MAX_FACTS) -> list[dict]:
        self.cur.execute("""
            SELECT entity, fact, confidence, recall_count
            FROM   memory_facts
            ORDER  BY (recall_count * 0.6 + confidence * 0.4) DESC,
                      last_seen DESC
            LIMIT  ?
        """, (n,))
        return [
            {"entity": r[0], "fact": r[1], "confidence": r[2], "recall_count": r[3]}
            for r in self.cur.fetchall()
        ]

    def search_facts(self, query: str, n: int = 10) -> list[dict]:
        like = f"%{query}%"
        self.cur.execute("""
            SELECT entity, fact, confidence
            FROM   memory_facts
            WHERE  entity LIKE ? OR fact LIKE ?
            ORDER  BY confidence DESC, last_seen DESC
            LIMIT  ?
        """, (like, like, n))
        return [
            {"entity": r[0], "fact": r[1], "confidence": r[2]}
            for r in self.cur.fetchall()
        ]

    def prune_stale_facts(self, keep: int = 500):
        self.cur.execute("""
            DELETE FROM memory_facts
            WHERE id NOT IN (
                SELECT id FROM memory_facts
                ORDER BY (recall_count * 0.6 + confidence * 0.4) DESC
                LIMIT ?
            )
        """, (keep,))
        self.con.commit()


# WORKING MEMORY ASSEMBLER
def build_working_memory(db: MemoryDB, current_query: str) -> str:
    parts = []

    # 1. Keyword-matched semantic facts
    keywords = [w for w in re.split(r"\W+", current_query) if len(w) > 3]
    matched: list[dict] = []
    seen: set[str] = set()

    for kw in keywords:
        for f in db.search_facts(kw, n=5):
            key = f"{f['entity']}|{f['fact']}"
            if key not in seen:
                matched.append(f)
                seen.add(key)

    for f in db.top_facts(n=MAX_FACTS):
        key = f"{f['entity']}|{f['fact']}"
        if key not in seen:
            matched.append(f)
            seen.add(key)
        if len(matched) >= MAX_FACTS:
            break

    if matched:
        lines = [f"- {f['entity']}: {f['fact']}" for f in matched[:MAX_FACTS]]
        parts.append("Known facts from previous sessions:\n" + "\n".join(lines))

    # 2. Recent episodes (Q + brief response snippet)
    episodes = db.recent_episodes(n=RECENT_EPISODES)
    if episodes:
        ep_lines = []
        for ep in episodes:
            short_q = ep["query"][:100]
            short_r = ep["response"][:200].replace("\n", " ")
            ep_lines.append(f"  User asked: {short_q}\n  Assistant: {short_r}")
        parts.append("Recent conversation history:\n" + "\n\n".join(ep_lines))

    working = "\n\n".join(parts)
    if len(working) > MAX_WORKING:
        working = working[:MAX_WORKING] + "\n[trimmed]"

    return working



# MEMORY EXTRACTOR
EXTRACT_SYSTEM = """\
You are a memory extractor for a SQL Server database assistant.

Extract facts worth remembering from the exchange below.
Return ONLY a raw JSON array — no markdown fences, no explanation, nothing else.

Each object must have exactly these keys:
  "entity"     : subject (e.g. "AdventureWorksLT2022", "SalesLT schema", "user")
  "fact"       : short self-contained statement (no newlines, no double-quotes inside)
  "confidence" : 0.0-1.0

Good facts to extract:
- Confirmed table/schema names (e.g. SalesLT.SalesOrderDetail exists)
- Column names, data types, key insights
- User preferences (chart type, sort order)
- Errors seen and their cause
- Aggregated values (top product, sales figures)

Bad facts (skip):
- Raw data rows
- Intermediate SQL steps
- Highly query-specific details that won't generalise

Return [] if nothing is worth keeping.
"""


async def extract_facts(
    openai_client: OpenAI,
    query: str,
    response: str,
    tools_used: list[str],
) -> list[dict]:
    # Strip code blocks and table formatting from the response before extraction
    # so the LLM extractor doesn't produce broken JSON
    clean_response = re.sub(r"```[\s\S]*?```", "[code block]", response)
    clean_response = re.sub(r"\|[^\n]+", "", clean_response)   # strip markdown tables
    clean_response = clean_response[:800]

    prompt = (
        f"User query: {query}\n\n"
        f"Tools used: {', '.join(tools_used) or 'none'}\n\n"
        f"Assistant response:\n{clean_response}"
    )
    try:
        result = openai_client.chat.completions.create(
            model=EXTRACT_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user",   "content": prompt},
            ]
        )
        raw   = result.choices[0].message.content or "[]"
        facts = safe_extract_json_array(raw)

        if not isinstance(facts, list):
            return []

        clean_facts = []
        for f in facts:
            if not isinstance(f, dict):
                continue
            f["entity"] = str(f.get("entity", "")).strip()[:100]
            f["fact"]   = str(f.get("fact",   "")).strip()[:200]
            f["fact"]   = f["fact"].replace('"', "'")   # prevent future JSON breaks
            f["source_query"] = query[:200]
            if f["entity"] and f["fact"]:
                clean_facts.append(f)

        return clean_facts

    except Exception as exc:
        print(f"[memory] extraction error: {exc}")
        return []


# MCP CLIENT
_KNOWN_TOOLS: set[str] = set()

SYSTEM_PROMPT_BASE = """\
You are a helpful assistant that queries SQL Server databases using tools.

CRITICAL RULES:
1. Only call tools by their EXACT name as listed. Never append commentary, \
channel markers, or any suffix to a tool name.
2. Always use the SalesLT schema for this database (e.g. SalesLT.Product, \
SalesLT.SalesOrderDetail). Never guess at schema names.
3. If the user references "those", "that data", or similar, re-use the \
query from the recent conversation history provided below rather than \
querying from scratch.
4. When asked to make a chart, call plot_query_results with the same SQL \
you already ran for the data.
"""


class MCPClient:

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )
        self.db = MemoryDB(MEMORY_DB)
        self._last_response: str = ""   # carry-forward buffer
        print(f"[memory] DB ready → {MEMORY_DB}")

    # connection

    async def connect(self, url: str):
        self._streams_ctx = sse_client(url=url)
        streams = await self._streams_ctx.__aenter__()
        self._session_ctx = ClientSession(*streams)
        self.session = await self._session_ctx.__aenter__()
        await self.session.initialize()
        tools = (await self.session.list_tools()).tools
        _KNOWN_TOOLS.update(t.name for t in tools)
        print(f"[mcp] connected — tools: {sorted(_KNOWN_TOOLS)}")

    async def cleanup(self):
        if hasattr(self, "_session_ctx"):
            await self._session_ctx.__aexit__(None, None, None)
        if hasattr(self, "_streams_ctx"):
            await self._streams_ctx.__aexit__(None, None, None)

    # query processing

    async def process_query(self, query: str) -> str:
        working_mem = build_working_memory(self.db, query)

        system_prompt = SYSTEM_PROMPT_BASE
        if working_mem:
            system_prompt += f"\n\n--- Memory context ---\n{working_mem}\n--- End memory ---"

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

        if CARRY_LAST_RESPONSE and self._last_response:
            messages.append({"role": "assistant", "content": self._last_response})

        messages.append({"role": "user", "content": query})

        tools_resp = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                }
            }
            for t in tools_resp.tools
        ]

        final_text: list[str] = []
        tools_used: list[str] = []

        for round_num in range(MAX_TOOL_ROUNDS):
            print(f"\n--- Round {round_num + 1} ---")

            try:
                resp = self.openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto"
                )
            except Exception as exc:
                print(f"[llm] error: {exc}")
                break

            msg = resp.choices[0].message

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls,
            })

            if msg.content:
                final_text.append(msg.content)
                print("Assistant:", msg.content[:300])

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                raw_name  = tc.function.name
                tool_name = sanitize_tool_name(raw_name)

                if raw_name != tool_name:
                    print(f"[warn] tool name sanitized: {raw_name!r} → {tool_name!r}")
                if tool_name not in _KNOWN_TOOLS:
                    print(f"[warn] unknown tool '{tool_name}' — skipping")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            f"Error: '{tool_name}' is not a valid tool. "
                            f"Valid tools: {sorted(_KNOWN_TOOLS)}"
                        ),
                    })
                    continue

                tools_used.append(tool_name)

                try:
                    tool_args = json.loads(tc.function.arguments)
                except Exception:
                    tool_args = {}

                print(f"\nCalling tool: {tool_name} | Args: {tool_args}")

                try:
                    result = await self.session.call_tool(tool_name, tool_args)
                    if isinstance(result.content, list):
                        result_content = "\n".join(str(x) for x in result.content)
                    else:
                        result_content = str(result.content)
                except Exception as exc:
                    result_content = f"Tool error: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
                })
                print(f"Tool result: {result_content[:200]}")

        final_response = "\n".join(final_text) or "No response"

        self._last_response = final_response[:600]
        self.db.log_episode(query, final_response, tools_used)

        asyncio.create_task(
            self._extract_and_store(query, final_response, tools_used)
        )

        return final_response

    async def _extract_and_store(self, query: str, response: str, tools: list[str]):
        facts = await extract_facts(self.openai_client, query, response, tools)
        if facts:
            self.db.upsert_facts(facts)
            labels = ", ".join(
                f'"{f.get("entity","?")}:{f.get("fact","?")[:40]}"'
                for f in facts[:3]
            )
            print(f"[memory] +{len(facts)} fact(s): {labels}")
        self.db.prune_stale_facts(keep=500)

    # chat loop

    async def chat_loop(self):
        print("\nMCP Memory Client ready.")
        print("Commands: 'memory' = inspect store | 'quit' = exit\n")

        while True:
            try:
                query = input("\nQuery: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit"):
                break
            if query.lower() == "memory":
                self._print_memory()
                continue

            print("\n[INPUT JSON]")
            print(json.dumps({"query": query}, indent=2))

            response = await self.process_query(query)

            print("\n[OUTPUT JSON]")
            print(json.dumps({"response": response}, indent=2))

    def _print_memory(self):
        episodes = self.db.recent_episodes(n=5)
        facts    = self.db.top_facts(n=20)

        print("\n══ MEMORY SNAPSHOT ══")
        print(f"\n— Episodic ({len(episodes)} recent) —")
        for ep in episodes:
            print(f"  [{ep['ts'][:19]}] {ep['query'][:70]}")

        print(f"\n— Semantic ({len(facts)} top facts) —")
        for f in facts:
            print(f"  [{f['entity']}] {f['fact']}  "
                  f"(conf={f['confidence']:.2f}, seen={f['recall_count']})")
        print("══════════════════════")


# ENTRY POINT
async def main():
    if len(sys.argv) < 2:
        print("Usage: python memory_client.py <SSE URL>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())