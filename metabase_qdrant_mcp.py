#!/usr/bin/env python3
"""
Vedantu Data Analyst MCP Server (Qdrant Edition)
Schemas fetched from Qdrant at startup, SQL execution via Metabase BigQuery.
No local YAML files needed — just Qdrant + Metabase credentials.
"""

import os
import sys
import json
import asyncio
import warnings
from typing import Any
import httpx
from qdrant_client import QdrantClient
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="Api key is used with")
warnings.filterwarnings("ignore", message="Qdrant client version")


def _log(*args):
    print(*args, file=sys.stderr, flush=True)

load_dotenv()

METABASE_URL = os.getenv("METABASE_URL", "https://metabase.vedantu.com").rstrip("/")
METABASE_API_KEY = os.getenv("METABASE_API_KEY", "")
METABASE_SESSION_TOKEN = os.getenv("METABASE_SESSION_TOKEN", "")
METABASE_DEFAULT_DATABASE_ID = int(os.getenv("METABASE_DATABASE_ID", "0"))

QDRANT_HOST = os.getenv("QDRANT_HOST", "")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_SCHEMAS", "bigquery_table_schemas")


# ---------------------------------------------------------------------------
# Schema Store — loads from Qdrant at startup, keyword matches in memory
# ---------------------------------------------------------------------------

def _format_value(value, indent=2) -> str:
    """Format any value (dict, list, or scalar) into readable text lines."""
    prefix = " " * indent
    lines = []
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(_format_value(v, indent + 2))
            else:
                lines.append(f"{prefix}{k}: {v}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(_format_value(item, indent + 2))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{value}")
    return "\n".join(lines)


def _payload_to_context(payload: dict) -> str:
    table_name = payload.get("table_name", "?")
    project_id = payload.get("project_id", "")
    dataset_id = payload.get("dataset_id", "")
    description = str(payload.get("description", "")).strip()

    metabase_db_id = payload.get("metabase_database_id", METABASE_DEFAULT_DATABASE_ID)

    lines = [f"{'='*60}", f"TABLE: {table_name}", f"{'='*60}"]

    if project_id and dataset_id:
        lines.append(f"FULL REFERENCE: `{project_id}.{dataset_id}.{table_name}`")
        database_type = payload.get("database_type", "BigQuery")
        lines.append(f"DATABASE: {database_type}")
    if metabase_db_id:
        lines.append(f"METABASE_DATABASE_ID: {metabase_db_id}")

    if description:
        lines.append(f"\nDESCRIPTION:\n{description}")

    columns_summary = payload.get("columns_summary", [])
    if columns_summary:
        lines.append(f"\nCOLUMNS SUMMARY: {', '.join(str(c) for c in columns_summary)}")

    schema = payload.get("full_schema", {})
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except Exception:
            schema = {}
    if not isinstance(schema, dict):
        schema = {}

    columns = schema.get("columns", [])
    if columns:
        lines.append("\nCOLUMN DETAILS:")
        lines.append("(This is the COMPLETE column list. If a column is not listed here, it does NOT exist.)\n")
        for col in columns:
            col_name = col.get("name", "?")
            col_type = col.get("type", "?")
            col_desc = str(col.get("description", "")).strip()
            pii = " [PII]" if col.get("pii") else ""
            mode = col.get("mode", "")
            mode_str = f" [{mode}]" if mode else ""
            lines.append(f"  - {col_name} ({col_type}{mode_str}){pii}:")
            for desc_line in col_desc.split("\n"):
                desc_line = desc_line.strip()
                if desc_line:
                    lines.append(f"      {desc_line}")
            examples = col.get("example_values", [])
            if examples:
                lines.append(f"      Examples: {examples}")
            if "critical_note" in col:
                note = str(col["critical_note"]).strip()
                lines.append(f"      CRITICAL: {note[:300]}")
            if "bigquery_syntax" in col and isinstance(col["bigquery_syntax"], dict):
                for hint_key, hint_val in col["bigquery_syntax"].items():
                    lines.append(f"      SQL({hint_key}): {hint_val}")

    relationships = payload.get("relationships", [])
    if relationships:
        lines.append("\nRELATIONSHIPS:")
        for rel in relationships:
            if isinstance(rel, dict):
                lines.append(
                    f"  - {rel.get('table')} via {rel.get('join_key')}={rel.get('foreign_key')} "
                    f"({rel.get('type','?')}): {rel.get('description','')}"
                )
            elif isinstance(rel, str):
                lines.append(f"  - {rel}")

    common = payload.get("common_queries", [])
    if common:
        lines.append(f"\nCOMMON QUERIES: {', '.join(str(q) for q in common[:10])}")

    # Dynamically render ALL remaining fields not already handled above
    handled_keys = {
        "table_name", "project_id", "dataset_id", "database_type",
        "metabase_database_id", "description", "columns_summary",
        "full_schema", "relationships", "common_queries",
    }
    for key, value in payload.items():
        if key in handled_keys or not value:
            continue
        header = key.upper().replace("_", " ")
        if isinstance(value, (dict, list)):
            lines.append(f"\n{header}:")
            lines.append(_format_value(value))
        else:
            lines.append(f"\n{header}: {value}")

    return "\n".join(lines)


class QdrantDocsStore:
    """Fetches all schemas from Qdrant at startup, indexes by keyword in memory."""

    def __init__(self):
        self.table_contexts: dict[str, str] = {}
        self.keywords_map: dict[str, list[str]] = {}
        self.loaded = False

    def load_from_qdrant(self):
        if not QDRANT_HOST:
            _log("ERROR: QDRANT_HOST not set")
            return

        try:
            client = QdrantClient(
                url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
                timeout=30,
                https=False,
                prefer_grpc=False,
                check_compatibility=False,
            )

            offset = None
            all_points = []
            while True:
                results, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                all_points.extend(results)
                if next_offset is None:
                    break
                offset = next_offset

            _log(f"Fetched {len(all_points)} schemas from Qdrant ({QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION})")

            for point in all_points:
                try:
                    payload = point.payload
                    table_name = payload.get("table_name", "")
                    if not table_name:
                        continue

                    context_text = _payload_to_context(payload)
                    self.table_contexts[table_name] = context_text

                    kws = [table_name, table_name.replace("_", " ")]
                    if "keywords" in payload:
                        kws.extend([str(k).lower() for k in payload["keywords"]])
                    columns_summary = payload.get("columns_summary", [])
                    for col in columns_summary:
                        col_str = str(col).lower()
                        col_name = col_str.split("(")[0].strip().split(" ")[0].strip()
                        if col_name and len(col_name) > 2:
                            kws.append(col_name)

                    self.keywords_map[table_name] = [k.lower() for k in kws if k]
                    _log(f"  [LOADED] {table_name} ({len(kws)} keywords)")
                except Exception as e:
                    _log(f"  [ERROR] Skipping table {payload.get('table_name', '?')}: {e}")

            if self.table_contexts:
                self.loaded = True
            _log(f"Schemas ready: {len(self.table_contexts)} tables")

        except Exception as e:
            _log(f"ERROR loading from Qdrant: {e}")

    def get_relevant_contexts(self, query: str) -> tuple[list[str], list[str]]:
        query_lower = query.lower()
        matched = []
        for table_name, keywords in self.keywords_map.items():
            for kw in keywords:
                if kw and kw in query_lower:
                    matched.append(table_name)
                    break
        if not matched:
            matched = list(self.table_contexts.keys())
        return matched, [self.table_contexts[t] for t in matched if t in self.table_contexts]


DOCS = QdrantDocsStore()

# ---------------------------------------------------------------------------
# Metabase API client
# ---------------------------------------------------------------------------

def _get_headers() -> dict:
    if METABASE_API_KEY:
        return {"Content-Type": "application/json", "x-api-key": METABASE_API_KEY}
    if METABASE_SESSION_TOKEN:
        return {"Content-Type": "application/json", "X-Metabase-Session": METABASE_SESSION_TOKEN}
    return {"Content-Type": "application/json"}


async def _api_get(path: str) -> Any:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{METABASE_URL}{path}", headers=_get_headers())
        if resp.status_code == 401:
            raise Exception(
                "Session token expired. Refresh METABASE_SESSION_TOKEN.\n"
                "Steps: Log in -> F12 -> Application -> Cookies -> metabase.SESSION"
            )
        resp.raise_for_status()
        return resp.json()


async def _api_post(path: str, payload: dict) -> Any:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{METABASE_URL}{path}", headers=_get_headers(), json=payload)
        if resp.status_code == 401:
            raise Exception(
                "Session token expired. Refresh METABASE_SESSION_TOKEN.\n"
                "Steps: Log in -> F12 -> Application -> Cookies -> metabase.SESSION"
            )
        resp.raise_for_status()
        return resp.json()


async def _auto_detect_database_id() -> int:
    try:
        data = await _api_get("/api/database")
        dbs = data.get("data", data) if isinstance(data, dict) else data
        for db in dbs:
            if db.get("engine") in ("bigquery-cloud-sdk", "bigquery"):
                _log(f"Auto-detected BigQuery database: ID={db['id']} Name={db.get('name')}")
                return int(db["id"])
        _log("WARNING: No BigQuery database found in Metabase")
    except Exception as e:
        _log(f"WARNING: Could not auto-detect database ID: {e}")
    return 0


# ---------------------------------------------------------------------------
# MCP Server — 2 tools only
# ---------------------------------------------------------------------------

server = Server("vedantu-data-analyst")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="answer_data_question",
            description=(
                "MANDATORY FIRST STEP for ANY data question. Returns COMPLETE and "
                "AUTHORITATIVE table schemas, column details, SQL patterns, and rules. "
                "The schema returned IS the source of truth — do NOT query "
                "INFORMATION_SCHEMA, do NOT run SELECT * to explore tables, do NOT "
                "try to discover or verify schemas from the database. "
                "Covers ALL Vedantu data: students, revenue, orders, sessions, "
                "batches, feedback, YouTube, Instagram, admissions, wallet, refunds, coupons. "
                "After calling this, write the final SQL and call execute_sql. "
                "NEVER ask the user which platform — always call this tool first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's data question exactly as they asked it",
                    }
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="execute_sql",
            description=(
                "Execute the FINAL BigQuery SQL query that answers the user's question. "
                "ONLY use this for the actual data query — NEVER for schema exploration, "
                "INFORMATION_SCHEMA, SHOW TABLES, SELECT *, or any metadata queries. "
                "All schema info comes from answer_data_question. "
                "Pass the SQL and the METABASE_DATABASE_ID from the schema context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The final BigQuery SQL that answers the user's question (SELECT only, no schema exploration)",
                    },
                    "database_id": {
                        "type": "integer",
                        "description": "The METABASE_DATABASE_ID from the schema context returned by answer_data_question",
                    },
                },
                "required": ["sql", "database_id"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _format_results(data: Any, max_rows: int = 50) -> str:
    if isinstance(data, dict) and "data" in data:
        cols = [c.get("display_name", c.get("name", "?")) for c in data["data"].get("cols", [])]
        rows = data["data"].get("rows", [])
        total = data["data"].get("rows_truncated", len(rows))
        lines = [f"Columns: {', '.join(cols)}", f"Rows: {len(rows)} (of {total} total)", ""]
        for row in rows[:max_rows]:
            lines.append("  | ".join(str(v) for v in row))
        if len(rows) > max_rows:
            lines.append(f"... and {len(rows) - max_rows} more rows")
        return "\n".join(lines)
    return json.dumps(data, indent=2, default=str)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:

        if name == "answer_data_question":
            if not DOCS.loaded:
                return [TextContent(type="text", text=(
                    "ERROR: No schemas loaded from Qdrant.\n"
                    "Check QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY in your config."
                ))]

            question = arguments["question"]
            matched_tables, contexts = DOCS.get_relevant_contexts(question)

            parts = []
            parts.append(f"=== SCHEMA CONTEXT for: '{question}' (matched: {', '.join(matched_tables)}) ===")
            parts.extend(contexts)

            parts.append(
                "---\n"
                "RULES:\n"
                "- The schema above is COMPLETE and AUTHORITATIVE. Trust it fully.\n"
                "- ONLY columns listed above exist. If a column is not listed, it does NOT exist.\n"
                "- Do NOT run INFORMATION_SCHEMA, SHOW TABLES, SELECT *, or any exploratory query.\n"
                "- Do NOT try to discover or verify column/table names from the database.\n"
                "- If the user asks for data that no column covers, tell them it's not available "
                "in the current schema and suggest the closest alternative.\n"
                "- Write the FINAL SQL using ONLY columns from the schema above.\n"
                "- Call execute_sql with the SQL query AND the METABASE_DATABASE_ID from the schema context.\n"
                "- IMPORTANT: Each table has a METABASE_DATABASE_ID. Pass this value as database_id to execute_sql."
            )

            return [TextContent(type="text", text="\n\n".join(parts))]

        elif name == "execute_sql":
            database_id = arguments.get("database_id") or METABASE_DEFAULT_DATABASE_ID
            if not database_id:
                return [TextContent(type="text", text=(
                    "ERROR: database_id not provided and METABASE_DATABASE_ID not configured.\n"
                    "Use answer_data_question first to get the correct METABASE_DATABASE_ID."
                ))]

            sql_upper = arguments["sql"].upper()
            blocked_patterns = [
                "INFORMATION_SCHEMA",
                "SHOW TABLES",
                "SHOW COLUMNS",
                "DESCRIBE ",
                "SHOW DATABASES",
                "SHOW SCHEMAS",
                "__TABLES__",
            ]
            is_blocked = any(p in sql_upper for p in blocked_patterns)

            if not is_blocked and sql_upper.strip().startswith("SELECT *") and "LIMIT" in sql_upper:
                tokens = sql_upper.split()
                if "WHERE" not in tokens:
                    is_blocked = True

            if is_blocked:
                return [TextContent(type="text", text=(
                    "BLOCKED: Schema exploration queries are not allowed.\n"
                    "You already have the COMPLETE schema from answer_data_question.\n"
                    "If a column doesn't exist in the schema docs, it does NOT exist in the table.\n"
                    "Write the final SQL using ONLY the columns from the schema provided.\n"
                    "If the user asks for something not in the schema, tell them it's not available."
                ))]

            payload = {
                "database": database_id,
                "native": {"query": arguments["sql"]},
                "type": "native",
            }
            data = await _api_post("/api/dataset", payload)
            return [TextContent(type="text", text=f"SQL Results:\n\n{_format_results(data)}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except httpx.HTTPStatusError as e:
        return [TextContent(type="text", text=f"Metabase API error {e.response.status_code}: {e.response.text}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    global METABASE_DEFAULT_DATABASE_ID

    if not METABASE_SESSION_TOKEN and not METABASE_API_KEY:
        _log("ERROR: Set METABASE_SESSION_TOKEN in your config")
        return

    if not QDRANT_HOST:
        _log("ERROR: Set QDRANT_HOST in your config")
        return

    _log("Loading schemas from Qdrant...")
    DOCS.load_from_qdrant()

    if not METABASE_DEFAULT_DATABASE_ID:
        _log("METABASE_DATABASE_ID not set, auto-detecting...")
        METABASE_DEFAULT_DATABASE_ID = await _auto_detect_database_id()

    _log(f"Vedantu Data Analyst MCP (Qdrant) starting - {METABASE_URL}")
    _log(f"Auth: {'API Key' if METABASE_API_KEY else 'Session Token'}")
    _log(f"Default Database ID: {METABASE_DEFAULT_DATABASE_ID or 'FAILED TO DETECT'}")
    _log(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}/{QDRANT_COLLECTION}")
    _log(f"Schemas loaded: {len(DOCS.table_contexts)}")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
