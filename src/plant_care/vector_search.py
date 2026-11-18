"""Lakebase pgvector search for plant care chunks."""

from __future__ import annotations

import os
import re
import urllib.parse
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import psycopg
from databricks.sdk import WorkspaceClient
from loguru import logger

from plant_care.config import ProjectConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

EMBEDDING_DIMENSION = 1024
CHUNK_SOURCE_COLUMNS = {"chunk_id", "slug", "url", "section", "text"}


def _query_terms(query: str) -> list[str]:
    """Split a query into distinct lowercase words."""
    return list(dict.fromkeys(re.findall(r"\w+", query.lower())))
TYPED_TEXT_FIELDS = ["category", "difficulty", "family"]
TYPED_BOOL_FIELDS = ["pet_safe_cat", "pet_safe_dog"]
ALLOWED_FILTERS = {"slug", "tags", *TYPED_TEXT_FIELDS, *TYPED_BOOL_FIELDS}
CHUNK_COLUMNS = [
    "id",
    "slug",
    "url",
    "section",
    "text",
    "tags",
    *TYPED_TEXT_FIELDS,
    *TYPED_BOOL_FIELDS,
    "metadata",
]


class PlantCarePgVectorManager:
    """Store and search plant care chunk embeddings in a Lakebase Postgres database."""

    table_name = "plant_care_embeddings"
    source_table_name = "plant_care_chunks"

    def __init__(
        self,
        config: ProjectConfig,
        spark: SparkSession | None = None,
        *,
        workspace: WorkspaceClient | None = None,
    ) -> None:
        self.config = config
        self.spark = spark
        self.workspace = workspace
        self.lakebase_instance = config.lakebase_instance
        self.embedding_model = config.embedding_endpoint
        self.embedding_service = config.embedding_service
        self._connection_string: str | None = None

    def _get_connection_string(self) -> str:
        """Create a short-lived Lakebase connection string."""
        if self._connection_string:
            return self._connection_string

        workspace = self.workspace or WorkspaceClient()
        postgres_api = workspace.postgres
        endpoint_name = os.environ.get("LAKEBASE_ENDPOINT")
        if endpoint_name:
            host = os.environ.get("PGHOST")
            database = os.environ.get("PGDATABASE", "databricks_postgres")
        else:
            project_name = f"projects/{self.lakebase_instance}"
            branch = next(iter(postgres_api.list_branches(parent=project_name)))
            endpoint = next(iter(postgres_api.list_endpoints(parent=branch.name)))
            endpoint_name = endpoint.name
            host = endpoint.status.hosts.host
            database = "databricks_postgres"
        credential = postgres_api.generate_database_credential(endpoint=endpoint_name)
        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        username = urllib.parse.quote_plus(client_id or workspace.current_user.me().user_name)

        self._connection_string = (
            f"postgresql://{username}:{credential.token}@{host}:5432/{database}?sslmode=require"
        )
        return self._connection_string

    def connection(self) -> psycopg.Connection:
        """Open a Lakebase connection using a fresh workspace credential."""
        self._connection_string = None
        return psycopg.connect(self._get_connection_string())

    def ensure_schema(self) -> None:
        """Enable pgvector and create the embedding table and indexes."""
        with self.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    slug TEXT NOT NULL,
                    url TEXT,
                    section TEXT,
                    tags TEXT[],
                    category TEXT,
                    difficulty TEXT,
                    family TEXT,
                    pet_safe_cat BOOLEAN,
                    pet_safe_dog BOOLEAN,
                    metadata JSONB,
                    text TEXT NOT NULL,
                    embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
                    search_vector TSVECTOR GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(section, '') || ' ' || text)
                    ) STORED,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw
                ON {self.table_name} USING hnsw (embedding vector_cosine_ops)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_search_gin
                ON {self.table_name} USING gin (search_vector)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_slug
                ON {self.table_name} (slug)
                """
            )
        logger.info(f"Lakebase pgvector table ready: {self.table_name}")

    @staticmethod
    def _vector_literal(vector: list[float] | tuple[float, ...]) -> str:
        """Format a Python vector for PostgreSQL's vector input syntax."""
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def _embed_dataframe(self, chunks: DataFrame) -> DataFrame:
        """Generate embeddings in Databricks using ai_query on the configured endpoint."""
        if self.spark is None:
            raise ValueError("A SparkSession is required to embed a DataFrame")
        from pyspark.sql import functions as F

        # Some Databricks Runtime versions expose ai_query only as a SQL function.
        endpoint = self.embedding_model.replace("'", "''")
        return chunks.withColumn(
            "embedding",
            F.expr(f"CAST(ai_query('{endpoint}', text) AS ARRAY<FLOAT>)"),
        )

    @staticmethod
    def _rows_from_dataframe(chunks: DataFrame) -> Iterator[dict[str, Any]]:
        for row in chunks.select(*CHUNK_COLUMNS, "embedding").toLocalIterator():
            values = row.asDict(recursive=True)
            if values.get("id") and values.get("text") and values.get("embedding"):
                yield values

    def _upsert(self, rows: Iterator[dict[str, Any]]) -> int:
        """Upsert embedded chunk rows into Lakebase in batches of 100."""
        insert_sql = f"""
            INSERT INTO {self.table_name}
                (id, slug, url, section, tags, category, difficulty, family,
                 pet_safe_cat, pet_safe_dog, metadata, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                slug = EXCLUDED.slug,
                url = EXCLUDED.url,
                section = EXCLUDED.section,
                tags = EXCLUDED.tags,
                category = EXCLUDED.category,
                difficulty = EXCLUDED.difficulty,
                family = EXCLUDED.family,
                pet_safe_cat = EXCLUDED.pet_safe_cat,
                pet_safe_dog = EXCLUDED.pet_safe_dog,
                metadata = EXCLUDED.metadata,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
        """
        count = 0
        with self.connection() as conn:
            with conn.cursor() as cursor:
                batch: list[tuple[Any, ...]] = []
                for row in rows:
                    batch.append(
                        (
                            row["id"],
                            row["slug"],
                            row.get("url"),
                            row.get("section"),
                            row.get("tags"),
                            row.get("category"),
                            row.get("difficulty"),
                            row.get("family"),
                            row.get("pet_safe_cat"),
                            row.get("pet_safe_dog"),
                            row.get("metadata"),
                            row["text"],
                            self._vector_literal(row["embedding"]),
                        )
                    )
                    if len(batch) >= 100:
                        cursor.executemany(insert_sql, batch)
                        count += len(batch)
                        batch.clear()
                if batch:
                    cursor.executemany(insert_sql, batch)
                    count += len(batch)
        return count

    def index_delta_chunks(self, refresh: bool = False) -> int:
        """Embed and upsert the Delta plant_care_chunks rows into Lakebase.

        If the table already has rows and refresh is False, nothing is embedded and 0 is returned.
        """
        if self.spark is None:
            raise ValueError("A SparkSession is required to index Delta chunks")
        from pyspark.sql import functions as F

        self.ensure_schema()
        if not refresh:
            with self.connection() as conn:
                existing = conn.execute(f"SELECT EXISTS (SELECT 1 FROM {self.table_name})").fetchone()[0]
            if existing:
                logger.info(f"{self.table_name} already populated; skipping embedding (pass refresh=True to rebuild)")
                return 0
        source_table = f"{self.config.catalog}.{self.config.schema}.{self.source_table_name}"
        source = self.spark.table(source_table)
        typed_fields = [*TYPED_TEXT_FIELDS, *TYPED_BOOL_FIELDS]
        metadata_fields = [
            c for c in source.columns if c not in CHUNK_SOURCE_COLUMNS | set(typed_fields) | {"tags"}
        ]
        chunks = source.select(
            F.col("chunk_id").alias("id"),
            "slug",
            "url",
            "section",
            "text",
            F.when(F.col("tags") != "", F.split(F.col("tags"), ", ")).alias("tags"),
            *typed_fields,
            F.to_json(F.struct(*metadata_fields)).alias("metadata"),
        )

        count = self._upsert(self._rows_from_dataframe(self._embed_dataframe(chunks)))
        logger.info(f"Indexed {count} plant care chunks in Lakebase")
        return count

    def embed_query(self, query: str) -> list[float]:
        """Generate a query embedding with the same model used for the documents."""
        if self.spark is not None:
            query_df = self.spark.createDataFrame([(query,)], ["text"])
            row = self._embed_dataframe(query_df).select("embedding").first()
            if not row or not row.embedding:
                raise ValueError("The embedding endpoint returned no vector")
            return list(row.embedding)

        workspace = self.workspace or WorkspaceClient()
        client = workspace.serving_endpoints.get_open_ai_client()
        response = client.embeddings.create(model=self.embedding_service, input=query)
        return list(response.data[0].embedding)

    @staticmethod
    def _filters(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
        """Build a WHERE clause of equality filters on whitelisted columns.

        A tags filter matches rows whose tag array contains the given label.
        """
        filters = filters or {}
        unknown = set(filters) - ALLOWED_FILTERS
        if unknown:
            raise ValueError(f"Unsupported filters: {sorted(unknown)}")
        clauses = [
            "%s = ANY(tags)" if column == "tags" else f"{column} = %s" for column in filters
        ]
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, list(filters.values())

    def _fetch(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                columns = [description.name for description in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run cosine-similarity search, optionally filtered by equality on allowed columns."""
        query_vector = self._vector_literal(self.embed_query(query))
        where, filter_params = self._filters(filters)
        sql = f"""
            SELECT id, slug, url, section, tags, text, category, difficulty, family,
                   pet_safe_cat, pet_safe_dog, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self.table_name}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        return self._fetch(sql, [query_vector, *filter_params, query_vector, num_results])

    def hybrid_search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Combine cosine similarity with the fraction of query terms each chunk contains."""
        query_vector = self._vector_literal(self.embed_query(query))
        where, filter_params = self._filters(filters)
        terms = _query_terms(query)
        sql = f"""
            WITH ranked AS (
                SELECT id, slug, url, section, tags, text, category, difficulty, family,
                       pet_safe_cat, pet_safe_dog, metadata,
                       1 - (embedding <=> %s::vector) AS vector_score,
                       (
                           SELECT count(*)::float
                           FROM unnest(%s::text[]) AS term
                           WHERE search_vector @@ to_tsquery('english', term)
                       ) / %s AS text_score
                FROM {self.table_name}
                {where}
            )
            SELECT *, (%s * vector_score + %s * text_score) AS score
            FROM ranked
            ORDER BY score DESC
            LIMIT %s
        """
        params = [
            query_vector,
            terms,
            max(len(terms), 1),
            *filter_params,
            vector_weight,
            1.0 - vector_weight,
            num_results,
        ]
        return self._fetch(sql, params)
