from html.parser import HTMLParser

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from plant_care.config import ProjectConfig

SKIP_TAGS = {"script", "style"}
HEADING_TAGS = {f"h{level}" for level in range(1, 7)}
BLOCK_TAGS = HEADING_TAGS | {"p", "li", "tr", "blockquote", "pre"}
CELL_TAGS = {"td", "th"}
CELL_SEPARATOR = ": "

BLOCK_SCHEMA = ArrayType(StructType([StructField("tag", StringType()), StructField("text", StringType())]))
PARSED_SCHEMA = StructType(
    [
        StructField("slug", StringType(), False),
        StructField("url", StringType(), True),
        StructField("blocks", BLOCK_SCHEMA, True),
    ]
)

MAX_LIST_CHARS = 500
CHUNK_SCHEMA = StructType(
    [
        StructField("chunk_id", StringType(), False),
        StructField("slug", StringType(), False),
        StructField("url", StringType(), True),
        StructField("section", StringType(), True),
        StructField("text", StringType(), True),
    ]
)


class _ArticleParser(HTMLParser):
    """Collect the text of each block-level element together with its tag name."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._tag: str | None = None
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._flush()
            self._tag = tag
        elif tag in CELL_TAGS and self._tag == "tr" and "".join(self._parts).strip():
            self._parts = ["".join(self._parts).rstrip(), CELL_SEPARATOR]

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_TAGS:
            self._flush()
            self._tag = None

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = " ".join("".join(self._parts).split())
        if text:
            self.blocks.append((self._tag or "text", text))
        self._parts = []


def _join(*parts: str) -> str:
    """Join the non-empty parts with the ": " separator."""
    return ": ".join(part for part in parts if part)


class DataProcessor:
    """
    DataProcessor handles the processing workflow of:
    - Reading saved article HTML from the Volume
    - Parsing each article into tagged text blocks
    - Chunking the blocks and joining plant metadata to each chunk
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with catalog and schema
        """
        self.spark = spark
        self.cfg = config
        self.plant_table = f"{config.catalog}.{config.schema}.plant_care"
        self.parsed_table = f"{config.catalog}.{config.schema}.plant_care_parsed"
        self.chunks_table = f"{config.catalog}.{config.schema}.plant_care_chunks"

    @staticmethod
    def parse_html(html: str) -> list[tuple[str, str]]:
        """
        Parse article HTML into a list of (tag, text) blocks.

        Args:
            html: Article HTML

        Returns:
            List of (tag, text) tuples in document order, such as ("h2", "Light") or ("p", "...")
        """
        parser = _ArticleParser()
        parser.feed(html)
        parser.close()
        return parser.blocks

    def _save(self, df: DataFrame, table: str) -> None:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
        logger.info(f"Saved to {table}")

    def parse_all_articles(self, save_table: bool = True) -> DataFrame:
        """
        Parse the saved article of every plant, optionally saving the result to the plant_care_parsed table.

        Args:
            save_table: Whether to overwrite the plant_care_parsed table with the result

        Returns:
            DataFrame with slug, url, and blocks (array of tag and text)
        """
        rows = (
            self.spark.table(self.plant_table)
            .filter("volume_path IS NOT NULL")
            .select("slug", "url", "volume_path")
            .collect()
        )
        parsed = []
        for row in rows:
            with open(row["volume_path"], encoding="utf-8") as f:
                parsed.append((row["slug"], row["url"], self.parse_html(f.read())))

        parsed_df = self.spark.createDataFrame(parsed, schema=PARSED_SCHEMA)
        if save_table:
            self._save(parsed_df, self.parsed_table)
        return parsed_df

    @staticmethod
    def chunk_blocks(slug: str, url: str, blocks: list[tuple[str, str]]) -> list[tuple[str, ...]]:
        """
        Turn one article's blocks into chunks, one per non-heading block.

        A heading merges into the body block after it, so the chunk starts with the heading.
        Consecutive li items in a section merge into one chunk joined with "; ", up to
        MAX_LIST_CHARS characters. A tr with no merged heading gets its section prefixed.
        section is the most recent heading. A trailing heading with no body becomes its own chunk.

        chunk_id is the block's position in the article, counting headings too, so IDs are
        unique and stable but not contiguous. A merged list uses the position of its first item.
        For example, blocks h2, p, p, h3, p give IDs _1, _2, _4.

        Args:
            slug: Plant slug, used as the prefix of each chunk_id
            url: Article URL, copied to each chunk
            blocks: (tag, text) tuples in document order

        Returns:
            List of (chunk_id, slug, url, section, text) tuples matching CHUNK_SCHEMA
        """
        chunks = []
        section = ""
        heading = ""
        items: list[tuple[int, str]] = []

        def add(chunk_id: str, text: str) -> None:
            chunks.append((chunk_id, slug, url, section, text))

        def flush_items() -> None:
            if items:
                add(f"{slug}_{items[0][0]}", "; ".join(t for _, t in items))
                items.clear()

        for index, (tag, text) in enumerate(blocks):
            if tag in HEADING_TAGS:
                flush_items()
                section = text
                heading = _join(heading, text)
            elif tag == "li":
                text = _join(heading, text)
                heading = ""
                if items and len("; ".join([*(t for _, t in items), text])) > MAX_LIST_CHARS:
                    flush_items()
                items.append((index, text))
            else:
                flush_items()
                prefix = heading or (section if tag == "tr" else "")
                add(f"{slug}_{index}", _join(prefix, text))
                heading = ""
        flush_items()
        if heading:
            add(f"{slug}_{len(blocks)}", heading)
        return chunks

    def chunk_all_articles(self, save_table: bool = True) -> DataFrame:
        """
        Chunk every parsed article, optionally saving the result to the plant_care_chunks table.

        Each chunk carries the plant_care metadata of its plant, except summary, volume_path and
        ingestion_timestamp. Summary would repeat on every chunk of the plant, and the other two
        are operational fields.

        Args:
            save_table: Whether to overwrite the plant_care_chunks table with the result

        Returns:
            DataFrame with one row per chunk
        """
        rows = self.spark.table(self.parsed_table).select("slug", "url", "blocks").collect()
        chunks = [
            chunk
            for row in rows
            for chunk in self.chunk_blocks(
                row["slug"], row["url"], [(block["tag"], block["text"]) for block in row["blocks"] or []]
            )
        ]
        logger.info(f"Created {len(chunks)} chunks from {len(rows)} articles")

        plants_df = self.spark.table(self.plant_table).drop(
            "summary", "url", "volume_path", "ingestion_timestamp", "published_at", "updated_at"
        )
        chunks_df = self.spark.createDataFrame(chunks, schema=CHUNK_SCHEMA).join(plants_df, "slug", "left")
        if save_table:
            self._save(chunks_df, self.chunks_table)
        return chunks_df
