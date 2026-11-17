from html.parser import HTMLParser

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from plant_care.config import ProjectConfig

SKIP_TAGS = {"script", "style"}
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "blockquote", "pre"}
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

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
MAX_LIST_CHARS = 500
CHUNK_SCHEMA = StructType(
    [
        StructField("chunk_id", StringType(), False),
        StructField("slug", StringType(), False),
        StructField("url", StringType(), True),
        StructField("section", StringType(), True),
        StructField("tag", StringType(), True),
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


class DataProcessor:
    """
    DataProcessor handles the processing workflow of:
    - Reading saved article HTML from the Volume
    - Parsing each article into tagged text blocks
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
                blocks = self.parse_html(f.read())
            parsed.append((row["slug"], row["url"], blocks))

        logger.info(f"Parsed {len(parsed)} articles")
        parsed_df = self.spark.createDataFrame(parsed, schema=PARSED_SCHEMA)
        if save_table:
            parsed_df.write.format("delta").mode("overwrite").saveAsTable(self.parsed_table)
            logger.info(f"Saved parsed articles to {self.parsed_table}")
        return parsed_df

    @staticmethod
    def chunk_blocks(slug: str, blocks: list[tuple[str, str]]) -> list[tuple[str, str, str, str]]:
        """
        Turn one article's blocks into chunks, one per non-heading block.

        Headings are merged into the body block that follows them, so the chunk text
        starts with its heading. Consecutive list items (li) in the same section are
        merged into one chunk, joined with "; ", up to MAX_LIST_CHARS characters. Table
        rows that are not merged with a heading get the current section prefixed, so
        every row carries its section. Each chunk's section is the most recent heading.
        Headings at the end of an article with no body after them become their own chunk.

        chunk_id is the position of the block in the article, counting headings too, so
        the IDs skip the heading positions. For example, an article whose blocks are
        h2, p, p, h3, p gives IDs _1, _2, _4. For a merged list chunk, the ID is the
        position of its first item. The IDs are unique and stable, but not contiguous.

        Args:
            slug: Plant slug, used as the prefix of each chunk_id
            blocks: (tag, text) tuples in document order

        Returns:
            List of (chunk_id, section, tag, text) tuples
        """
        chunks = []
        section = ""
        pending: list[str] = []
        heading_tag = ""
        items: list[tuple[int, str]] = []

        def flush_items() -> None:
            if items:
                chunks.append((f"{slug}_{items[0][0]}", section, "li", "; ".join(t for _, t in items)))
                items.clear()

        for index, (tag, text) in enumerate(blocks):
            if tag in HEADING_TAGS:
                flush_items()
                section = text
                pending.append(text)
                heading_tag = tag
                continue
            if tag == "li":
                text = ": ".join([*pending, text])
                pending = []
                if items and len("; ".join([*(t for _, t in items), text])) > MAX_LIST_CHARS:
                    flush_items()
                items.append((index, text))
                continue
            flush_items()
            if tag == "tr" and not pending and section:
                text = f"{section}: {text}"
            chunks.append((f"{slug}_{index}", section, tag, ": ".join([*pending, text])))
            pending = []
        flush_items()
        if pending:
            chunks.append((f"{slug}_{len(blocks)}", section, heading_tag, ": ".join(pending)))
        return chunks

    def chunk_all_articles(self, save_table: bool = True) -> DataFrame:
        """
        Chunk every parsed article, optionally saving the result to the plant_care_chunks table.

        Args:
            save_table: Whether to overwrite the plant_care_chunks table with the result

        Returns:
            DataFrame with one row per chunk
        """
        rows = self.spark.table(self.parsed_table).select("slug", "url", "blocks").collect()

        chunks = []
        for row in rows:
            blocks = [(block["tag"], block["text"]) for block in row["blocks"] or []]
            for chunk_id, section, tag, text in self.chunk_blocks(row["slug"], blocks):
                chunks.append((chunk_id, row["slug"], row["url"], section, tag, text))

        logger.info(f"Created {len(chunks)} chunks from {len(rows)} articles")
        chunks_df = self.spark.createDataFrame(chunks, schema=CHUNK_SCHEMA)
        if save_table:
            chunks_df.write.format("delta").mode("overwrite").saveAsTable(self.chunks_table)
            logger.info(f"Saved chunks to {self.chunks_table}")
        return chunks_df
