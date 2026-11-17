from html.parser import HTMLParser

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from plant_care.config import ProjectConfig

SKIP_TAGS = {"script", "style"}
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "blockquote", "pre"}

BLOCK_SCHEMA = ArrayType(StructType([StructField("tag", StringType()), StructField("text", StringType())]))
PARSED_SCHEMA = StructType(
    [
        StructField("slug", StringType(), False),
        StructField("url", StringType(), True),
        StructField("blocks", BLOCK_SCHEMA, True),
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
