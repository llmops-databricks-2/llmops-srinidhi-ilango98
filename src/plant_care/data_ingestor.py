"""PlantSolve API ingestion pipeline.

   ↓ (fetch_plant_care)
Plant profiles flattened into care fields
   ↓ (save_article)
Article HTML in Volume (volume_path on each plant)
   ↓ (create_dataframe)
Spark DataFrame with the plant care schema

Source: PlantSolve API (CC BY 4.0, https://www.plantsolve.com). Requires a backlink when used.
"""

import json
import os
import urllib.request
from datetime import datetime

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

from plant_care.config import ProjectConfig

BASE_URL = "https://www.plantsolve.com/api/v1/plants"
TIMEOUT_SECONDS = 30

NON_NULLABLE_FIELDS = {"plant_id", "slug", "common_name"}
BOOLEAN_FIELDS = {"pet_safe_cat", "pet_safe_dog"}
FIELD_NAMES = [
    "plant_id",
    "slug",
    "common_name",
    "scientific_name",
    "family",
    "category",
    "tags",
    "summary",
    "difficulty",
    "placement",
    "watering",
    "soil",
    "toxicity",
    "light_description",
    "sunlight_hours",
    "light_intensity",
    "temperature_recommended",
    "humidity_recommended",
    "soil_ph_recommended",
    "fertilizer_frequency",
    "growth_rate",
    "mature_height",
    "pet_safe_cat",
    "pet_safe_dog",
    "url",
    "volume_path",
    "ingestion_timestamp",
]
PLANT_SCHEMA = StructType(
    [
        StructField(
            name,
            BooleanType() if name in BOOLEAN_FIELDS else StringType(),
            name not in NON_NULLABLE_FIELDS,
        )
        for name in FIELD_NAMES
    ]
)


def _section(data: dict, key: str) -> dict:
    """Get a nested section of a profile, defaulting to an empty dict.

    Args:
        data: Dictionary containing the section
        key: Section name

    Returns:
        The section dictionary, or an empty dict if it is missing or null

    """
    return data.get(key) or {}


def _recommended(parameters: dict, key: str) -> str | None:
    """Get the recommended value of a care parameter.

    Args:
        parameters: Parameters section of a plant profile
        key: Parameter name, such as temperature, humidity, or soilPH

    Returns:
        Recommended value, or None if the parameter is missing

    """
    return _section(parameters, key).get("recommended")


class DataIngestor:
    """DataIngestor handles the ingestion workflow.

    It performs these steps:
    - Fetching the plant index and each plant profile from PlantSolve
    - Flattening profiles into structured care fields
    - Saving article HTML to the Volume
    - Creating the Spark DataFrame for the plant_care table
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """Initialize DataIngestor with a Spark session and configuration.

        Args:
            spark: Active SparkSession
            config: ProjectConfig object with catalog, schema, and volume

        """
        self.spark = spark
        self.cfg = config
        self.article_dir = f"/Volumes/{config.catalog}/{config.schema}/{config.volume}/articles"
        os.makedirs(self.article_dir, exist_ok=True)

    def _get_json(self, url: str) -> dict | list:
        """Fetch a URL and decode its JSON body.

        Args:
            url: URL of the JSON resource

        Returns:
            Decoded JSON object or array

        """
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))

    def save_article(self, slug: str, html: str | None) -> str | None:
        """Save a plant's article HTML to the Volume.

        Args:
            slug: Plant slug, used as the file name
            html: Article HTML from the profile's content field

        Returns:
            Volume path of the saved file, or None if the profile has no article

        """
        if not html:
            return None
        path = f"{self.article_dir}/{slug}.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def parse_plant_profile(self, profile: dict) -> dict:
        """Flatten a PlantSolve plant profile into structured care fields.

        Args:
            profile: Raw plant profile from the PlantSolve detail endpoint

        Returns:
            Flat dictionary of plant care fields

        """
        care = _section(profile, "care")
        taxonomy = _section(profile, "taxonomy")
        lighting = _section(profile, "lighting")
        parameters = _section(profile, "parameters")
        nutrients = _section(profile, "nutrients")
        growth = _section(profile, "growthCharacteristics")
        pet_safe = _section(profile, "petSafe")

        return {
            "plant_id": profile.get("id"),
            "slug": profile.get("slug"),
            "common_name": profile.get("commonName"),
            "scientific_name": taxonomy.get("sciName"),
            "family": taxonomy.get("family"),
            "category": profile.get("category"),
            "tags": ", ".join(profile.get("taxonomyTags") or []),
            "summary": profile.get("summary"),
            "difficulty": care.get("difficulty"),
            "placement": care.get("placement"),
            "watering": care.get("watering"),
            "soil": care.get("soil"),
            "toxicity": care.get("toxicity"),
            "light_description": lighting.get("description"),
            "sunlight_hours": lighting.get("sunlightHours"),
            "light_intensity": lighting.get("intensity"),
            "temperature_recommended": _recommended(parameters, "temperature"),
            "humidity_recommended": _recommended(parameters, "humidity"),
            "soil_ph_recommended": _recommended(parameters, "soilPH"),
            "fertilizer_frequency": nutrients.get("fertilizerFrequency"),
            "growth_rate": growth.get("growthRate"),
            "mature_height": growth.get("matureHeight"),
            "pet_safe_cat": pet_safe.get("cat"),
            "pet_safe_dog": pet_safe.get("dog"),
            "url": profile.get("canonicalUrl"),
            "ingestion_timestamp": datetime.now().isoformat(),
        }

    def fetch_plant_care(self) -> list[dict]:
        """Fetch all plant care profiles from the PlantSolve API and save their articles.

        Plants whose profile fails to download are logged and skipped.

        Returns:
            List of structured plant care dictionaries, each with a volume_path

        """
        index = self._get_json(f"{BASE_URL}/index.json")
        logger.info(f"Found {len(index)} plants in PlantSolve index")

        plants = []
        for entry in index:
            slug = entry["slug"]
            try:
                profile = self._get_json(f"{BASE_URL}/{slug}.json")
            except Exception as e:
                logger.warning(f"Failed to fetch plant '{slug}': {e}")
                continue
            plant = self.parse_plant_profile(profile)
            plant["volume_path"] = self.save_article(slug, profile.get("content"))
            plants.append(plant)

        return plants

    def create_dataframe(self, plants: list[dict]) -> DataFrame:
        """Create a Spark DataFrame from the flattened plant care records.

        Args:
            plants: List of plant care dictionaries from fetch_plant_care

        Returns:
            DataFrame with the plant_care schema

        """
        return self.spark.createDataFrame(plants, schema=PLANT_SCHEMA)
