"""Dataset implementation for Prosocial Bench test cases.

Loads test cases from JSON files in the test_cases/cases/ directory.
Each file contains a JSON array of test cases conforming to schema.json.

Implements MIRROR-Eval's DatasetInterface if available; otherwise falls back
to a standalone base class so the package works without mirroreval installed.
"""

import json
from pathlib import Path
from typing import Any, Iterator, Optional

# Optional MIRROR-Eval integration
try:
    from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset  # type: ignore

    _MIRROR_EVAL_AVAILABLE = True
except ImportError:
    _MIRROR_EVAL_AVAILABLE = False

    class DatasetInterface:  # type: ignore
        """Standalone stub matching the MIRROR-Eval DatasetInterface contract."""

        def load_data(self) -> None:
            raise NotImplementedError

        def __iter__(self) -> Iterator[dict[str, Any]]:
            raise NotImplementedError

        def __len__(self) -> int:
            raise NotImplementedError("Length not supported.")

        def get_split(self, name: str) -> Optional["DatasetInterface"]:
            raise NotImplementedError("Splits not supported.")

    def register_dataset(name: str):  # type: ignore
        """No-op decorator when mirroreval is not installed."""
        return lambda cls: cls


@register_dataset("prosocial-bench")
class ProsocialBenchDataset(DatasetInterface):
    """Loads Prosocial Bench test cases from the test_cases/cases/ directory.

    Each .json file in the cases directory should contain a JSON array of
    test case objects conforming to test_cases/schema.json.

    Args:
        cases_dir: Path to the directory containing per-domain JSON files.
                   Defaults to test_cases/cases/ relative to the package root.
        domains: Optional list of domain names to load (e.g. ["productivity"]).
                 If None, all domains are loaded.

    Example:
        dataset = ProsocialBenchDataset()
        dataset.load_data()
        for case in dataset:
            print(case["id"], case["stated_goal"])
    """

    # Default cases directory relative to this file's location
    _DEFAULT_CASES_DIR = (
        Path(__file__).parent.parent.parent / "test_cases" / "cases"
    )

    def __init__(
        self,
        cases_dir: str | Path | None = None,
        domains: list[str] | None = None,
    ):
        self.cases_dir = Path(cases_dir) if cases_dir else self._DEFAULT_CASES_DIR
        self.domains = domains
        self._data: list[dict[str, Any]] = []
        self._loaded = False

    def load_data(self) -> None:
        """Load all test case JSON files from the cases directory."""
        self._data = []

        if not self.cases_dir.exists():
            raise FileNotFoundError(
                f"Test cases directory not found: {self.cases_dir}\n"
                "Run Phase 3 test case generation to populate it."
            )

        json_files = sorted(self.cases_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No .json files found in {self.cases_dir}"
            )

        for json_file in json_files:
            domain_name = json_file.stem  # filename without extension
            if self.domains and domain_name not in self.domains:
                continue

            with open(json_file, encoding="utf-8") as f:
                cases = json.load(f)

            if isinstance(cases, list):
                self._data.extend(cases)
            elif isinstance(cases, dict):
                # Single test case stored as a bare object
                self._data.append(cases)
            else:
                raise ValueError(
                    f"Expected a JSON array or object in {json_file}, "
                    f"got {type(cases).__name__}"
                )

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_data()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        self._ensure_loaded()
        yield from self._data

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._data)

    def get_split(self, name: str) -> "ProsocialBenchDataset":
        """Return a dataset filtered to a single domain.

        Args:
            name: Domain name (e.g. "productivity", "addiction")
        """
        split = ProsocialBenchDataset(cases_dir=self.cases_dir, domains=[name])
        return split
