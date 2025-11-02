import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[2] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from quarto_mssql_gemini import get_valid_combinations


def create_combinations_file(output_file: str = "valid_combinations.csv") -> None:
    """Create or refresh the cached CSV of valid filter combinations."""

    try:
        df = get_valid_combinations(include_counts=True)
    except Exception as exc:
        raise RuntimeError(
            "Unable to fetch combinations from SQL Server. "
            "Check your database credentials and network connectivity."
        ) from exc

    df = df.sort_values("combination_count", ascending=False)
    df.to_csv(output_file, index=False)

    print(f"Successfully created {output_file}")
    print(f"Total number of combinations: {len(df)}")
    print("\nTop 5 most common combinations:")
    print(df.head().to_string())


if __name__ == "__main__":
    create_combinations_file()
