# TODO: Annotate what this script does.

import duckdb
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    print(f"Remote data source: {cfg.data.hf_url}")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # con.execute("SET hf_token = 'hf_xxxxx';")  # if private

    BASE = cfg.data.hf_url
    clocks_column = "clocks_black"
    # clocks_column = "clocks_white"

    # Get sorted file list (assuming chronological naming)
    files = con.execute(
        f"""
        SELECT file FROM glob('{BASE}/**/*.parquet') ORDER BY file
    """
    ).fetchall()
    files = [f[0] for f in files]

    def has_clocks(filepath: str) -> bool:
        result = con.execute(
            f"""
            SELECT 1 FROM read_parquet('{filepath}')
            WHERE {clocks_column} IS NOT NULL
            LIMIT 1
        """
        ).fetchone()
        return result is not None

    # Binary search for first file with clocks
    lo, hi = 0, len(files) - 1
    first_with_clocks = None
    # first_with_clocks = 50

    if first_with_clocks is None:
        while lo <= hi:
            mid = (lo + hi) // 2
            print(f"Checking {mid}/{len(files)}: {files[mid]}")

            if has_clocks(files[mid]):
                first_with_clocks = mid
                hi = mid - 1  # keep searching earlier
            else:
                lo = mid + 1

    if first_with_clocks:
        print(f"\nFirst file with clocks: {files[first_with_clocks]}")

        # Now find the actual first game
        row = con.execute(
            f"""
            SELECT * FROM read_parquet('{files[first_with_clocks]}')
            WHERE {clocks_column} IS NOT NULL
            LIMIT 1
        """
        ).fetchone()
        print(row)
    else:
        print("No clocks found anywhere")


if __name__ == "__main__":
    main()
