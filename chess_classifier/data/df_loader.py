# Loads a subset of a parquet file into a pandas dataframe, in memory.

import duckdb


def load_df_from_parquet(parquet_path: str, n_positions, shuffle_seed=0):
    duckdb.sql("INSTALL aixchess FROM community")
    duckdb.sql("LOAD aixchess")

    games = duckdb.read_parquet(parquet_path)
    games_filtered = duckdb.sql(f"FROM games WHERE result != '*' LIMIT {n_positions}")

    positions = duckdb.sql(
        """
        SELECT
        lc.* EXCLUDE (movedata, clocks_white, clocks_black, tournament),
        t.ply,
        UNNEST(board_at_position(lc.movedata, t.ply))
        FROM games_filtered AS lc
        CROSS JOIN LATERAL (
        SELECT 1 + CAST(floor(random() * lc.ply_count) AS INTEGER) AS ply
        ) AS t
    """
    )
    positions = duckdb.sql("""CREATE OR REPLACE TABLE positions AS FROM positions""")

    # cheating the linter for duckdb; variables are used, they're just inside strings
    _ = games
    _ = games_filtered
    _ = positions

    return duckdb.sql(f"FROM positions ORDER BY hash(rowid + {shuffle_seed})").df()
