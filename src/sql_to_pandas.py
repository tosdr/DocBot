import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

DB_URL = f'postgresql+psycopg2://phoenix:@127.0.0.1:5432/phoenix'
TABLES = {'cases', 'documents', 'points', 'services', 'topics'}
VERSION = '280126'  # day month year
DUMP_DIR = here / f'../data/db_dumps/{VERSION}'

def convert(table_name):
    df = pd.read_sql_table(table_name, create_engine(DB_URL))
    # Set the DataFrame index to the `id` column from sql
    df = df.set_index(df.id, drop=True)
    logger.info(f"{table_name} size: {len(df)}")

    out_path = DUMP_DIR / f'{table_name}.pkl'
    df.to_pickle(out_path)
    logger.info(f"Exported to {out_path.relative_to(here)}")


if __name__ == '__main__':
    for table_name in TABLES:
        convert(table_name)
