import logging
from pathlib import Path

import pandas
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

DB_URL = f'postgresql+psycopg2://phoenix:@127.0.0.1:5432/tosdr'
TABLES = {'cases', 'documents', 'points', 'services', 'topics', 'reasons'}
VERSION = '211222'  # day month year
DATA_DIR = here / 'data'

def convert(table_name):
    sql_path = DATA_DIR / f'{table_name}_{VERSION}.sql'
    df = pandas.read_sql_table(table_name, create_engine(DB_URL))
    # Set the DataFrame index to the `id` column from sql
    df = df.set_index(df.id, drop=True)
    logger.info(f"{table_name} size: {len(df)}")

    out_path = sql_path.with_name(f'{table_name}_{VERSION}.pkl')
    df.to_pickle(out_path)
    logger.info(f"Converted {sql_path.relative_to(here)} to {out_path.relative_to(here)}")


if __name__ == '__main__':
    for table_name in TABLES:
        convert(table_name)
