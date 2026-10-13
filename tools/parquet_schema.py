"""Print the schema and the row count of a parquet file (homelab, uses pyarrow from the oracle environment).   python tools/parquet_schema.py FILE"""
import sys

import pyarrow.parquet as pq

pf = pq.ParquetFile(sys.argv[1])
print(pf.schema_arrow)
print(pf.metadata.num_rows, "rows,", pf.metadata.num_row_groups, "row groups")
