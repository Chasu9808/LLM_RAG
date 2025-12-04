import sys, pysqlite3
sys.modules["sqlite3"] = pysqlite3.dbapi2
sys.modules["_sqlite3"] = pysqlite3.dbapi2

import sqlite3
def patch_sqlite():
    print(f"✅ SQLite version (patched): {sqlite3.sqlite_version}")
