
import tempfile
import gc
import time
from pathlib import Path

def test_foo():
    # Exact pattern from the test
    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    path = tmp.name
    
    # Don't close tmp, but use its name to open another connection
    import sqlite3
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE test (id INTEGER PRIMARY KEY)')
    conn.commit()
    
    # Close the sqlite connection
    conn.close()
    
    # Now try to delete via Path
    try:
        Path(path).unlink()
        print("Deleted successfully")
    except PermissionError as e:
        print(f"Failed to delete: {e}")
