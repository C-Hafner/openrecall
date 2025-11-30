import sqlite3
from collections import namedtuple
import numpy as np
from typing import Any, List, Optional, Tuple

from openrecall.config import db_path

# Define the structure of a database entry using namedtuple
Entry = namedtuple("Entry", ["id", "app", "title", "text", "timestamp", "embedding", "monitor_id", "monitor_name"])

# Define the structure of a monitor configuration entry
Monitor = namedtuple("Monitor", ["id", "monitor_index", "name", "width", "height", "enabled"])


def create_db() -> None:
    """
    Creates the SQLite database and the 'entries' and 'monitors' tables if they don't exist.

    The table schema includes columns for an auto-incrementing ID, application name,
    window title, extracted text, timestamp, text embedding, monitor_id, and monitor_name.
    Also performs schema migration for existing databases.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Create monitors configuration table
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS monitors (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       monitor_index INTEGER UNIQUE,
                       name TEXT,
                       width INTEGER,
                       height INTEGER,
                       enabled INTEGER DEFAULT 1
                   )"""
            )

            # Check if entries table exists and needs migration
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
            )
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Check if monitor_id column exists (migration check)
                cursor.execute("PRAGMA table_info(entries)")
                columns = [col[1] for col in cursor.fetchall()]

                if 'monitor_id' not in columns:
                    # Perform migration: add new columns
                    cursor.execute("ALTER TABLE entries ADD COLUMN monitor_id INTEGER DEFAULT 1")
                    cursor.execute("ALTER TABLE entries ADD COLUMN monitor_name TEXT DEFAULT 'Primary Monitor'")

                    # Drop the old UNIQUE constraint on timestamp by recreating the table
                    cursor.execute(
                        """CREATE TABLE entries_new (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               app TEXT,
                               title TEXT,
                               text TEXT,
                               timestamp INTEGER,
                               embedding BLOB,
                               monitor_id INTEGER DEFAULT 1,
                               monitor_name TEXT DEFAULT 'Primary Monitor',
                               UNIQUE(timestamp, monitor_id)
                           )"""
                    )
                    cursor.execute(
                        """INSERT INTO entries_new (id, app, title, text, timestamp, embedding, monitor_id, monitor_name)
                           SELECT id, app, title, text, timestamp, embedding, monitor_id, monitor_name FROM entries"""
                    )
                    cursor.execute("DROP TABLE entries")
                    cursor.execute("ALTER TABLE entries_new RENAME TO entries")
                    print("Database migrated to support multi-monitor")
            else:
                # Create new entries table with multi-monitor support
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS entries (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           app TEXT,
                           title TEXT,
                           text TEXT,
                           timestamp INTEGER,
                           embedding BLOB,
                           monitor_id INTEGER DEFAULT 1,
                           monitor_name TEXT DEFAULT 'Primary Monitor',
                           UNIQUE(timestamp, monitor_id)
                       )"""
                )

            # Add index on timestamp for faster lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON entries (timestamp)"
            )
            # Add index on monitor_id for filtering
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_monitor_id ON entries (monitor_id)"
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error during table creation: {e}")


def get_all_entries() -> List[Entry]:
    """
    Retrieves all entries from the database.

    Returns:
        List[Entry]: A list of all entries as Entry namedtuples.
                     Returns an empty list if the table is empty or an error occurs.
    """
    entries: List[Entry] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
            cursor = conn.cursor()
            cursor.execute("SELECT id, app, title, text, timestamp, embedding, monitor_id, monitor_name FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            for row in results:
                # Deserialize the embedding blob back into a NumPy array
                embedding = np.frombuffer(row["embedding"], dtype=np.float32) # Assuming float32, adjust if needed
                entries.append(
                    Entry(
                        id=row["id"],
                        app=row["app"],
                        title=row["title"],
                        text=row["text"],
                        timestamp=row["timestamp"],
                        embedding=embedding,
                        monitor_id=row["monitor_id"],
                        monitor_name=row["monitor_name"],
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching all entries: {e}")
    return entries


def get_timestamps() -> List[int]:
    """
    Retrieves all timestamps from the database, ordered descending.

    Returns:
        List[int]: A list of all timestamps.
                   Returns an empty list if the table is empty or an error occurs.
    """
    timestamps: List[int] = []
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Use the index for potentially faster retrieval
            cursor.execute("SELECT timestamp FROM entries ORDER BY timestamp DESC")
            results = cursor.fetchall()
            timestamps = [result[0] for result in results]
    except sqlite3.Error as e:
        print(f"Database error while fetching timestamps: {e}")
    return timestamps


def insert_entry(
    text: str, timestamp: int, embedding: np.ndarray, app: str, title: str,
    monitor_id: int = 1, monitor_name: str = "Primary Monitor"
) -> Optional[int]:
    """
    Inserts a new entry into the database.

    Args:
        text (str): The extracted text content.
        timestamp (int): The Unix timestamp of the screenshot.
        embedding (np.ndarray): The embedding vector for the text.
        app (str): The name of the active application.
        title (str): The title of the active window.
        monitor_id (int): The monitor index (1-based).
        monitor_name (str): The friendly name of the monitor.

    Returns:
        Optional[int]: The ID of the newly inserted row, or None if insertion fails.
                       Prints an error message to stderr on failure.
    """
    embedding_bytes: bytes = embedding.astype(np.float32).tobytes() # Ensure consistent dtype
    last_row_id: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO entries (text, timestamp, embedding, app, title, monitor_id, monitor_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(timestamp, monitor_id) DO NOTHING""", # Avoid duplicates based on timestamp+monitor_id
                (text, timestamp, embedding_bytes, app, title, monitor_id, monitor_name),
            )
            conn.commit()
            if cursor.rowcount > 0: # Check if insert actually happened
                last_row_id = cursor.lastrowid
            # else:
                # Optionally log that a duplicate timestamp was encountered
                # print(f"Skipped inserting entry with duplicate timestamp: {timestamp}")

    except sqlite3.Error as e:
        # More specific error handling can be added (e.g., IntegrityError for UNIQUE constraint)
        print(f"Database error during insertion: {e}")
    return last_row_id


def get_all_monitors() -> List[Monitor]:
    """
    Retrieves all monitor configurations from the database.

    Returns:
        List[Monitor]: A list of all monitors as Monitor namedtuples.
                       Returns an empty list if the table is empty or an error occurs.
    """
    monitors: List[Monitor] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, monitor_index, name, width, height, enabled FROM monitors ORDER BY monitor_index")
            results = cursor.fetchall()
            for row in results:
                monitors.append(
                    Monitor(
                        id=row["id"],
                        monitor_index=row["monitor_index"],
                        name=row["name"],
                        width=row["width"],
                        height=row["height"],
                        enabled=bool(row["enabled"]),
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching monitors: {e}")
    return monitors


def upsert_monitor(monitor_index: int, name: str, width: int, height: int, enabled: bool = True) -> Optional[int]:
    """
    Inserts or updates a monitor configuration.

    Args:
        monitor_index (int): The monitor index from mss (1-based).
        name (str): The friendly name for the monitor.
        width (int): The width of the monitor in pixels.
        height (int): The height of the monitor in pixels.
        enabled (bool): Whether this monitor is enabled for screenshots.

    Returns:
        Optional[int]: The ID of the inserted/updated row, or None if operation fails.
    """
    last_row_id: Optional[int] = None
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO monitors (monitor_index, name, width, height, enabled)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(monitor_index) DO UPDATE SET
                       name = excluded.name,
                       width = excluded.width,
                       height = excluded.height,
                       enabled = excluded.enabled""",
                (monitor_index, name, width, height, int(enabled)),
            )
            conn.commit()
            last_row_id = cursor.lastrowid
    except sqlite3.Error as e:
        print(f"Database error during monitor upsert: {e}")
    return last_row_id


def update_monitor_enabled(monitor_index: int, enabled: bool) -> bool:
    """
    Updates the enabled status of a monitor.

    Args:
        monitor_index (int): The monitor index to update.
        enabled (bool): Whether the monitor should be enabled.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE monitors SET enabled = ? WHERE monitor_index = ?",
                (int(enabled), monitor_index),
            )
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Database error during monitor enabled update: {e}")
        return False


def update_monitor_name(monitor_index: int, name: str) -> bool:
    """
    Updates the name of a monitor.

    Args:
        monitor_index (int): The monitor index to update.
        name (str): The new name for the monitor.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE monitors SET name = ? WHERE monitor_index = ?",
                (name, monitor_index),
            )
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Database error during monitor name update: {e}")
        return False


def get_enabled_monitors() -> List[Monitor]:
    """
    Retrieves all enabled monitor configurations from the database.

    Returns:
        List[Monitor]: A list of enabled monitors as Monitor namedtuples.
                       Returns an empty list if no enabled monitors or an error occurs.
    """
    monitors: List[Monitor] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, monitor_index, name, width, height, enabled FROM monitors WHERE enabled = 1 ORDER BY monitor_index")
            results = cursor.fetchall()
            for row in results:
                monitors.append(
                    Monitor(
                        id=row["id"],
                        monitor_index=row["monitor_index"],
                        name=row["name"],
                        width=row["width"],
                        height=row["height"],
                        enabled=bool(row["enabled"]),
                    )
                )
    except sqlite3.Error as e:
        print(f"Database error while fetching enabled monitors: {e}")
    return monitors
