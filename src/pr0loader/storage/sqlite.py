"""SQLite storage backend."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Iterator

from pr0loader.models import Item, Tag, Comment

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """SQLite-based storage for pr0loader data."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """Connect to the database with performance optimizations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Apply performance optimizations
        self._optimize_connection()
        self._init_schema()
        logger.info(f"Connected to database: {self.db_path}")

    def _optimize_connection(self):
        """Apply SQLite performance optimizations."""
        cursor = self.conn.cursor()

        try:
            # WAL mode: allows concurrent reads during writes, much faster
            # Note: WAL may not work on all filesystems (e.g., network drives, some WSL setups)
            result = cursor.execute('PRAGMA journal_mode=WAL').fetchone()
            if result and result[0] != 'wal':
                logger.warning(f"Could not enable WAL mode, using {result[0]} instead. Performance may be reduced.")
                logger.warning("This can happen on network drives or certain filesystems.")
        except Exception as e:
            logger.warning(f"Failed to set WAL mode: {e}. Continuing with default journal mode.")

        # Synchronous NORMAL: balanced performance/safety (not FULL which is very slow)
        # NORMAL is safe for WAL mode and much faster than FULL
        try:
            cursor.execute('PRAGMA synchronous=NORMAL')
        except Exception as e:
            logger.warning(f"Failed to set synchronous mode: {e}")

        # Increase cache size to 512MB for systems with lots of RAM (default is ~2MB)
        # This is CRUCIAL for RAID5 HDDs - keeps more data in memory to reduce writes
        # Negative value = KB, so -524288 = 512MB
        try:
            cursor.execute('PRAGMA cache_size=-524288')
        except Exception as e:
            logger.warning(f"Failed to set cache size: {e}")

        # Page size can ONLY be set before any tables exist
        # Check if database is new (no tables yet)
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            has_tables = cursor.fetchone() is not None

            if not has_tables:
                # New database - we can set page size
                cursor.execute('PRAGMA page_size=4096')
                logger.debug("Set page size to 4096 (new database)")
            else:
                # Existing database - cannot change page size
                logger.debug("Skipping page_size setting (existing database)")
        except Exception as e:
            logger.warning(f"Failed to check/set page size: {e}")

        # Memory-mapped I/O for faster reads (1GB for systems with plenty of RAM)
        try:
            cursor.execute('PRAGMA mmap_size=1073741824')
        except Exception as e:
            logger.warning(f"Failed to set mmap_size: {e}")

        # Temp storage in memory for faster sorting/indexing
        try:
            cursor.execute('PRAGMA temp_store=MEMORY')
        except Exception as e:
            logger.warning(f"Failed to set temp_store: {e}")

        # Set temp cache to 256MB for large operations
        try:
            cursor.execute('PRAGMA temp_cache_size=-262144')
        except Exception as e:
            logger.warning(f"Failed to set temp_cache_size: {e}")

        try:
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to commit pragma settings: {e}")

        logger.debug("SQLite performance optimizations applied")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_schema(self):
        """Initialize the database schema."""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY,
                image TEXT,
                promoted INTEGER DEFAULT 0,
                up INTEGER DEFAULT 0,
                down INTEGER DEFAULT 0,
                created INTEGER DEFAULT 0,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                audio INTEGER DEFAULT 0,
                source TEXT DEFAULT '',
                flags INTEGER DEFAULT 0,
                user_name TEXT DEFAULT '',
                mark INTEGER DEFAULT 0,
                gift INTEGER DEFAULT 0,
                item_data TEXT,
                tags_data TEXT,
                comments_data TEXT,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_items_created ON items(created)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_items_promoted ON items(promoted)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_items_image ON items(image)')

        # Note: tags table removed - tags are stored in tags_data JSON column
        # and extracted during prepare step when needed

        self.conn.commit()

    def upsert_item(self, item: Item, commit: bool = True):
        """Insert or update an item.

        Args:
            item: Item to upsert
            commit: If True, commits immediately. If False, caller must commit manually.
        """
        cursor = self.conn.cursor()

        tags_json = json.dumps([t.model_dump() for t in item.tags])
        comments_json = json.dumps([c.model_dump() for c in item.comments])
        item_json = json.dumps(item.model_dump(exclude={'tags', 'comments'}))

        cursor.execute('''
            INSERT OR REPLACE INTO items
            (id, image, promoted, up, down, created, width, height, audio, source,
             flags, user_name, mark, gift, item_data, tags_data, comments_data, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
        ''', (
            item.id, item.image, item.promoted, item.up, item.down, item.created,
            item.width, item.height, item.audio, item.source, item.flags,
            item.user, item.mark, item.gift, item_json, tags_json, comments_json
        ))


        if commit:
            self.conn.commit()

    def upsert_items_batch(self, items: list[Item]):
        """Insert or update multiple items in a single transaction.

        This is much faster than calling upsert_item() repeatedly, especially
        on HDDs, as it only commits once at the end.

        Args:
            items: List of items to upsert
        """
        if not items:
            return

        cursor = self.conn.cursor()

        # Begin an immediate transaction to lock the database early
        cursor.execute('BEGIN IMMEDIATE')

        try:
            # Prepare batch data
            items_data = []

            for item in items:
                tags_json = json.dumps([t.model_dump() for t in item.tags])
                comments_json = json.dumps([c.model_dump() for c in item.comments])
                item_json = json.dumps(item.model_dump(exclude={'tags', 'comments'}))

                items_data.append((
                    item.id, item.image, item.promoted, item.up, item.down, item.created,
                    item.width, item.height, item.audio, item.source, item.flags,
                    item.user, item.mark, item.gift, item_json, tags_json, comments_json
                ))

            # Execute batch insert - tags are already in tags_data JSON column
            cursor.executemany('''
                INSERT OR REPLACE INTO items
                (id, image, promoted, up, down, created, width, height, audio, source,
                 flags, user_name, mark, gift, item_data, tags_data, comments_data, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
            ''', items_data)

            # Commit the transaction
            self.conn.commit()
            logger.debug(f"Batch upserted {len(items)} items")

        except Exception as e:
            # Rollback on error
            self.conn.rollback()
            logger.error(f"Error during batch upsert: {e}")
            raise

    def get_min_id(self) -> int:
        """Get the minimum item ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT MIN(id) as min_id FROM items')
        row = cursor.fetchone()
        return row['min_id'] if row and row['min_id'] else -1

    def get_max_id(self) -> int:
        """Get the maximum item ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(id) as max_id FROM items')
        row = cursor.fetchone()
        return row['max_id'] if row and row['max_id'] else -1

    def get_item_count(self) -> int:
        """Get total number of items."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM items')
        return cursor.fetchone()['count']

    def optimize_database(self):
        """Optimize the database by running ANALYZE and optional VACUUM.

        ANALYZE updates statistics for query optimizer.
        Should be called after large batch operations.
        """
        logger.info("Optimizing database (running ANALYZE)...")
        cursor = self.conn.cursor()
        cursor.execute('ANALYZE')
        self.conn.commit()
        logger.info("Database optimization complete")

    def get_item(self, item_id: int) -> Optional[Item]:
        """Get an item by ID."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM items WHERE id = ?', (item_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def _row_to_item(self, row: sqlite3.Row) -> Item:
        """Convert a database row to an Item."""
        tags = [Tag.model_validate(t) for t in json.loads(row['tags_data'] or '[]')]
        comments = [Comment.model_validate(c) for c in json.loads(row['comments_data'] or '[]')]

        return Item(
            id=row['id'],
            image=row['image'] or '',
            promoted=row['promoted'] or 0,
            up=row['up'] or 0,
            down=row['down'] or 0,
            created=row['created'] or 0,
            width=row['width'] or 0,
            height=row['height'] or 0,
            audio=row['audio'] or 0,
            source=row['source'] or '',
            flags=row['flags'] or 0,
            user=row['user_name'] or '',
            mark=row['mark'] or 0,
            gift=row['gift'] or 0,
            tags=tags,
            comments=comments,
        )

    def iter_items(self, batch_size: int = 1000) -> Iterator[Item]:
        """Iterate over all items in batches."""
        cursor = self.conn.cursor()
        offset = 0

        while True:
            cursor.execute(
                'SELECT * FROM items ORDER BY id LIMIT ? OFFSET ?',
                (batch_size, offset)
            )
            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                yield self._row_to_item(row)

            offset += batch_size

    def iter_items_with_tags(
        self,
        min_tags: int = 5,
        image_only: bool = True,
        batch_size: int = 1000
    ) -> Iterator[Item]:
        """Iterate over items with minimum tags, optionally filtering to images only."""
        cursor = self.conn.cursor()
        offset = 0

        # Filter for image extensions
        image_filter = ""
        if image_only:
            image_filter = "AND (image LIKE '%.jpg' OR image LIKE '%.jpeg' OR image LIKE '%.png')"

        while True:
            cursor.execute(f'''
                SELECT * FROM items 
                WHERE json_array_length(tags_data) >= ?
                {image_filter}
                ORDER BY id 
                LIMIT ? OFFSET ?
            ''', (min_tags, batch_size, offset))

            rows = cursor.fetchall()
            if not rows:
                break

            for row in rows:
                yield self._row_to_item(row)

            offset += batch_size

    def get_all_unique_tags(self) -> list[str]:
        """Get all unique tag names from tags_data JSON column."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT tags_data FROM items WHERE tags_data IS NOT NULL')

        unique_tags = set()
        for row in cursor.fetchall():
            tags = json.loads(row['tags_data'] or '[]')
            for tag_obj in tags:
                unique_tags.add(tag_obj['tag'])

        return sorted(list(unique_tags))

    def get_tag_counts(self, limit: Optional[int] = 100) -> list[tuple[str, int]]:
        """Get most common tags with counts from tags_data JSON column.

        Args:
            limit: Maximum number of tags to return. If None, returns all tags.
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT tags_data FROM items WHERE tags_data IS NOT NULL')

        tag_counts = {}
        for row in cursor.fetchall():
            tags = json.loads(row['tags_data'] or '[]')
            for tag_obj in tags:
                tag_name = tag_obj['tag']
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1

        # Sort by count descending and return top N (or all if limit is None)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:limit] if limit else sorted_tags

