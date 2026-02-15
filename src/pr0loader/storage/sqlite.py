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
        """Connect to the database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info(f"Connected to database: {self.db_path}")

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

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER,
                tag TEXT,
                confidence REAL,
                FOREIGN KEY (item_id) REFERENCES items(id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_item ON tags(item_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)')

        self.conn.commit()

    def upsert_item(self, item: Item):
        """Insert or update an item."""
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

        # Update tags table
        cursor.execute('DELETE FROM tags WHERE item_id = ?', (item.id,))
        for tag in item.tags:
            cursor.execute(
                'INSERT INTO tags (item_id, tag, confidence) VALUES (?, ?, ?)',
                (item.id, tag.tag, tag.confidence)
            )

        self.conn.commit()

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
        """Get all unique tag names."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT tag FROM tags ORDER BY tag')
        return [row['tag'] for row in cursor.fetchall()]

    def get_tag_counts(self, limit: int = 100) -> list[tuple[str, int]]:
        """Get most common tags with counts."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT tag, COUNT(*) as count 
            FROM tags 
            GROUP BY tag 
            ORDER BY count DESC 
            LIMIT ?
        ''', (limit,))
        return [(row['tag'], row['count']) for row in cursor.fetchall()]

