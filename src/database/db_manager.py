import sqlite3
import os

class DBManager:
    def __init__(self, db_path="app.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER, email TEXT)")
        
        # Insert dummy data if table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            cursor.executemany("INSERT INTO users (name, age, email) VALUES (?, ?, ?)", [
                ("Alice", 28, "alice@example.com"),
                ("Bob", 32, "bob@example.com"),
                ("Charlie", 24, "charlie@example.com")
            ])
            
        conn.commit()
        conn.close()

    def execute_query(self, query: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                result = cursor.fetchall()
                # Get column names
                column_names = [description[0] for description in cursor.description]
                return [dict(zip(column_names, row)) for row in result]
            else:
                conn.commit()
                result = {"status": "success", "rows_affected": cursor.rowcount}
        finally:
            conn.close()
            
        return result
