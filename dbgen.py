import sqlite3
import os

def create_sample_database(db_path='sample.db'):
    """Create a sample SQLite database with tables and data."""
    # Check if database already exists
    db_exists = os.path.exists(db_path)
    
    # Create a sample database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL NOT NULL
    )
    ''')
    
    # Only insert data if the database is newly created
    if not db_exists:
        # Insert sample data
        cursor.executemany('''
        INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)
        ''', [
            ('Alice Smith', 'Engineering', 95000.00),
            ('Bob Johnson', 'Marketing', 85000.00),
            ('Carol Williams', 'Engineering', 92000.00),
            ('Dave Brown', 'Finance', 88000.00),
            ('Eve Davis', 'HR', 78000.00)
        ])
    
    conn.commit()
    conn.close()
    
    return not db_exists  # Return True if we created a new database

# If script is run directly, create the default sample database
if __name__ == "__main__":
    created = create_sample_database()
    if created:
        print("Sample database created successfully!")
    else:
        print("Sample database already existed, table structure ensured.")