import sqlite3

DB_PATH = "local_cache.db"

def init_db():
    """Initializes the SQLite database tables with the updated enterprise schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Employees Table (With FatherName and Designation)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Employees (
            EmployeeID TEXT PRIMARY KEY,
            Name TEXT NOT NULL,
            FatherName TEXT,
            Department TEXT,
            Designation TEXT,
            FaceEncoding BLOB,
            Status TEXT DEFAULT 'Active'
        )
    ''')
    
    # 2. Attendance Logs Table (Session-based matching row format)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            EmployeeID TEXT,
            LogDate TEXT,          -- YYYY-MM-DD
            TimeIn TEXT,           -- HH:MM:SS (Date)
            TimeOut TEXT,          -- HH:MM:SS (Date)
            TotalDutyHours TEXT,   -- e.g., "08:15"
            CameraID INTEGER,
            Confidence REAL,
            Synced INTEGER DEFAULT 0,
            FOREIGN KEY(EmployeeID) REFERENCES Employees(EmployeeID)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database cache storage files verified and healthy.")

if __name__ == "__main__":
    init_db()