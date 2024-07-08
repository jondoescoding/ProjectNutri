import sqlite3

# Path to the SQLite database
db_path = r'D:\CODING\projectNutri\tests\database-test\client_health_improved.db'

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Function to get the schema of a table
def get_table_schema(cursor, table_name):
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    return cursor.fetchall()

# Retrieve all table names in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

# Open a file to write the schema information
with open('database_schema.txt', 'w') as file:
    for table_info in tables:
        table_name = table_info[0]
        # Write the table name to the file
        file.write(f"Table: {table_name}\n")
        file.write(f"{'-' * len(table_name)}\n")

        # Retrieve the table schema
        schema_info = get_table_schema(cursor, table_name)
        
        # Write the schema details to the file
        for column_info in schema_info:
            cid, name, ctype, notnull, dflt_value, pk = column_info
            file.write(f"Column ID: {cid}\n")
            file.write(f"Column Name: {name}\n")
            file.write(f"Data Type: {ctype}\n")
            file.write(f"Not Null: {notnull}\n")
            file.write(f"Default Value: {dflt_value}\n")
            file.write(f"Primary Key: {pk}\n\n")

# Close the database connection
conn.close()