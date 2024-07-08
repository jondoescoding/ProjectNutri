import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect(r'D:\CODING\LOCAL\projectNutri\tests\agents\client_health_data.db')
c = conn.cursor()


# SQL statement to create a new table named 'treatment_plan' with the specified columns
# and setting 'client_id' as the PRIMARY KEY
create_treatment_plan_table_sql = """
CREATE TABLE treatment_plan (
    client_id INTEGER PRIMARY KEY,
    treatment_and_meal_plan TEXT,
    timestamp DATETIME
)
"""

# Execute the SQL statement to create the new 'treatment_plan' table
c.execute(create_treatment_plan_table_sql)

# Commit the changes and close the connection
conn.commit()
conn.close()