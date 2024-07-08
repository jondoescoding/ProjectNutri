import sqlite3

# Establish a connection to the database (creates new if not existent)
conn = sqlite3.connect('client_health_improved.db')

# Create a cursor object
cursor = conn.cursor()

# Create 'Clients' table and index on 'LastName'
cursor.execute('''
CREATE TABLE IF NOT EXISTS Clients (
    ClientID INTEGER PRIMARY KEY,
    LastName TEXT NOT NULL,
    FirstName TEXT NOT NULL,
    Gender TEXT,
    Address TEXT,
    City TEXT,
    TelephoneNumber TEXT,
    Email TEXT,
    DOB DATE,
    CurrentLocation TEXT,
    FamilyDoctorName TEXT
)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_clients_lastname ON Clients (LastName)')

# StaticHealthProfile Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS StaticHealthProfile (
    ProfileID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    Height TEXT,
    Allergies TEXT,
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')



# Create 'DynamicHealthProfile' table and index on 'ClientID' and 'DateRecorded'
cursor.execute('''
CREATE TABLE IF NOT EXISTS DynamicHealthProfile (
    RecordID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    DateRecorded DATE DEFAULT (date('now')),
    Weight INTEGER,
    BodyFatPercentage REAL,
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_dynamichealth_client ON DynamicHealthProfile (ClientID)')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_dynamichealth_date ON DynamicHealthProfile (DateRecorded)')

# ...


# Create 'Supplements' table and index on 'ClientID'
cursor.execute('''
CREATE TABLE IF NOT EXISTS Supplements (
    SupplementID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    Name TEXT,
    Dosage TEXT,
    Frequency TEXT,
    Notes TEXT,
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_supplements_client ON Supplements (ClientID)')

# Activities Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Activities (
    ActivityID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    Occupation TEXT,
    WorkActivityLevel TEXT,
    ExerciseDetails TEXT,
    StressLevel INTEGER,
    SleepDetails TEXT,
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')



# Create 'DietaryHabits' table and index on 'ClientID'
cursor.execute('''
CREATE TABLE IF NOT EXISTS DietaryHabits (
    DietaryHabitID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    Schedule TEXT,
    LivingArrangement TEXT,
    GroceryShoppingResponsibility TEXT,
    CookingResponsibility TEXT,
    EatingOutFrequency INTEGER,
    DietDetails TEXT,
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_dietaryhabits_client ON DietaryHabits (ClientID)')

# ...

# Create 'MealPlan' table and index on 'ClientID'
cursor.execute('''
CREATE TABLE IF NOT EXISTS MealPlan (
    MealPlanID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    WeekNumber INTEGER,
    Notes TEXT,
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_mealplan_client ON MealPlan (ClientID)')

# ...

# Create 'Meals' table and index on 'MealPlanID'
cursor.execute('''
CREATE TABLE IF NOT EXISTS Meals (
    MealID INTEGER PRIMARY KEY,
    MealPlanID INTEGER,
    MealType TEXT,
    Description TEXT,
    FOREIGN KEY (MealPlanID) REFERENCES MealPlan (MealPlanID)
)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_meals_mealplan ON Meals (MealPlanID)')

# MealComponents Table (optional)
cursor.execute('''
CREATE TABLE IF NOT EXISTS MealComponents (
    ComponentID INTEGER PRIMARY KEY,
    MealID INTEGER,
    Ingredient TEXT,
    Quantity TEXT,
    Notes TEXT,
    FOREIGN KEY (MealID) REFERENCES Meals (MealID)
)
''')

# Create 'HealthConditions' table and index on 'ClientID'
cursor.execute('''
CREATE TABLE IF NOT EXISTS HealthConditions (
    ConditionID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    ConditionName TEXT,
    ConditionDetails TEXT,
    DateNoted DATE DEFAULT (date('now')),
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')
cursor.execute('CREATE INDEX IF NOT EXISTS idx_healthcond_client ON HealthConditions (ClientID)')

# ClientSelfAssessment Table
cursor.execute('''
CREATE TABLE IF NOT EXISTS ClientSelfAssessment (
    AssessmentID INTEGER PRIMARY KEY,
    ClientID INTEGER,
    DateAssessed DATE DEFAULT (date('now')),
    HealthGoal TEXT,
    Obstacles TEXT,
    ImportanceRating INTEGER,
    ReadinessRating INTEGER,
    ConfidenceRating INTEGER,
    FavouriteFood TEXT,
    FOREIGN KEY (ClientID) REFERENCES Clients (ClientID)
)
''')

# Commit changes and close the connection
conn.commit()
conn.close()