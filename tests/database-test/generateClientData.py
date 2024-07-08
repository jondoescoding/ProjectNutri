import random
from faker import Faker
import sqlite3

fake = Faker()


# Constants based on provided ratios and requirements
GENDER_RATIO = ['Male'] * 6 + ['Female'] * 4
IBS_RATIO = ['IBS'] * 5 + ['Healthy'] * 3 + ['Random Ailment'] * 2
DIET_PREFERENCES = ['Omnivore'] * 8 + ['Vegetarian', 'Vegan']
HEALTH_STATUS = ['Very Healthy'] * 2 + ['Moderately Healthy'] * 3 + ['Poor Health'] * 5

RANDOM_AILMENTS = ['Asthma', 'Diabetes']
HEALTHY_METRICS = {
    'Very Healthy': (22, 10),   # BMI, Body Fat Percentage
    'Moderately Healthy': (26, 20),
    'Poor Health': (30, 30)
}

# Extended list of Jamaican cuisine dishes, roughly categorized by macronutrients

# High in protein
JAMAICAN_PROTEINS = [
    'Jerk chicken', 'Grilled snapper', 'Curried chicken', 'Stewed pork',
    'Pepper steak', 'Brown stew chicken', 'Curried goat', 'Curried shrimp',
    'Escoveitch fish', 'Saltfish fritters', 'Ackee and saltfish',
    'Rundown (fish)', 'Oxtail with butter beans', 'Jamaican patty (beef)',
    'Manish water (goat soup)', 'Steamed fish', 'Chicken foot soup',
    'Cow foot', 'Tripe and beans'
]

# High in carbohydrates
JAMAICAN_CARBS = [
    'Rice and peas', 'Jamaican patty (vegetable)', 'Fried plantains', 'Bammy',
    'Festival', 'Coco bread', 'Sweet potato pudding', 'Pumpkin rice',
    'Cornmeal porridge', 'Johnny cakes', 'Rice porridge', 'Breadfruit',
    'Cassava pone', 'Gizzada', 'Bulla', 'Toto (dessert)', 'Jamaican sorrel cake',
    'Potato pudding', 'Stewed peas with dumplings'
]

# High in fats
JAMAICAN_FATS = [
    'Jamaican patty (cheese)', 'Mackerel run down', 'Coconut gizzada',
    'Stamp and go (saltfish cakes)', 'Grater cake', 'Dukunoo',
    'Solomon gundy (herring spread)', 'Jamaican blue draws (dessert)',
    'Ackee and saltfish with coconut oil', 'Roast breadfruit with ackee and saltfish',
    'Peanut porridge', 'Coconut drops', 'Grilled lobster with butter',
    'Jerk pork', 'Jerk sausage', 'Coconut curry fish', 'Spicy shrimp with coconut sauce',
    'Banana fritters', 'Plantain tarts'
]

# Combine the lists to create a diverse list of Jamaican dishes
JAMAICAN_CUISINE = JAMAICAN_PROTEINS + JAMAICAN_CARBS + JAMAICAN_FATS

# Constants for health status and gender ratio
HEALTH_STATUS = ['Very Healthy'] * 2 + ['Moderately Healthy'] * 3 + ['Poor Health'] * 5
GENDER_RATIO = ['Male'] * 6 + ['Female'] * 4
IBS_AVOIDANCE_LIST = JAMAICAN_FATS  # Assuming clients with IBS should avoid high-fat dishes

# Function to get health issue status
def get_health_status():
    status = random.choice(HEALTH_STATUS)
    bmi, body_fat = HEALTHY_METRICS[status]
    if status == 'Poor Health':
        bmi += random.randint(5, 15)
        body_fat += random.randint(5, 15)
    return status, bmi, body_fat

# Function to get a random ailment
def get_ailment(health):
    if health == 'IBS':
        return health
    elif health == 'Random Ailment':
        return random.choice(RANDOM_AILMENTS)
    else:
        return 'None'

# Function to get meals according to dietary needs
def get_meals(client_health_issue):
    if client_health_issue == 'IBS':
        suitable_meals = list(set(JAMAICAN_CUISINE) - set(IBS_AVOIDANCE_LIST))
    else:
        suitable_meals = JAMAICAN_CUISINE

    # Sample 3 meals for the client
    return random.sample(suitable_meals, 3)

# Connect to the SQLite database (replace 'path_to_database.db' with the actual database file path)
conn = sqlite3.connect(r'D:\CODING\projectNutri\tests\database-test\client_health_improved.db')
cursor = conn.cursor()

# Generate sample client data and include meals with dietary consideration
clients_data = []
used_ids = set()  # Keep track of used IDs to ensure uniqueness

for _ in range(10):
    # Generate a unique ID no more than 4 digits long
    client_id = random.randint(1000, 9999)
    while client_id in used_ids:
        client_id = random.randint(1000, 9999)
    used_ids.add(client_id)

    health_status, bmi, body_fat = get_health_status()
    health_issue = get_ailment(random.choice(IBS_RATIO))
    client_meals = get_meals(health_issue)
    
    client = {
        'Client_ID': client_id,
        'First_Name': fake.first_name(),
        'Last_Name': fake.last_name(),
        'Gender': random.choice(GENDER_RATIO),
        'DOB': fake.date_of_birth(minimum_age=20, maximum_age=60).strftime('%Y-%m-%d'),
        'Health_Issue': health_issue,
        'Diet': random.choice(DIET_PREFERENCES),
        'Meals': client_meals,
        'Health_Status': health_status,
        'BMI': bmi,
        'Body_Fat_Percentage': body_fat,
        'Occupation': fake.job(),
        'Stress_Level': random.randint(1, 10),
        'Exercise_Habit': random.choice(['Active', 'Moderate', 'Sedentary'])
    }
    clients_data.append(client)

    # Insert client data into the Clients table
    cursor.execute('''
    INSERT INTO Clients (ClientID, FirstName, LastName, DOB, Gender)
    VALUES (?, ?, ?, ?, ?)
''', (client['Client_ID'], client['First_Name'], client['Last_Name'], client['DOB'], client['Gender']))

# Insert client health issue into the HealthConditions table
    cursor.execute('''
    INSERT INTO HealthConditions (ClientID, ConditionName, DateNoted)
    VALUES (?, ?, date('now'))
''', (client['Client_ID'], client['Health_Issue']))

# Insert client occupation and stress level into the Activities table
    cursor.execute('''
    INSERT INTO Activities (ClientID, Occupation, StressLevel)
    VALUES (?, ?, ?)
''', (client['Client_ID'], client['Occupation'], client['Stress_Level']))

# Insert client diet details into the DietaryHabits table
    cursor.execute('''
    INSERT INTO DietaryHabits (ClientID, DietDetails)
    VALUES (?, ?)
''', (client['Client_ID'], client['Diet']))

# Insert client dynamic health profile into the DynamicHealthProfile table
    cursor.execute('''
    INSERT INTO DynamicHealthProfile (ClientID, Weight, BodyFatPercentage)
    VALUES (?, ?, ?)
''', (client['Client_ID'], client['BMI'], client['Body_Fat_Percentage']))

# Generate fake StaticHealthProfile data
    height = fake.random_int(min=150, max=200)  # Height in centimeters
    allergies = fake.word()  # Placeholder for allergies
    cursor.execute('''
        INSERT INTO StaticHealthProfile (ClientID, Height, Allergies)
    VALUES (?, ?, ?)
''', (client['Client_ID'], height, allergies))

# Generate fake Supplements data
    supplement_name = fake.word()
    dosage = f"{fake.random_int(min=1, max=500)} mg"
    frequency = f"{fake.random_int(min=1, max=3)} times a day"
    supplement_notes = fake.sentence()
    cursor.execute('''
    INSERT INTO Supplements (ClientID, Name, Dosage, Frequency, Notes)
    VALUES (?, ?, ?, ?, ?)
''', (client['Client_ID'], supplement_name, dosage, frequency, supplement_notes))

    # Generate and insert MealPlan data
    week_number = fake.random_int(min=1, max=52)
    meal_plan_notes = fake.sentence()
    cursor.execute('''
        INSERT INTO MealPlan (ClientID, WeekNumber, Notes)
        VALUES (?, ?, ?)
    ''', (client_id, week_number, meal_plan_notes))
    meal_plan_id = cursor.lastrowid  # Get the last inserted id to use as the MealPlanID

    # Generate and insert Meals data
    for meal in client['Meals']:
        meal_type = fake.random_element(elements=('Breakfast', 'Lunch', 'Dinner', 'Snack'))
        cursor.execute('''
            INSERT INTO Meals (MealPlanID, MealType, Description)
            VALUES (?, ?, ?)
        ''', (meal_plan_id, meal_type, meal))


    # Generate fake MealComponents data
    ingredient = fake.word()
    quantity = f"{fake.random_int(min=1, max=500)} grams"
    meal_component_notes = fake.sentence()
    cursor.execute('''
    INSERT INTO MealComponents (MealID, Ingredient, Quantity, Notes)
    VALUES (?, ?, ?, ?)
''', (1, ingredient, quantity, meal_component_notes))  # Assuming MealID is 1 for simplicity

    # Generate fake ClientSelfAssessment data
    health_goal = fake.sentence()
    obstacles = fake.sentence()
    importance_rating = fake.random_int(min=1, max=10)
    readiness_rating = fake.random_int(min=1, max=10)
    confidence_rating = fake.random_int(min=1, max=10)
    favourite_food = fake.word()
    cursor.execute('''
    INSERT INTO ClientSelfAssessment (ClientID, HealthGoal, Obstacles, ImportanceRating, ReadinessRating, ConfidenceRating, FavouriteFood)
    VALUES (?, ?, ?, ?, ?, ?, ?)
''', (client['Client_ID'], health_goal, obstacles, importance_rating, readiness_rating, confidence_rating, favourite_food))
    
    conn.commit()

# Close the database connection
conn.close()