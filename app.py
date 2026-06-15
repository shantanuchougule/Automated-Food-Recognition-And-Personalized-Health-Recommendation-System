# app.py
from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array  # type: ignore[reportMissingImports]
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the food recognition CNN model
cnn_model = tf.keras.models.load_model('FV.h5')

# Labels for classification
class_names = { 
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
    7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic',
    13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate',
    27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato',
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Function to predict image label
def predict_food_category(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = cnn_model.predict(img)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class].capitalize()

# Gemini interaction
def get_gemini_response(prompt, image, user_data):
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content([prompt, image[0], user_data])
    return response.text

# Format for uploading image
def input_image_setup(uploaded_file):
    if uploaded_file:
        return [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]
    raise FileNotFoundError("No file uploaded")

# PDF report generator
def generate_pdf_report(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Report", ln=True, align="C")
    for line in text.split('\n'):
        safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 10, safe_line)
    pdf.output(filename)

# App Configuration
st.set_page_config(page_title="AI Health Recommendation")
st.title("🤖 AI-Powered Health Recommendation System")

# Sidebar Navigation
page = st.sidebar.selectbox("Go to", ["🍎 Food & Diet Recommendation", "💪 Workout Recommendation", "🍽️ Recipe Generator"])

# ----------------- FOOD & DIET RECOMMENDATION -------------------

if page == "🍎 Food & Diet Recommendation":
    st.header("🍎 Food Recognition & Diet Recommendation")
    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display image after upload
        st.image(uploaded_file, caption="Uploaded Image", width=400)  # Sets fixed width in pixels


        # Add a button to trigger recognition
        if st.button("🔍 Recognize Food"):
            with st.spinner("Food Recognition..."):
                os.makedirs('./upload_images', exist_ok=True)
                save_path = f'./upload_images/{uploaded_file.name}'
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                result = predict_food_category(save_path)
                category = "Fruit" if result in fruits else "Vegetable"
                st.info(f"**Category: {category}**")
                st.success(f"**Predicted Food: {result}**")

                os.remove(save_path)  # Clean up
    else:
        st.warning("Upload an image to begin food recognition.")



    # Dietary Inputs
    ingredients = st.text_area("Ingredients & Measurements (mango:20g, strawberry:30g, kivi:15g, cherry:30g, blueberry:30g, orange:20g)")
    age = st.number_input("Age", min_value=1, value=22)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=10, value=57)
    height = st.number_input("Height (cm)", min_value=100, value=164)
    waist = st.number_input("Waist size (cm)", min_value=15, value=80)
    neck = st.number_input("Neck size (cm)", min_value=1, value=40)
    activity = st.selectbox("Workout Frequency", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
    diet_preference = st.selectbox("Diet Preference", ["Veg", "Non-Veg"])
    disease = st.selectbox("Medical Condition", ["None", "Diabetes", "Obesity", "Heart Failure", "Kidney Disease", "Cancer", "Others"])
    region = st.text_input("Region", value="India")
    allergies = st.text_input("Allergies", value="None")
    food_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])

    if st.button("Get Diet Recommendation"):
        with st.spinner("Analyzing food and generating personalized diet plan..."):
            try:
                image_data = input_image_setup(uploaded_file)
                user_data = f"""Age: {age}, Gender: {gender}, Weight: {weight}, Height: {height}, Waist: {waist}, Neck: {neck}, Activity: {activity}, Diet Preference: {diet_preference}, Disease: {disease}, Region: {region}, Allergies: {allergies}, Food Type: {food_type}, Ingredients: {ingredients}"""
                prompt = """
    You are an expert in nutrition. You need to recognize the food and analyze the food items from the image.
    Calculate the estimated calorie details of each food item and the total calories intake in the following format:

    Table format with columns:
        Item | Total Calories | Protein | Carbs | Fats | Fiber | Vitamins
        Row for each food item: Item 1, Item 2, Item 3, ...

    ---

    After that, consider the following user input values:
    age, gender, weight, height, diet_preference, disease, region, food_type

    Based on this information:
    1. Tell whether the food is healthy for the user or not.
    2. Provide personalized **Diet Recommendations** (including 4 daily meals with measured calorie values).
    3. Suggest:
    - Foods to take
    - Foods to avoid
    4. Offer additional nutrition tips.

    > Food Recognition
    > Calorie Estimation
    > Food Analysis
    > Diet Recommendation (4 meals + tips)
    """
                response = get_gemini_response(prompt, image_data, user_data)
                st.subheader("Nutrition Report")
                st.write(response)
                generate_pdf_report(response, "diet_recommendation_report.pdf")
                with open("diet_recommendation_report.pdf", "rb") as pdf_file:
                    st.download_button("Download Diet Report", pdf_file.read(), file_name="diet_recommendation_report.pdf")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ----------------- WORKOUT RECOMMENDATION -------------------
elif page == "💪 Workout Recommendation":
    st.header("💪 Workout & Fitness Recommendation")
    age = st.number_input("Age", value=22, key="age_wr")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender_wr")
    weight = st.number_input("Weight (kg)", value=57, key="weight_wr")
    height = st.number_input("Height (cm)", value=164, key="height_wr")
    waist = st.number_input("Waist (cm)", value=80, key="waist_wr")
    neck = st.number_input("Neck (cm)", value=40, key="neck_wr")
    activity = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"], key="activity_wr")
    goal = st.selectbox("Weight Goal", ["Weight Loss", "Maintain Weight", "Weight Gain"], key="goal_wr")
    change = st.number_input("Target Weight Change (kg)", min_value=0.0, step=0.5, value=2.0, key="weight_change")
    disease = st.selectbox("Medical Condition", ["None", "Diabetes", "Obesity", "Heart Failure", "Kidney Disease", "Cancer", "Others"], key="disease_wr")

    if st.button("Generate Workout Plan"):
        with st.spinner("Generating personalized workout plan..."):
            try:
                user_data = f"Age: {age}, Gender: {gender}, Weight: {weight}kg, Height: {height}cm, Waist: {waist}cm, Neck: {neck}cm, Activity: {activity}, Weight Goal: {goal}, Target Change: {change}kg, Disease: {disease}"
                image_data = [{"mime_type": "text/plain", "data": b"placeholder"}]
                workout_prompt = """
    You are a fitness and health expert. Using the following user details:

    Age, Gender, Weight (kg), Height (cm), Waist (cm), Neck (cm), Activity Level, Weight Goal, Desired Weight Change (kg), Medical Condition

    Please perform the following:

    1. Calculate **BMI (Body Mass Index)** and classify it.
    2. Calculate **BFP (Body Fat Percentage)** using:
    BFP = 86.010 × log10(Waist - Neck) - 70.041 × log10(Height) + 36.76
    3. Estimate the user's **Daily Calorie Requirement** based on profile, activity, and goal.
    4. Suggest a **Weekly Workout Plan** (Day | Exercise | Duration), focused on their weight goal.
    5. Offer **Additional Tips** for achieving the target weight safely and effectively.
    """

                response = get_gemini_response(workout_prompt, image_data, user_data)
                st.subheader("Workout Report")
                st.write(response)
                generate_pdf_report(response, "workout_recommendation.pdf")
                with open("workout_recommendation.pdf", "rb") as pdf_file:
                    st.download_button("Download Workout Report", pdf_file.read(), file_name="workout_recommendation.pdf")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# ----------------- RECIPE GENERATOR -------------------
elif page == "🍽️ Recipe Generator":
    st.header("🍽️ AI-Powered Recipe Generator")
    dish_name = st.text_input("Dish Name:")
    servings = st.number_input("Number of People", min_value=1, value=2)

    if st.button("Generate Recipe"):
        with st.spinner("Creating delicious recipe..."):
            try:
                image_data = [{"mime_type": "text/plain", "data": b"placeholder"}]
                user_data = f"Dish Name: {dish_name}, People: {servings}"
                recipe_prompt = """
    You are a professional chef and nutritionist.
    Given a dish name and number of people, generate:

    1. A detailed **Ingredients list** (scaled appropriately for the number of people) in tabular format
    2. **Step-by-step cooking instructions**.
    3. **Nutrition Information** (Calories, Protein, Carbs, Fats) per serving.
    Calculate the estimated calorie details of each food item and the total calories intake in the following format:
    Table format with columns:
        Item | Total Calories | Protein | Carbs | Fats | Fiber | Vitamins
        Row for each food item: Item 1, Item 2, Item 3, ...
    4. **Estimated Making Cost** (in indian rupees)
    ---

    Be very clear, concise, and friendly in your explanation.
    """
                response = get_gemini_response(recipe_prompt, image_data, user_data)
                st.subheader(f"Recipe for {dish_name}")
                st.write(response)
                generate_pdf_report(response, "recipe_generated.pdf")
                with open("recipe_generated.pdf", "rb") as pdf_file:
                    st.download_button("Download Recipe", pdf_file.read(), file_name="recipe_generated.pdf")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
