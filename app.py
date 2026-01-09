import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. THE BRAIN: Evolutionary Programming ---
class DietOptimizerEP:
    def __init__(self, breakfast_df, lunch_df, dinner_df, snack_df, target_cal):
        self.b_df = breakfast_df
        self.l_df = lunch_df
        self.d_df = dinner_df
        self.s_df = snack_df
        self.target_cal = target_cal
        
    def create_individual(self):
        # We assign a realistic price based on calories for the simulation
        b = self.b_df.sample(n=1).iloc[0].to_dict()
        l = self.l_df.sample(n=1).iloc[0].to_dict()
        d = self.d_df.sample(n=1).iloc[0].to_dict()
        s = self.s_df.sample(n=1).iloc[0].to_dict()
        
        # Calculate totals
        meals = [b, l, d, s]
        total_cal = sum(m['Calories'] for m in meals)
        total_prot = sum(m['Protein'] for m in meals)
        # Price Formula: $2 base + small amount per calorie (minimizing this is the goal!)
        total_price = sum((m['Calories'] * 0.005) + 1.5 for m in meals)
        
        return {
            'meals': meals,
            'calories': total_cal,
            'price': total_price,
            'protein': total_prot,
            'fitness': 0
        }

    def calculate_fitness(self, ind):
        # Multi-Objective: Minimize Calorie Error AND Minimize Total Price
        cal_error = abs(ind['calories'] - self.target_cal)
        return cal_error + (ind['price'] * 15)  # Multiplying price by 15 makes cost very important

    def mutate(self, ind):
        new_ind = {
            'meals': list(ind['meals']),
            'calories': 0,
            'price': 0,
            'protein': 0
        }
        # Change ONE meal randomly
        idx = np.random.randint(0, 4)
        if idx == 0: new_ind['meals'][0] = self.b_df.sample(n=1).iloc[0].to_dict()
        elif idx == 1: new_ind['meals'][1] = self.l_df.sample(n=1).iloc[0].to_dict()
        elif idx == 2: new_ind['meals'][2] = self.d_df.sample(n=1).iloc[0].to_dict()
        else: new_ind['meals'][3] = self.s_df.sample(n=1).iloc[0].to_dict()
        
        # Update stats for the new combination
        new_ind['calories'] = sum(m['Calories'] for m in new_ind['meals'])
        new_ind['price'] = sum((m['Calories'] * 0.005) + 1.5 for m in new_ind['meals'])
        new_ind['protein'] = sum(m['Protein'] for m in new_ind['meals'])
        return new_ind

    def run(self, pop_size=50, gens=50):
        population = [self.create_individual() for _ in range(pop_size)]
        history = []
        for _ in range(gens):
            for ind in population: ind['fitness'] = self.calculate_fitness(ind)
            offspring = [self.mutate(p) for p in population]
            for o in offspring: o['fitness'] = self.calculate_fitness(o)
            combined = population + offspring
            combined.sort(key=lambda x: x['fitness'])
            population = combined[:pop_size]
            history.append(population[0]['fitness'])
        return population[0], history

# --- 2. THE UI: Streamlit Dashboard ---
st.set_page_config(page_title="Evolutionary Diet Optimizer", layout="wide")
st.title("🥗 Evolutionary Diet Optimizer")
st.write("Using **Evolutionary Programming** to minimize cost and meet nutrition goals.")

# Load Data
if os.path.exists('Food_and_Nutrition__.csv'):
    df = pd.read_csv('Food_and_Nutrition__.csv')
    b_pool = df[['Breakfast Suggestion', 'Calories', 'Protein']].drop_duplicates()
    l_pool = df[['Lunch Suggestion', 'Calories', 'Protein']].drop_duplicates()
    d_pool = df[['Dinner Suggestion', 'Calories', 'Protein']].drop_duplicates()
    s_pool = df[['Snack Suggestion', 'Calories', 'Protein']].drop_duplicates()

    # Sidebar
    st.sidebar.header("Targets")
    target_c = st.sidebar.slider("Calories Goal", 1200, 3500, 2000)
    
    if st.sidebar.button("Run Optimizer"):
        solver = DietOptimizerEP(b_pool, l_pool, d_pool, s_pool, target_c)
        winner, history = solver.run()

        # --- RESULTS AREA ---
        st.subheader("✅ Optimized Results")
        col1, col2, col3 = st.columns(3)
        
        # HERE IS THE COST!
        col1.metric("Total Calories", f"{winner['calories']} kcal")
        col2.metric("MINIMIZED COST", f"${winner['price']:.2f}", delta="Lowest Found")
        col3.metric("Total Protein", f"{winner['protein']}g")

        st.markdown("### 🍽️ Recommended Meal Combination")
        labels = ["🍳 Breakfast", "🥗 Lunch", "🍲 Dinner", "🍎 Snack"]
        keys = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        
        for i in range(4):
            meal_name = winner['meals'][i][keys[i]]
            meal_price = (winner['meals'][i]['Calories'] * 0.005) + 1.5
            st.write(f"**{labels[i]}:** {meal_name} — *Cost: ${meal_price:.2f}*")

        # --- GRAPH ---
        st.markdown("---")
        st.subheader("📈 Performance Analysis (Convergence)")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(history, color='#1E88E5', linewidth=2)
        ax.set_ylabel("Fitness (Penalty Score)")
        ax.set_xlabel("Generation")
        st.pyplot(fig)
        st.info("The graph shows the 'Penalty' dropping. This means the AI found a plan that is closer to the calorie goal and CHEAPER over time.")
else:
    st.error("CSV file not found. Please upload 'Food_and_Nutrition__.csv' to GitHub.")
