import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. THE BRAIN ---
class DietOptimizerEP:
    def __init__(self, b_pool, l_pool, d_pool, s_pool, target_cal):
        self.pools = [b_pool, l_pool, d_pool, s_pool]
        self.target_cal = target_cal
        
    def evaluate(self, meals):
        total_cal = sum(m['Calories'] for m in meals)
        total_prot = sum(m['Protein'] for m in meals)
        total_fat = sum(m['Fat'] for m in meals)
        # Synthetic Price: $1.50 base + $0.005 per calorie
        total_cost = sum((m['Calories'] * 0.005) + 1.50 for m in meals)
        
        # Fitness: Penalty for calorie gap + high cost
        fitness = abs(total_cal - self.target_cal) + (total_cost * 10)
        
        return {
            'meals': meals, 'calories': total_cal, 'protein': total_prot,
            'fat': total_fat, 'cost': total_cost, 'fitness': fitness
        }

    def run(self, pop_size=30, gens=30):
        # Create initial population
        pop = []
        for _ in range(pop_size):
            meals = [pool.sample(n=1).iloc[0].to_dict() for pool in self.pools]
            pop.append(self.evaluate(meals))
            
        history = []
        for _ in range(gens):
            pop.sort(key=lambda x: x['fitness'])
            history.append(pop[0]['fitness'])
            # Mutate: Replace 1 meal in the top plans
            for i in range(len(pop)//2):
                new_meals = list(pop[i]['meals'])
                idx = np.random.randint(0, 4)
                new_meals[idx] = self.pools[idx].sample(n=1).iloc[0].to_dict()
                pop.append(self.evaluate(new_meals))
            pop = pop[:pop_size]
        return pop[0], history

# --- 2. THE UI ---
st.set_page_config(page_title="Diet Optimizer", layout="wide")
st.title("🥗 Diet Optimizer: Cost & Nutrition")

# LOAD DATA
if os.path.exists('Food_and_Nutrition__.csv'):
    df = pd.read_csv('Food_and_Nutrition__.csv')
    
    # Check if 'Fat' exists in CSV, if not, create it to prevent errors
    if 'Fat' not in df.columns:
        df['Fat'] = (df['Calories'] * 0.02).round(1)

    # Prepare Pools
    b_pool = df[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()
    l_pool = df[['Lunch Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()
    d_pool = df[['Dinner Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()
    s_pool = df[['Snack Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()

    target = st.sidebar.number_input("Target Calories", value=2000)

    if st.button("🚀 GENERATE MEAL PLAN"):
        optimizer = DietOptimizerEP(b_pool, l_pool, d_pool, s_pool, target)
        winner, history = optimizer.run()

        # --- THE RESULTS (Visible boxes) ---
        st.markdown("### 📊 DAILY TOTALS")
        col1, col2, col3, col4 = st.columns(4)
        col1.warning(f"**PRICE:** ${winner['cost']:.2f}")
        col2.info(f"**CALORIES:** {winner['calories']} kcal")
        col3.success(f"**PROTEIN:** {winner['protein']}g")
        col4.error(f"**FAT:** {winner['fat']}g")

        st.markdown("---")
        st.subheader("🍴 MEAL DETAILS")
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        for i, m in enumerate(winner['meals']):
            name = list(m.values())[0]
            st.write(f"**{labels[i]}:** {name} | {m['Calories']} kcal | {m['Fat']}g Fat")

        st.line_chart(history)
else:
    st.error("Missing CSV file!")
