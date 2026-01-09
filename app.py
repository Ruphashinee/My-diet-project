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
        """Creates a random meal combination."""
        # We assume each meal costs $2-$10 based on its calories
        b = self.b_df.sample(n=1).iloc[0]
        l = self.l_df.sample(n=1).iloc[0]
        d = self.d_df.sample(n=1).iloc[0]
        s = self.s_df.sample(n=1).iloc[0]
        
        total_cal = b['Calories'] + l['Calories'] + d['Calories'] + s['Calories']
        total_prot = b['Protein'] + l['Protein'] + d['Protein'] + s['Protein']
        # Price calculation: $0.005 per calorie
        total_price = total_cal * 0.005 
        
        return {
            'meals': [b, l, d, s],
            'calories': total_cal,
            'price': total_price,
            'protein': total_prot,
            'fitness': 0
        }

    def calculate_fitness(self, ind):
        """Lower score is better. Penalty for calorie gap + price."""
        cal_gap = abs(ind['calories'] - self.target_cal)
        # Multi-objective: Minimize calorie gap AND minimize price
        return cal_gap + (ind['price'] * 10)

    def mutate(self, ind):
        """EP Mutation: Change ONE random meal in the plan."""
        new_ind = ind.copy()
        meal_to_change = np.random.randint(0, 4)
        if meal_to_change == 0: new_meal = self.b_df.sample(n=1).iloc[0]
        elif meal_to_change == 1: new_meal = self.l_df.sample(n=1).iloc[0]
        elif meal_to_change == 2: new_meal = self.d_df.sample(n=1).iloc[0]
        else: new_meal = self.s_df.sample(n=1).iloc[0]
        
        new_ind['meals'][meal_to_change] = new_meal
        # Recalculate stats
        new_ind['calories'] = sum(m['Calories'] for m in new_ind['meals'])
        new_ind['price'] = new_ind['calories'] * 0.005
        new_ind['protein'] = sum(m['Protein'] for m in new_ind['meals'])
        return new_ind

    def run(self, pop_size=50, gens=100):
        # Initial Population
        population = [self.create_individual() for _ in range(pop_size)]
        history = []

        for _ in range(gens):
            # Evaluate current population
            for ind in population:
                ind['fitness'] = self.calculate_fitness(ind)
            
            # Create offspring via mutation
            offspring = [self.mutate(p) for p in population]
            for o in offspring:
                o['fitness'] = self.calculate_fitness(o)
            
            # Selection: Survival of the fittest
            combined = population + offspring
            combined.sort(key=lambda x: x['fitness'])
            population = combined[:pop_size]
            history.append(population[0]['fitness'])
            
        return population[0], history

# --- 2. THE UI: Streamlit ---
st.set_page_config(page_title="Professional Diet Optimizer", layout="wide")
st.title("🍎 Multi-Objective Meal Combinator (EP)")

# Load and split data
csv_files = [f for f in os.listdir() if f.endswith('.csv')]
if csv_files:
    raw_df = pd.read_csv(csv_files[0])
    
    # Split into meal pools
    b_pool = raw_df[['Breakfast Suggestion', 'Calories', 'Protein']].drop_duplicates()
    l_pool = raw_df[['Lunch Suggestion', 'Calories', 'Protein']].drop_duplicates()
    d_pool = raw_df[['Dinner Suggestion', 'Calories', 'Protein']].drop_duplicates()
    s_pool = raw_df[['Snack Suggestion', 'Calories', 'Protein']].drop_duplicates()

    with st.sidebar:
        target = st.number_input("Target Daily Calories", value=2000)
        pop_s = st.slider("Population", 10, 100, 50)
        gen_s = st.slider("Generations", 10, 200, 100)

    if st.button("Generate Best Combination"):
        optimizer = DietOptimizerEP(b_pool, l_pool, d_pool, s_pool, target)
        winner, log = optimizer.run(pop_s, gen_s)

        # Show Results
        st.subheader("✅ Best Found Combination")
        cols = st.columns(3)
        cols[0].metric("Total Calories", f"{winner['calories']} kcal")
        cols[1].metric("Total Cost", f"${winner['price']:.2f}")
        cols[2].metric("Total Protein", f"{winner['protein']}g")

        # Menu Table
        meal_names = ["Breakfast", "Lunch", "Dinner", "Snack"]
        for i, m in enumerate(winner['meals']):
            name_col = list(m.keys())[0]
            st.write(f"**{meal_names[i]}:** {m[name_col]} ({m['Calories']} kcal)")

        # Analysis Graph
        st.line_chart(log)
        st.caption("Evolutionary Programming Performance: Fitness (Penalty) vs Generation")
else:
    st.error("Please upload the CSV file!")
