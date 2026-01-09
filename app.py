import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. THE BRAIN: Evolutionary Programming (EP) ---
class DietOptimizerEP:
    def __init__(self, b_pool, l_pool, d_pool, s_pool, target_cal):
        self.pools = [b_pool, l_pool, d_pool, s_pool]
        self.target_cal = target_cal
        
    def calculate_cost(self, calories):
        """Synthetic price: $1.50 base + $0.005 per calorie."""
        return (calories * 0.005) + 1.50

    def create_individual(self):
        """Random combination of 4 meals."""
        meals = [pool.sample(n=1).iloc[0].to_dict() for pool in self.pools]
        return self.evaluate(meals)

    def evaluate(self, meals):
        """Calculates Calories, Protein, Fat, and Cost."""
        total_cal = sum(m['Calories'] for m in meals)
        total_prot = sum(m['Protein'] for m in meals)
        total_fat = sum(m['Fat'] for m in meals) # Now calculating Fat
        total_cost = sum(self.calculate_cost(m['Calories']) for m in meals)
        
        # FITNESS FUNCTION (Multi-Objective)
        cal_error = abs(total_cal - self.target_cal)
        fitness = cal_error + (total_cost * 15)
        
        return {
            'meals': meals,
            'calories': total_cal,
            'protein': total_prot,
            'fat': total_fat, # Storing Fat
            'cost': total_cost,
            'fitness': fitness
        }

    def mutate(self, ind):
        """EP Mutation."""
        new_meals = list(ind['meals'])
        idx = np.random.randint(0, 4)
        new_meals[idx] = self.pools[idx].sample(n=1).iloc[0].to_dict()
        return self.evaluate(new_meals)

    def run(self, pop_size=50, generations=50):
        population = [self.create_individual() for _ in range(pop_size)]
        history = []
        for _ in range(generations):
            offspring = [self.mutate(p) for p in population]
            combined = population + offspring
            combined.sort(key=lambda x: x['fitness'])
            population = combined[:pop_size]
            history.append(population[0]['fitness'])
        return population[0], history

# --- 2. THE UI ---
st.set_page_config(page_title="Evolutionary Diet Optimizer", layout="wide")
st.title("🍱 Evolutionary Diet Meal Planner")

if os.path.exists('Food_and_Nutrition__.csv'):
    df = pd.read_csv('Food_and_Nutrition__.csv')
    
    # Pools including Fat
    b_pool = df[['Breakfast Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()
    l_pool = df[['Lunch Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()
    d_pool = df[['Dinner Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()
    s_pool = df[['Snack Suggestion', 'Calories', 'Protein', 'Fat']].drop_duplicates()

    with st.sidebar:
        st.header("🎯 Settings")
        target_cal = st.number_input("Daily Calorie Target", value=2000)
        pop_size = st.slider("Population", 10, 100, 50)
        gens = st.slider("Generations", 10, 100, 50)

    if st.button("🚀 Start Evolutionary Optimization"):
        solver = DietOptimizerEP(b_pool, l_pool, d_pool, s_pool, target_cal)
        winner, history = solver.run(pop_size, gens)

        # Display Metrics (Now with Fat!)
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Calories", f"{winner['calories']} kcal")
        m2.metric("Total Cost", f"${winner['cost']:.2f}")
        m3.metric("Protein", f"{winner['protein']}g")
        m4.metric("Fat", f"{winner['fat']}g") # Fat metric displayed here

        # Display Meals
        st.subheader("📋 Your Optimized Daily Menu")
        labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
        keys = ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]
        cols = st.columns(4)
        
        for i in range(4):
            with cols[i]:
                meal = winner['meals'][i]
                meal_name = meal[keys[i]]
                st.info(f"**{labels[i]}**")
                st.write(f"{meal_name}")
                st.write(f"*{meal['Calories']} kcal | {meal['Fat']}g Fat*")

        # Performance Graph
        st.divider()
        st.subheader("📈 Performance Analysis")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(history, color='#e67e22')
        ax.set_title("Algorithm Convergence (Penalty over Time)")
        st.pyplot(fig)
else:
    st.error("CSV Not Found!")
