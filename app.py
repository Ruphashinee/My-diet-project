import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PROFESSIONAL CLASS STRUCTURE ---
class DietEvolutionaryProgramming:
    """
    Implementation of Evolutionary Programming (EP) for Diet Optimization.
    Focuses on Mutation and Selection (no crossover) to simulate species evolution.
    """
    def __init__(self, data, target_calories):
        self.data = data
        self.target_calories = target_calories
        # Professional tip: Define weights for Multi-Objective optimization
        self.weight_calories = 1.0  # Priority for meeting energy needs
        self.weight_cost = 5.0      # Priority for minimizing cost
        
    def calculate_fitness(self, individual):
        """Calculates a fitness score. Lower is better (Cost + Penalty)."""
        cal_error = abs(individual['Calories'] - self.target_calories)
        cost = individual['Price']
        # Multi-objective formula: Combine both goals into one fitness value
        return (cal_error * self.weight_calories) + (cost * self.weight_cost)

    def mutate(self, individual):
        """EP Mutation: Creates a variation by choosing a new random meal plan."""
        return self.data.sample(n=1).iloc[0]

    def evolve(self, pop_size=50, generations=100):
        """The main evolutionary loop."""
        # Initialize Population
        population = [self.data.sample(n=1).iloc[0] for _ in range(pop_size)]
        history = []

        for gen in range(generations):
            # Generate Offspring via Mutation (Cloning and changing)
            offspring = [self.mutate(p) for p in population]
            
            # Combine and Evaluate
            total_pool = population + offspring
            # Sorting by fitness (Professional lambda function)
            total_pool.sort(key=lambda x: self.calculate_fitness(x))
            
            # Survival of the Fittest: Select the top 'pop_size'
            population = total_pool[:pop_size]
            
            # Track progress
            best_fitness = self.calculate_fitness(population[0])
            history.append(best_fitness)
            
        return population[0], history

# --- STREAMLIT DASHBOARD ---
def main():
    st.set_page_config(page_title="EP Diet Optimizer", layout="centered")
    st.title("🛡️ Professional Evolutionary Programming")
    st.markdown("### Case Study: Multi-Objective Diet Optimization")

    # Data Ingestion
    try:
        df = pd.read_csv('Food_and_Nutrition__.csv')
        # Realistic Cost Modeling
        df['Price'] = (df['Calories'] * 0.005) + (df['Protein'] * 0.02)
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return

    # User Parameters
    with st.sidebar:
        st.header("Algorithm Hyperparameters")
        target_cal = st.number_input("Target Calories", value=2000, step=100)
        pop_size = st.slider("Population Size", 20, 200, 100)
        generations = st.slider("Generations", 10, 200, 100)

    if st.button("Execute Optimization"):
        # Instantiate and Run
        optimizer = DietEvolutionaryProgramming(df, target_cal)
        best_sol, log = optimizer.evolve(pop_size, generations)

        # Output Results
        st.success("Optimal Solution Identified")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Calories", f"{best_sol['Calories']}")
        m2.metric("Estimated Cost", f"${best_sol['Price']:.2f}")
        m3.metric("Protein", f"{best_sol['Protein']}g")

        with st.expander("View Selected Menu Details"):
            st.write(f"**Breakfast:** {best_sol['Breakfast Suggestion']}")
            st.write(f"**Lunch:** {best_sol['Lunch Suggestion']}")
            st.write(f"**Dinner:** {best_sol['Dinner Suggestion']}")

        # Performance Visualization
        st.subheader("Algorithm Convergence Analysis")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(log, label='Best Fitness', color='#1f77b4', linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (Penalty Score)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
