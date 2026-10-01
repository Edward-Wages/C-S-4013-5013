import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. LOAD AND ENCODE THE DATA ----------
df = pd.read_csv("CreditCard.csv") #Read the CSV
enc = df.copy()

#Row 142 in the CSV is missing a value for Gender. We will drop the row to avoid errors when encoding.
enc = enc.dropna()

# Adjust the values of alphabetic columns to numeric values for the GA to evaluate them.
enc["Gender"] = enc["Gender"].map({"M": 1, "F": 0})
enc["CarOwner"] = enc["CarOwner"].map({"Y": 1, "N": 0})
enc["PropertyOwner"] = enc["PropertyOwner"].map({"Y": 1, "N": 0})

# X = the 6 attributes (gender ... email), shape (340, 6)
# y = CreditApprove, shape (340,)
# Selecting by position: col 0 = Applicant_ID, col 1 = CreditApprove, cols 2-7 = attributes
X = enc.iloc[:, 2:8].to_numpy(dtype=float)
y = enc.iloc[:, 1].to_numpy(dtype=float)


# ---------- 2. EVALUATE: score every chromosome ----------
def evaluate(population, X, y):
    # er(w) = average of (f(x_i) - y_i)^2 over all applicants.
    # X @ w computes f(x_i) for all 340 applicants at once
    er = np.array([np.mean((X @ w - y) ** 2) for w in population])

    # Fitness = e^(-er). 
    fitness = np.exp(-er)
    return er, fitness


# ---------- 3. THE GENETIC ALGORITHM ----------
def genetic_algorithm(X, y, pop_size=20, num_gens=50, mut_rate=0.05):
    # Initial population: pop_size random chromosomes, each a length-6 vector of -1/+1.
    # These are the ONLY things the GA changes; X and y are never modified.
    population = np.random.choice([-1, 1], size=(pop_size, 6))

    best_w, best_er = None, float("inf")   # best solution seen in any generation
    history = []                           # best er of each generation, for Figure 3

    for gen in range(num_gens):
        # Score the current population (See evaluate() for details)
        er, fitness = evaluate(population, X, y)

        # Record the best chromosome in this generation
        i = er.argmin()                  # index of the lowest error
        history.append(er[i])

        # Keep the best-ever solution (a later generation can get worse by chance)
        if er[i] < best_er:
            best_er = er[i]
            best_w = population[i].copy()

        # Selection probabilities: each chromosome's fitness divided by the total.
        # Fitter chromosomes are more likely to be chosen as parents.
        probs = fitness / fitness.sum()

        # Build the next generation, one child at a time, until it is full
        new_population = []
        while len(new_population) < pop_size:
            # Pick 2 parents (by index), weighted by probs
            idx = np.random.choice(pop_size, size=2, p=probs)
            parent_a, parent_b = population[idx[0]], population[idx[1]]

            # Crossover at the middle: first 3 genes from A, last 3 genes from B
            child = np.concatenate([parent_a[:3], parent_b[3:]])

            # Mutation: each gene flips sign (+1 <-> -1) with probability mut_rate
            flips = np.random.rand(6) < mut_rate
            child[flips] = -child[flips]

            new_population.append(child)

        population = np.array(new_population)   # the children replace the old generation

    return best_w, best_er, history


# ---------- 4. RUN AND REPORT ----------
best_w, best_er, history = genetic_algorithm(X, y)

print("w     =", best_w)       # equation (8)
print("er(w) =", best_er)      # equation (9)

plt.plot(history)
plt.xlabel("Generation")
plt.ylabel("er(w)")
plt.title("Genetic algorithm: er(w) vs. generation")
plt.show()