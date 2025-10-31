# C51 (Distributional RL)

This project implements the **C51 (Categorical)** distributional reinforcement learning algorithm.

The goal is to compare the performance of a standard Deep Q-Network (DQN) against the C51 algorithm in a stochastic environment (`NoisyCartPole`).

* **DQN** learns the *average* expected return (Q-value) for each action.
* **C51** learns the full *probability distribution* over a set of 51 possible returns ("atoms").

In a noisy environment, learning the full distribution provides a more stable and complete picture of risk and reward, often leading to superior performance.

## File Structure

```text
.
├── DQNResults.png
├── README.md
└── src
├── C51.py
├── DQN.py
├── **init**.py
└── utils
├── **init**.py
├── buffers.py
├── envs.py
├── seed.py
└── torch.py
```

## How to Run

### 1. Setup Environment

Ensure you have a Python environment with the required packages installed (`torch`, `numpy`, `matplotlib`, `gymnasium`).

Activate your virtual environment:

```bash
conda activate <your_env_name>
````

### 2\. Run Baseline (DQN)

This script will run the baseline DQN agent in the `NoisyCartPole` environment and generate a plot (`DQNResults.png`) showing its performance.

```bash
python -m src.DQN
```

### 3\. Run C51 Algorithm

This script will run the implemented C51 agent. It will produce a new plot showing its performance, which can be compared against the DQN baseline.

```bash
python -m src.C51
```

## Results

* **`DQNResults.png`**: The saved plot showing the baseline performance of the standard DQN.
* The `C51.py` script will generate a new plot. This should be compared to the DQN baseline to analyze the impact of learning a full reward distribution in a stochastic environment.

## Updating Distributional RL `Z` Networks

### 1. `update_networks` — Finding the Target Probabilities

* **Sampling:** "Sampling... is just grabbing the states, actions, rewards, next states, and done booleans from a random sample of steps stored in our buffer."
* **Getting Logits:** "`Zt(S2)` spits out... the LOGITS (not probabilities) of each atom for that action."
* **Softmax:** "We softmax over those atoms... to turn those atom logits into a probability distribution."
* **Getting Q-Values:** "We then multiply each of those atom probabilities by their values... 'z'. Then we take the weighted sum of each of those atom values with their probabilities... giving us the q-values of each action..."
* **Finding Next Action:** "These q-values tell us the next action our policy would take at S2..."
* **Gathering the Target:** "...each of THOSE respective actions have atom probabilities associated with them... which is why we gather `target_atom_probs`."
* **The Result:** "what we have achieved is a `BATCH_SIZE... x ATOMS` tensor that tells us the probability of each atom given S2 and the action our policy will take..."

---

### 2. `update_networks` — Building the Target Distribution

* **The Goal:** "We want to calculate an estimate for the atom probabilities over the entire minibatch... Each step... corresponds with a probability distribution over the atoms. THAT'S what we're trying to build a target value for."
* **The Process (Per Atom):**
    1. "FIRST, get the projected reward... which is the batch step's reward plus the discounted future reward of the atom stored in `z[j]`..."
    2. "We clip it to stay within range..."
    3. "...that value will probably land in between two buckets in `z` - each of which gets a weight based on how far that bucket value is from our projected atom value."
    4. "...the probability of both those buckets must increase..."
    5. "So we add to `target_distributions`... By the probability we had for atom `j` TIMES the bin weights for each of those two bins."

---

### 3. `update_networks` — Calculating the Loss

* **Get Predictions:** "Now that we have our target probability distributions... we calculate our Z networks values... by taking `Z(S)`."
* **Isolate Predictions:** "This gives us atom LOGITS, and we only take the ones associated with the actions we took... This gives us a `BATCH_SIZE x ATOMS` tensor..."
* **Log-Softmax:** "But we want the log probabilities of these atoms; not their logits. So that's why we log softmax."
* **Loss Calculation:** "Now we define our loss to be the... product and sum... over each atom log probability distribution, and... take the mean... The larger this is the better we are doing, so define loss as the OPPOSITE of this since we want to minimize loss."
