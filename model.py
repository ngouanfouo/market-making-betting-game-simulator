"""
Market-Making & Betting-Game Simulator

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - expected_value
def expected_value(values, probabilities):
    # Convert inputs to numpy arrays for consistent handling
    values_arr = np.asarray(values, dtype=float)
    probs_arr = np.asarray(probabilities, dtype=float)
    
    # Compute expected value: sum of value * probability
    # Using np.dot handles the element-wise multiplication and sum in one operation
    ev = np.dot(values_arr, probs_arr)
    
    # Return as a Python float
    return float(ev)

# Step 2 - one_reroll_die_value
def one_reroll_die_value(sides):
    # Create array of faces 1..sides
    faces = np.arange(1, sides + 1)
    
    # Create equal probabilities for each face
    prob = 1.0 / sides
    probabilities = np.full(sides, prob)
    
    # Expected value of a single roll (the reroll value)
    mu = expected_value(faces, probabilities)
    
    # Under optimal policy: keep if face >= mu, otherwise reroll
    # Payout for each first roll is max(face, mu)
    payouts = np.maximum(faces, mu)
    
    # Expected winnings under optimal policy
    value = expected_value(payouts, probabilities)
    
    # Faces to reroll: those strictly less than mu
    reroll_faces = [int(f) for f in faces if f < mu]
    
    return {
        'value': float(value),
        'reroll_faces': reroll_faces
    }

# Step 3 - pay_per_reroll_die_game (not yet solved)
# TODO: implement

# Step 4 - red_black_card_game_value (not yet solved)
# TODO: implement

# Step 5 - make_quotes (not yet solved)
# TODO: implement

# Step 6 - execute_trade (not yet solved)
# TODO: implement

# Step 7 - mark_to_market_pnl (not yet solved)
# TODO: implement

# Step 8 - adverse_selection_loss (not yet solved)
# TODO: implement

# Step 9 - uncertainty_spread (not yet solved)
# TODO: implement

# Step 10 - inventory_skewed_quotes (not yet solved)
# TODO: implement

# Step 11 - update_fair_value_from_trade (not yet solved)
# TODO: implement

# Step 12 - update_remaining_card_value (not yet solved)
# TODO: implement

# Step 13 - run_market_making_episode (not yet solved)
# TODO: implement

# Step 14 - summarize_episode_pnls (not yet solved)
# TODO: implement

