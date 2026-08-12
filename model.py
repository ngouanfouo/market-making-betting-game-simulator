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

# Step 3 - pay_per_reroll_die_game
def pay_per_reroll_die_game(sides, reroll_cost):
    N = sides
    c = reroll_cost
    
    best_threshold = 1
    best_value = -float('inf')
    
    # Sweep through all possible thresholds
    # t = minimum face value we keep (1..N)
    # t = N+1 would mean never keep, which gives infinite rerolls and diverges
    for t in range(1, N + 1):
        # Expected value of a roll given we keep it (average of t..N)
        expected_keep = (t + N) / 2.0
        
        # Probability of keeping: faces t, t+1, ..., N
        p_keep = (N - t + 1) / N
        
        # Probability of rerolling: faces 1, 2, ..., t-1
        p_reroll = (t - 1) / N
        
        # Solve the recursion:
        # V = p_keep * E[keep] + p_reroll * (V - c)
        # V = p_keep * E[keep] + p_reroll * V - p_reroll * c
        # V - p_reroll * V = p_keep * E[keep] - p_reroll * c
        # V * (1 - p_reroll) = p_keep * E[keep] - p_reroll * c
        # V * p_keep = p_keep * E[keep] - p_reroll * c
        # V = E[keep] - (p_reroll / p_keep) * c
        # Since p_keep > 0 for t <= N, we can divide safely
        
        # Special case: if p_keep = 0 (t = N+1), we never keep, but we don't consider this
        V = expected_keep - (p_reroll / p_keep) * c
        
        # Track best value with tie-breaking (smallest threshold wins)
        if V > best_value:
            best_value = V
            best_threshold = t
    
    return {
        'threshold': best_threshold,
        'value': float(best_value)
    }

# Step 4 - red_black_card_game_value
from functools import lru_cache

def red_black_card_game_value(num_red, num_black):
    @lru_cache(maxsize=None)
    def V(r, b):
        # Base cases
        if r == 0 and b == 0:
            return 0.0
        if r == 0:
            # Only black cards left, every draw loses money, so stop
            return 0.0
        if b == 0:
            # Only red cards left, draw all of them for guaranteed profit
            return float(r)
        
        # Probability of drawing red
        p_red = r / (r + b)
        # Probability of drawing black
        p_black = b / (r + b)
        
        # Expected value if we draw:
        # Draw red: gain +1, then continue with (r-1, b)
        # Draw black: gain -1, then continue with (r, b-1)
        draw_value = p_red * (1 + V(r - 1, b)) + p_black * (-1 + V(r, b - 1))
        
        # We can stop anytime (gain 0 additional)
        return max(0.0, draw_value)
    
    # Compute continuation value for the initial state
    cont = V(num_red, num_black)
    
    # If continuation <= 0, stopping is optimal
    stop_now = cont <= 0.0
    
    # Value is max(0, continuation)
    value = max(0.0, cont)
    
    return {
        'value': float(value),
        'stop_now': stop_now
    }

# Step 5 - make_quotes
def make_quotes(fair_value, spread_width):
    # Compute half of the total spread
    half_spread = spread_width / 2.0
    
    # Compute bid and ask symmetrically around fair value
    bid = fair_value - half_spread
    ask = fair_value + half_spread
    
    return {
        'bid': float(bid),
        'ask': float(ask)
    }

# Step 6 - execute_trade
def execute_trade(state, side, bid, ask, size=1):
    # Read current state without mutating
    cash = state['cash']
    inventory = state['inventory']
    
    # Apply trade from market maker's perspective
    if side == 'buy':
        # Counterparty buys from YOU at ask -> YOU sell
        cash += size * ask
        inventory -= size
    elif side == 'sell':
        # Counterparty sells to YOU at bid -> YOU buy
        cash -= size * bid
        inventory += size
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
    
    return {
        'cash': float(cash),
        'inventory': float(inventory)
    }

# Step 7 - mark_to_market_pnl
def mark_to_market_pnl(cash, inventory, settlement_value):
    # TODO: return total P&L given cash, remaining inventory, and settlement value.
    return cash+inventory*settlement_value

# Step 8 - adverse_selection_loss
import numpy as np

def adverse_selection_loss(fair_value, bid, ask, informed_values, informed_probabilities):
    # Convert inputs to numpy arrays
    values = np.asarray(informed_values, dtype=float)
    probs = np.asarray(informed_probabilities, dtype=float)
    
    # Loss when informed trader buys from us (v > ask): they gain v - ask, we lose that amount
    ask_loss = np.maximum(values - ask, 0.0)
    
    # Loss when informed trader sells to us (v < bid): they gain bid - v, we lose that amount
    bid_loss = np.maximum(bid - values, 0.0)
    
    # Expected total loss = weighted sum of both losses
    expected_loss = np.sum(probs * (ask_loss + bid_loss))
    
    return float(expected_loss)

# Step 9 - uncertainty_spread
def uncertainty_spread(base_spread, uncertainty):
    """Return a spread width >= base_spread that grows with uncertainty."""
    # Linear scaling: spread = base_spread + uncertainty
    # This ensures:
    # - When uncertainty = 0, spread = base_spread (minimum)
    # - When uncertainty > 0, spread > base_spread
    # - Strictly increasing with uncertainty
    spread = base_spread + uncertainty
    
    return float(spread)

# Step 10 - inventory_skewed_quotes
def inventory_skewed_quotes(fair_value, spread_width, inventory, skew_strength):
    # Compute half spread
    half_spread = spread_width / 2.0
    
    # Compute inventory shift: positive inventory lowers prices (to encourage selling)
    # Negative inventory raises prices (to encourage buying)
    shift = skew_strength * inventory
    
    # Shift the midpoint against the inventory position
    # For positive inventory (long), we want lower prices -> subtract shift
    # For negative inventory (short), we want higher prices -> subtract shift (negative shift raises prices)
    mid = fair_value - shift
    
    # Compute bid and ask around the shifted midpoint
    bid = mid - half_spread
    ask = mid + half_spread
    
    return {
        'bid': float(bid),
        'ask': float(ask)
    }

# Step 11 - update_fair_value_from_trade
def update_fair_value_from_trade(fair_value, side, bid, ask, adjustment):
    # If adjustment is zero, no learning occurs
    if adjustment == 0.0:
        return fair_value
    
    # Compute half spread as a measure of quote width
    half_spread = (ask - bid) / 2.0
    
    # Determine the update step
    step = adjustment * half_spread
    
    # Apply the update based on trade side
    # Counterparty 'buy' (they bought from us) suggests true value is higher -> increase fair value
    # Counterparty 'sell' (they sold to us) suggests true value is lower -> decrease fair value
    if side == 'buy':
        new_fair_value = fair_value + step
    else:  # side == 'sell'
        new_fair_value = fair_value - step
    
    return float(new_fair_value)

# Step 12 - update_remaining_card_value (not yet solved)
# TODO: implement

# Step 13 - run_market_making_episode (not yet solved)
# TODO: implement

# Step 14 - summarize_episode_pnls (not yet solved)
# TODO: implement

