import math

def calculate_tca_metrics(side, size_adv, vol, alpha_bp, trade_time, mi_params):
    """
    Calculates Market Impact (MI), Timing Risk (TR), and Price Appreciation (PA).
    
    Parameters:
    - side: 'Buy' or 'Sell'
    - size_adv: Order size as % of ADV (e.g., 0.15 for 15%)
    - vol: Annual volatility (e.g., 0.30 for 30%)
    - alpha_bp: Expected price drift in basis points
    - trade_time: Duration of trade in days (e.g., 0.75)
    - mi_params: Dictionary containing a1, a2, a3, a4, b1
    """
    
    # 1. Calculate POV (Participation Rate)
    # T = (S/ADV) / POV => POV = (S/ADV) / T
    pov = size_adv / trade_time
    
    # 2. Market Impact (MI) using I-Star Model
    a1, a2, a3, a4, b1 = mi_params['a1'], mi_params['a2'], mi_params['a3'], mi_params['a4'], mi_params['b1']
    
    i_star = a1 * (size_adv**a2) * (vol**a3)
    # mi_bp = i_star * (b1 * (pov**a4) + (1 - b1))
    mi_bp = (b1 * (pov**a4) * i_star)+ (1 - b1) * i_star  # MI in basis points
    
    # 3. Timing Risk (TR)
    # Formula: TR = vol * sqrt(1/250 * 1/3 * (S/ADV) * (1-POV)/POV) * 10,000
    tr_bp = vol * math.sqrt((1/250) * (1/3) * size_adv * (1 - pov) / pov) * 10000
    
    # 4. Price Appreciation (PA)
    # side_val: Buy = 1, Sell = -1
    side_val = 1 if side.lower() == 'buy' else -1
    # For a Sell order (side=-1) and positive Alpha, PA is a gain (negative cost)
    pa_bp = side_val * (0.5 * alpha_bp * trade_time)
    
    return {
        "POV": pov,
        "Market Impact (bp)": mi_bp,
        "Timing Risk (bp)": tr_bp,
        "Price Appreciation (bp)": pa_bp,
        "Expected Cost (bp)": mi_bp + pa_bp
    }

# Input Data
inputs = {
    "side": "Sell",
    "size_adv": 0.2,
    "vol": 0.3,
    "alpha_bp": 30,
    "trade_time": 0.5, # T = (S/ADV) / POV => T = 0.05 / 0.15
    "mi_params": {
        "a1": 955, 
        "a2": 0.3, 
        "a3": 0.7, 
        "a4": 0.8, 
        "b1": 0.9
    }
}

# Execution
results = calculate_tca_metrics(**inputs)

# Output
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")