import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm # For progress bars

# --- 0. Configuration ---
class Config:
    # Data & Environment
    N_ASSETS = 20           # Number of stocks in the market
    N_FEATURES = 7          # PR, VOL, TV, MC, PE, BM, Div (as in paper)
    LOOK_BACK_K = 12        # Look-back window for LSTM-HA (e.g., 12 months)
    SIMULATION_DAYS = 252 * 2 # Total days to simulate for training data generation (reduced for faster demo)
    HOLDING_PERIOD_DAYS = 21 # Approx. 21 trading days in a month
    
    # Model Hyperparameters
    LSTM_HIDDEN_SIZE = 64
    ATTENTION_DIM_LSTM_HA = 32
    RANK_EMBEDDING_DIM = 16
    QUANTIZATION_COEFF = 5    # For CAAN rank prior (d_ij calculation)
    CAAN_DK_DIM = 64          # Dimension for Q, K in CAAN
    PORTFOLIO_G_SIZE_RATIO = 0.25 # Top/Bottom 25% for long/short portfolios

    # RL Training
    N_TRAJECTORIES_PER_UPDATE = 10 # Number of T-period sequences for one gradient update
    T_PERIODS_PER_TRAJECTORY = 12  # Number of holding periods per trajectory (e.g., 12 months)
    LEARNING_RATE = 1e-4
    EPOCHS = 5 # Reduced for faster demo
    RISK_FREE_RATE_PER_PERIOD = 0.001 # e.g., 0.1% per month
    H0_MARKET_SHARPE_BASELINE = 0.05  # Baseline Sharpe Ratio
    TRANSACTION_COST_PCT = 0.001 # 0.1% transaction cost (as in paper for CWT)

    # Interpretation
    INTERPRETATION_N_SAMPLES = 20 # Reduced for faster demo

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
cfg.PORTFOLIO_G_SIZE = int(cfg.N_ASSETS * cfg.PORTFOLIO_G_SIZE_RATIO)
if cfg.PORTFOLIO_G_SIZE == 0 and cfg.N_ASSETS > 0: # Ensure G is at least 1 if possible
    cfg.PORTFOLIO_G_SIZE = 1


# --- 1. Synthetic Data Generation & Market Environment ---
class SyntheticStockMarketEnv:
    def __init__(self, config):
        self.cfg = config
        self.n_assets = config.N_ASSETS
        self.n_features = config.N_FEATURES
        self.look_back_k = config.LOOK_BACK_K
        self.total_days = config.SIMULATION_DAYS
        self.holding_period_days = config.HOLDING_PERIOD_DAYS

        self.prices = np.zeros((self.total_days, self.n_assets))
        self.prices[0, :] = np.random.uniform(50, 150, self.n_assets)
        daily_returns_mu = np.random.normal(0.0005, 0.0002, self.n_assets)
        daily_returns_sigma = np.random.uniform(0.01, 0.03, self.n_assets)
        
        for t in range(1, self.total_days):
            returns = np.random.normal(daily_returns_mu, daily_returns_sigma)
            self.prices[t, :] = self.prices[t-1, :] * (1 + returns)
        self.prices = np.maximum(self.prices, 1e-2)

        self.raw_features = np.random.randn(self.total_days, self.n_assets, self.n_features)
        
        for t_day in range(self.holding_period_days, self.total_days):
            pr = self.prices[t_day, :] / self.prices[t_day - self.holding_period_days, :]
            self.raw_features[t_day, :, 0] = pr + np.random.normal(0, 0.1, self.n_assets)
            if t_day >= self.holding_period_days:
                vol = np.std(self.prices[t_day-self.holding_period_days:t_day, :], axis=0)
                self.raw_features[t_day, :, 1] = vol + np.random.normal(0, 0.05, self.n_assets)
        
        self.raw_features_torch = torch.tensor(self.raw_features, dtype=torch.float32) # Keep torch version
        self.prices_torch = torch.tensor(self.prices, dtype=torch.float32) # Keep torch version

        self.feature_standardizer = ZScoreStandardizer()
        fit_upto_day = int(self.total_days * 0.7) 
        fit_data = self.raw_features[:fit_upto_day].reshape(-1, self.n_features)
        self.feature_standardizer.fit(fit_data) # Pass numpy array
        
        self.current_period_idx = 0
        # Max periods available for starting a trajectory
        self.max_periods = (self.total_days - self.look_back_k * self.holding_period_days - self.cfg.T_PERIODS_PER_TRAJECTORY * self.holding_period_days) // self.holding_period_days
        if self.max_periods < 0: self.max_periods = 0 # Ensure non-negative
        
        self.current_day_in_sim = self.look_back_k * self.holding_period_days

    def _get_current_day_for_period_start(self, period_idx):
        return self.look_back_k * self.holding_period_days + period_idx * self.holding_period_days

    def reset(self, start_period_idx=None):
        if start_period_idx is None:
            if self.max_periods > 0:
                 self.current_period_idx = np.random.randint(0, self.max_periods + 1)
            else: # Not enough data for even one full trajectory
                 self.current_period_idx = 0
        else:
            self.current_period_idx = start_period_idx
        
        self.current_day_in_sim = self._get_current_day_for_period_start(self.current_period_idx)
        return self._get_state()

    def _get_state(self):
        hist_features = torch.zeros((self.n_assets, self.look_back_k, self.n_features), device=cfg.DEVICE)
        for k_idx in range(self.look_back_k):
            day_for_feature_k = self.current_day_in_sim - (self.look_back_k - k_idx) * self.holding_period_days
            if day_for_feature_k >= 0:
                raw_feats_k_np = self.raw_features[day_for_feature_k, :, :]
                standardized_feats_k_np = self.feature_standardizer.transform(raw_feats_k_np)
                hist_features[:, k_idx, :] = torch.tensor(standardized_feats_k_np, device=cfg.DEVICE)
            # else: leave as zeros (e.g., at the very beginning of simulation)

        day_t = self.current_day_in_sim
        day_t_minus_1_period = day_t - self.holding_period_days
        
        if day_t_minus_1_period < 0:
            pr_ranks = torch.randperm(self.n_assets, device=cfg.DEVICE).long()
        else:
            prices_t = self.prices_torch[day_t, :]
            prices_t_minus_1_period = self.prices_torch[day_t_minus_1_period, :]
            price_rising_rates = prices_t / prices_t_minus_1_period
            
            # argsort gives indices that would sort the array.
            # To get ranks (0 for highest), we need to map these.
            sorted_indices_desc = torch.argsort(price_rising_rates, descending=True)
            temp_ranks = torch.empty_like(sorted_indices_desc)
            temp_ranks[sorted_indices_desc] = torch.arange(self.n_assets, device=price_rising_rates.device)
            pr_ranks = temp_ranks.to(cfg.DEVICE).long()

        return hist_features, pr_ranks

    def step(self, portfolio_b_c_tuple):
        b_long_alloc, b_short_alloc = portfolio_b_c_tuple
        
        day_t = self.current_day_in_sim
        day_t_plus_1_period = day_t + self.holding_period_days

        if day_t_plus_1_period >= self.total_days:
            return self._get_state(), torch.tensor(0.0, device=cfg.DEVICE), True 

        prices_t = self.prices_torch[day_t, :].to(cfg.DEVICE)
        prices_t_plus_1_period = self.prices_torch[day_t_plus_1_period, :].to(cfg.DEVICE)
        price_rising_rates_z_t = prices_t_plus_1_period / prices_t

        long_return_contrib = torch.sum(b_long_alloc * price_rising_rates_z_t)
        short_return_contrib = torch.sum(b_short_alloc * price_rising_rates_z_t)
        R_t = long_return_contrib - short_return_contrib
        
        transaction_cost_val = 2 * self.cfg.TRANSACTION_COST_PCT
        R_t_net = R_t - transaction_cost_val

        self.current_period_idx += 1
        self.current_day_in_sim = self._get_current_day_for_period_start(self.current_period_idx)
        
        # Check if the *next* step would be out of bounds for features or prices
        next_required_day_for_state = self.current_day_in_sim # for features of next state
        next_required_day_for_return = self.current_day_in_sim + self.holding_period_days # for prices of next R_t

        done = (self.current_period_idx >= self.max_periods + self.cfg.T_PERIODS_PER_TRAJECTORY) or \
               (next_required_day_for_return >= self.total_days) or \
               (self.current_day_in_sim < self.look_back_k * self.holding_period_days) # Should not happen if reset correctly

        return self._get_state(), R_t_net, done

# --- 2. Feature Standardization ---
class ZScoreStandardizer:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, data_np): # Expects numpy array
        self.mean = np.mean(data_np, axis=0, keepdims=True)
        self.std = np.std(data_np, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data_np): # Expects numpy array
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fit yet.")
        return (data_np - self.mean) / self.std

    def fit_transform(self, data_np):
        self.fit(data_np)
        return self.transform(data_np)

# --- 3. LSTM-HA Module ---
class LSTMHA(nn.Module):
    def __init__(self, n_features, lstm_hidden_size, attention_dim_lstm_ha):
        super().__init__()
        self.lstm = nn.LSTM(n_features, lstm_hidden_size, batch_first=True)
        self.W1_att = nn.Linear(lstm_hidden_size, attention_dim_lstm_ha)
        self.W2_att = nn.Linear(lstm_hidden_size, attention_dim_lstm_ha)
        self.w_att = nn.Linear(attention_dim_lstm_ha, 1)

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        h_K = h_n[-1]
        alpha_k_scores = self.w_att(torch.tanh(self.W1_att(lstm_out) + self.W2_att(h_K.unsqueeze(1))))
        attention_weights = F.softmax(alpha_k_scores, dim=1)
        r = torch.sum(attention_weights * lstm_out, dim=1)
        return r

# --- 4. CAAN Module ---
class CAAN(nn.Module):
    def __init__(self, d_model, n_assets_config, rank_embedding_dim, quantization_coeff, dk_dim):
        super().__init__()
        self.d_model = d_model
        self.dk_dim = dk_dim

        self.W_Q = nn.Linear(d_model, dk_dim)
        self.W_K = nn.Linear(d_model, dk_dim)
        self.W_V = nn.Linear(d_model, d_model)

        self.quantization_coeff = quantization_coeff
        num_rank_embeddings = int(np.floor((n_assets_config - 1) / quantization_coeff)) + 1
        if num_rank_embeddings <=0: num_rank_embeddings =1 # Ensure at least 1 embedding
        self.rank_embedding = nn.Embedding(num_rank_embeddings, rank_embedding_dim)
        self.w_L = nn.Linear(rank_embedding_dim, 1)

        self.fc_winner_score = nn.Linear(d_model, 1)

    def forward(self, stock_reprs, price_rising_ranks):
        q_i = self.W_Q(stock_reprs)
        k_j = self.W_K(stock_reprs)
        v_j = self.W_V(stock_reprs)

        rank_diff = price_rising_ranks.unsqueeze(1) - price_rising_ranks.unsqueeze(0)
        d_ij = torch.floor(torch.abs(rank_diff) / self.quantization_coeff).long()
        d_ij = torch.clamp(d_ij, 0, self.rank_embedding.num_embeddings - 1)

        embedded_rank_diffs = self.rank_embedding(d_ij)
        psi_ij = torch.sigmoid(self.w_L(embedded_rank_diffs)).squeeze(-1)

        score_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) / np.sqrt(float(self.dk_dim)) # Ensure dk_dim is float for sqrt
        beta_ij = psi_ij * score_ij

        satt_weights = F.softmax(beta_ij, dim=-1)
        a_i = torch.matmul(satt_weights, v_j)
        winner_scores = torch.sigmoid(self.fc_winner_score(a_i))
        
        return winner_scores.squeeze(-1)

# --- 5. AlphaStock Network (Combined Model) ---
class AlphaStockNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.lstm_ha = LSTMHA(config.N_FEATURES, config.LSTM_HIDDEN_SIZE, config.ATTENTION_DIM_LSTM_HA)
        self.caan = CAAN(config.LSTM_HIDDEN_SIZE, config.N_ASSETS, config.RANK_EMBEDDING_DIM,
                         config.QUANTIZATION_COEFF, config.CAAN_DK_DIM)

    def forward(self, all_stock_histories_at_t, all_stock_pr_ranks_at_t_minus_1):
        stock_reprs = self.lstm_ha(all_stock_histories_at_t)
        winner_scores = self.caan(stock_reprs, all_stock_pr_ranks_at_t_minus_1)
        return winner_scores

# --- 6. Portfolio Generator (Modified for RL and clarity) ---
def generate_portfolio_allocations(winner_scores, G, current_n_assets):
    sorted_indices = torch.argsort(winner_scores, descending=True)
    
    effective_G = min(G, current_n_assets // 2 if current_n_assets > 1 else 0)
    if current_n_assets == 1 and G > 0 : effective_G = 0 # Cannot long and short same single asset
    # If effective_G is 0 but we want to trade, it implies an issue or N_ASSETS is too small.
    # For BWSL, need at least 2 assets to pick one winner and one loser if G=1.
    
    b_long_alloc = torch.zeros_like(winner_scores)
    b_short_alloc = torch.zeros_like(winner_scores)
    b_c_for_log_prob = torch.zeros_like(winner_scores)

    if effective_G > 0:
        winner_indices = sorted_indices[:effective_G]
        # Ensure loser_indices don't overlap with winner_indices if current_n_assets is small
        loser_indices = sorted_indices[max(effective_G, current_n_assets - effective_G):] 
        # If current_n_assets - effective_G < effective_G, this means overlap or not enough unique losers.
        # A stricter BWSL would ensure winners and losers are distinct sets.
        # For simplicity, this selection is okay, but for very small N_ASSETS and large G, it might pick same stocks.
        # Let's ensure they are distinct if current_n_assets is small.
        if current_n_assets < 2 * effective_G : # Not enough distinct stocks for G winners and G losers
             # This case needs careful handling or implies G is too large for N_ASSETS
             # For now, we proceed, but this might lead to issues if not enough distinct stocks.
             # A better way:
             if current_n_assets >= 2: # Need at least two assets for a BWSL
                winner_indices = sorted_indices[:effective_G]
                # Losers are from the other end, ensuring no overlap if possible
                potential_loser_indices = sorted_indices[effective_G:]
                if len(potential_loser_indices) >= effective_G:
                    loser_indices = potential_loser_indices[-effective_G:]
                else: # Not enough distinct losers, take all available ones
                    loser_indices = potential_loser_indices
             else: # Not enough assets for BWSL with G > 0
                winner_indices = torch.tensor([], dtype=torch.long, device=winner_scores.device)
                loser_indices = torch.tensor([], dtype=torch.long, device=winner_scores.device)


        if len(winner_indices) > 0:
            s_winners = winner_scores[winner_indices]
            exp_s_winners = torch.exp(s_winners)
            proportions_plus = exp_s_winners / torch.sum(exp_s_winners)
            b_long_alloc[winner_indices] = proportions_plus
            b_c_for_log_prob[winner_indices] = proportions_plus
            
        if len(loser_indices) > 0:
            s_losers_scores = winner_scores[loser_indices]
            exp_one_minus_s_losers = torch.exp(1.0 - s_losers_scores) # Ensure 1.0 is float
            proportions_minus = exp_one_minus_s_losers / torch.sum(exp_one_minus_s_losers)
            b_short_alloc[loser_indices] = proportions_minus
            b_c_for_log_prob[loser_indices] = proportions_minus
            
    return b_long_alloc, b_short_alloc, b_c_for_log_prob

# --- 7. Sharpe Ratio Calculation ---
def calculate_sharpe_ratio(period_returns_list, risk_free_rate_per_period):
    if not period_returns_list: return torch.tensor(0.0, device=cfg.DEVICE)
    period_returns = torch.tensor(period_returns_list, dtype=torch.float32, device=cfg.DEVICE)
    
    avg_return = torch.mean(period_returns)
    if len(period_returns) < 2: volatility = torch.tensor(0.0, device=cfg.DEVICE)
    else: volatility = torch.std(period_returns, unbiased=False)

    if volatility.abs() < 1e-9: # Effectively zero volatility
        if (avg_return - risk_free_rate_per_period).abs() < 1e-9: # Effectively zero excess return
            return torch.tensor(0.0, device=cfg.DEVICE)
        # If excess return is positive with no risk, Sharpe is positive infinity
        # If excess return is negative with no risk, Sharpe is negative infinity
        # Paper doesn't specify. Let's cap it or return a large number.
        # For stability in RL, often capped or a fixed large value.
        # Here, returning 0 if no vol may be too punitive if there was profit.
        # Let's return raw ratio and handle inf outside if needed.
        return (avg_return - risk_free_rate_per_period) / (volatility + 1e-9) # Add epsilon for stability
    
    sharpe_ratio = (avg_return - risk_free_rate_per_period) / volatility
    return sharpe_ratio

# --- 8. RL Training Loop ---
def train_alphastock(config):
    env = SyntheticStockMarketEnv(config)
    model = AlphaStockNetwork(config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"Training on {config.DEVICE} with {config.N_ASSETS} assets.")
    print(f"Portfolio G size: {config.PORTFOLIO_G_SIZE}")

    for epoch in range(config.EPOCHS):
        batch_trajectory_rewards_H_pi_n = []
        batch_sum_log_G_terms = [] 

        model.train() # Ensure model is in training mode for RL loop
        
        pbar_traj = tqdm(range(config.N_TRAJECTORIES_PER_UPDATE), desc=f"Epoch {epoch+1}/{config.EPOCHS} Trajectories", leave=False)
        for traj_idx in pbar_traj:
            current_traj_period_returns_R_t = []
            # Ensure sum_log_G_for_traj is a tensor that requires grad from the start
            # by involving a model parameter or making it a sum of tensors that do.
            # Initializing with 0.0 and adding to it works if the added terms require grad.
            sum_log_G_for_traj = torch.tensor(0.0, device=config.DEVICE, requires_grad=False) # Will be re-assigned

            state = env.reset() 

            for t_step in range(config.T_PERIODS_PER_TRAJECTORY):
                hist_features, pr_ranks = state
                hist_features = hist_features.to(config.DEVICE)
                pr_ranks = pr_ranks.to(config.DEVICE)

                winner_scores = model(hist_features, pr_ranks)
                
                current_n_assets_in_state = winner_scores.shape[0]
                b_long, b_short, b_c_for_rl = generate_portfolio_allocations(
                    winner_scores, config.PORTFOLIO_G_SIZE, current_n_assets_in_state
                )
                
                selected_mask = b_c_for_rl > 1e-9 
                current_step_log_G_sum = torch.tensor(0.0, device=config.DEVICE)
                if torch.any(selected_mask):
                    # log G^(i) where G^(i) = b_c(i)_t. This b_c(i)_t is output of model.
                    log_b_c_selected = torch.log(b_c_for_rl[selected_mask])
                    current_step_log_G_sum = torch.sum(log_b_c_selected)
                
                if t_step == 0: # Initialize sum_log_G_for_traj with the first term
                    sum_log_G_for_traj = current_step_log_G_sum
                else:
                    sum_log_G_for_traj = sum_log_G_for_traj + current_step_log_G_sum


                next_state, R_t_net, done = env.step((b_long.detach(), b_short.detach()))
                current_traj_period_returns_R_t.append(R_t_net.item())
                
                state = next_state
                if done:
                    break
            
            H_pi_n = calculate_sharpe_ratio(current_traj_period_returns_R_t, config.RISK_FREE_RATE_PER_PERIOD)
            if torch.isinf(H_pi_n) or torch.isnan(H_pi_n): 
                # Replace inf/nan with a very small or zero reward to avoid issues
                # A large positive Sharpe (inf) might be good, but can destabilize gradients.
                # A large negative Sharpe (-inf) is also problematic.
                # Capping or using a surrogate is common. For now, set to 0.
                H_pi_n = torch.tensor(0.0, device=config.DEVICE) 
            
            batch_trajectory_rewards_H_pi_n.append(H_pi_n)
            batch_sum_log_G_terms.append(sum_log_G_for_traj) # This tensor should track gradients
            pbar_traj.set_postfix({"Avg Sharpe (curr_traj)": H_pi_n.item()})

        if not batch_sum_log_G_terms:
            print(f"Epoch {epoch+1}: No trajectories processed, skipping update.")
            continue

        rewards_tensor = torch.stack(batch_trajectory_rewards_H_pi_n)
        # Filter out terms that might not have grads if a trajectory was too short or had no actions
        valid_log_G_terms = [term for term in batch_sum_log_G_terms if term.requires_grad]
        if not valid_log_G_terms:
            print(f"Epoch {epoch+1}: No valid log_G terms with gradients, skipping update.")
            continue
        
        # We need to align rewards with log_G_terms that have grads
        # This is tricky if some trajectories don't produce grad-requiring log_G_terms
        # For simplicity, assume all do, or filter rewards accordingly.
        # A simpler way: if a term doesn't require grad, its product with advantage won't contribute to loss.grad.
        sum_log_G_tensor = torch.stack(batch_sum_log_G_terms)


        advantage = rewards_tensor - config.H0_MARKET_SHARPE_BASELINE
        
        # Ensure advantage doesn't require grad if it's used as a detached coefficient
        loss = -torch.mean(advantage.detach() * sum_log_G_tensor) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_sharpe_epoch = torch.mean(rewards_tensor).item()
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Avg Sharpe: {avg_sharpe_epoch:.4f}, Loss: {loss.item():.4f}")

    return model, env.feature_standardizer

# --- 9. Model Interpretation (Sensitivity Analysis) ---
def interpret_model(model, env, config):
    print("\n--- Starting Model Interpretation (Sensitivity Analysis) ---")
    original_mode_is_training = model.training
    model.train() # Switch to train mode for cuDNN RNN backward

    avg_sensitivities = torch.zeros((config.LOOK_BACK_K, config.N_FEATURES), device=config.DEVICE)
    num_samples_collected = 0

    pbar_interp = tqdm(range(config.INTERPRETATION_N_SAMPLES), desc="Interpretation Samples", leave=False)
    for sample_idx in pbar_interp:
        # Ensure env.max_periods is sensible for interpretation sampling
        # Interpretation might need states from anywhere, not just valid trajectory start states
        # Let's use a simpler random day selection for interpretation state
        # Ensure enough history for look_back_K
        min_day_for_state = config.LOOK_BACK_K * config.HOLDING_PERIOD_DAYS
        if env.total_days <= min_day_for_state:
            print("Not enough simulation days for interpretation.")
            if not original_mode_is_training: model.eval()
            return None
            
        random_current_day = np.random.randint(min_day_for_state, env.total_days)
        
        # Manually set env's current day to get a state
        env.current_day_in_sim = random_current_day 
        hist_features_all_assets, pr_ranks_all_assets = env._get_state() # Use internal method to get state for this day
        
        hist_features_all_assets = hist_features_all_assets.to(config.DEVICE).detach().clone()
        hist_features_all_assets.requires_grad_(True)
        pr_ranks_all_assets = pr_ranks_all_assets.to(config.DEVICE)

        winner_scores = model(hist_features_all_assets, pr_ranks_all_assets)

        for asset_idx in range(config.N_ASSETS):
            model.zero_grad() # Zero model param grads
            if hist_features_all_assets.grad is not None:
                hist_features_all_assets.grad.zero_() # Zero input grads

            # Backward on the specific winner score s_i
            winner_scores[asset_idx].backward(retain_graph=True) 
            
            if hist_features_all_assets.grad is not None:
                sensitivity_for_asset_i = hist_features_all_assets.grad[asset_idx, :, :]
                avg_sensitivities += sensitivity_for_asset_i.detach()
                num_samples_collected += 1
            # else: grad might be None if the output doesn't depend on this input part

        pbar_interp.set_postfix({"Collected": num_samples_collected})

    if num_samples_collected > 0:
        avg_sensitivities /= num_samples_collected
    else:
        print("No samples collected for interpretation, or gradients were always None.")
        if not original_mode_is_training: model.eval()
        return None

    print("Average Sensitivities (Influence of feature x_q to winner score):")
    feature_names = [f"Feat_{i}" for i in range(config.N_FEATURES)]
    if config.N_FEATURES > 0: feature_names[0] = "PR (PriceRise)"
    if config.N_FEATURES > 1: feature_names[1] = "VOL (Volatility)"
    if config.N_FEATURES > 2: feature_names[2] = "TV (TradeVol)"
    if config.N_FEATURES > 3: feature_names[3] = "MC (MarketCap)"
    if config.N_FEATURES > 4: feature_names[4] = "PE (PriceEarn)"
    if config.N_FEATURES > 5: feature_names[5] = "BM (BookMarket)"
    if config.N_FEATURES > 6: feature_names[6] = "Div (Dividend)"
    
    for feat_idx in range(config.N_FEATURES):
        print(f"\nFeature: {feature_names[feat_idx]}")
        for k_idx in range(config.LOOK_BACK_K):
            print(f"  Month -{config.LOOK_BACK_K - k_idx}: {avg_sensitivities[k_idx, feat_idx].item():.4f}")
    
    if not original_mode_is_training: model.eval() # Restore original mode
    return avg_sensitivities.cpu().numpy()

# --- Main Execution ---
if __name__ == '__main__':
    print(f"Using device: {cfg.DEVICE}")
    trained_model, standardizer_from_env = train_alphastock(cfg)
    
    if trained_model and standardizer_from_env:
        print("\n--- Proceeding to Interpretation ---")
        # For interpretation, we need an environment instance.
        # It's better to create a new one or ensure the existing one is reset appropriately.
        # The key is that it uses the *same* fitted standardizer.
        interp_env = SyntheticStockMarketEnv(cfg) 
        interp_env.feature_standardizer = standardizer_from_env # CRITICAL: Use the already fitted standardizer
        
        sensitivities = interpret_model(trained_model, interp_env, cfg)
        if sensitivities is not None:
            print("\nInterpretation Complete. Sensitivities calculated.")
            # Here you could plot sensitivities similar to Figure 3 in the paper.
    else:
        print("Training did not complete successfully or returned None. Skipping interpretation.")