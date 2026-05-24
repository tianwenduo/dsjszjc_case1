"""
Inventory Optimization with Newsvendor Model
=============================================
Goal: Predict target inventory for each (item_id, store_code) for
the 2-week period 2015-12-14 to 2015-12-27 that minimizes total cost.

Cost structure (from configData.csv):
  a = over-stocking cost per unit (补多成本)
  b = under-stocking cost per unit (补少成本)

Newsvendor model:
  Optimal target Q* satisfies: F(Q*) = b / (a + b)
  where F is the CDF of 2-week demand.
  For normal demand: Q* = μ + z·σ, where z = Φ⁻¹(b/(a+b))

Approach:
1. Aggregate daily demand → weekly demand
2. Build rolling 2-week blocks for distribution estimation
3. Forecast μ and σ using recent trend + seasonal reference
4. Apply newsvendor quantile to get optimal inventory
5. Fallback for sparse items using hierarchical pooling
"""
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("Loading data...")
config = pd.read_csv('D:/case1/case1/configData.csv', dtype={'store_code': str})
train = pd.read_csv('D:/case1/case1/traindata.csv', dtype={'store_code': str})
train['date'] = pd.to_datetime(train['date'])
# 删除 11月11日 和 12月12日 的数据（所有年份）
train = train[~((train['date'].dt.month == 11) & (train['date'].dt.day == 11) |
                ((train['date'].dt.month == 12) & (train['date'].dt.day == 12)))]

# Parse a, b costs
def parse_ab(val):
    parts = str(val).split('_')
    return float(parts[0]), float(parts[1])

config['a'], config['b'] = zip(*config['a_b'].apply(parse_ab))
config['critical_ratio'] = config['b'] / (config['a'] + config['b'])

print(f"Config: {config.shape[0]} item-store combos")
print(f"Train:  {train.shape[0]} records, {train['item_id'].nunique()} items, "
      f"dates {train['date'].min().date()} to {train['date'].max().date()}")

# ============================================================
# 2. TIME FEATURES
# ============================================================
# ISO week: Monday=1, Sunday=7; weeks start on Monday
train['iso_year'] = train['date'].dt.isocalendar().year.astype(int)
train['iso_week'] = train['date'].dt.isocalendar().week.astype(int)
# Create a composite week index for easy sorting
train['year_week'] = train['iso_year'].astype(str) + '-' + train['iso_week'].astype(str).str.zfill(2)

# ============================================================
# 3. WEEKLY DEMAND AGGREGATION
# ============================================================
print("Aggregating weekly demand...")
weekly = train.groupby(['item_id', 'store_code', 'iso_year', 'iso_week'])['qty_alipay'].sum().reset_index()
weekly.rename(columns={'qty_alipay': 'weekly_demand'}, inplace=True)

# Pivot to have a complete week grid
# Build full week range
all_weeks = sorted(weekly['iso_year'].astype(str) + '-' + weekly['iso_week'].astype(str).str.zfill(2))
min_year, min_week = weekly['iso_year'].min(), weekly['iso_week'].min()
max_year, max_week = weekly['iso_year'].max(), weekly['iso_week'].max()

# Generate all ISO weeks in range
def generate_iso_weeks(start_year, start_week, end_year, end_week):
    """Generate all ISO weeks between start and end"""
    from datetime import date, timedelta
    # Find the Monday of start week
    jan4_start = date(start_year, 1, 4)
    start_monday = jan4_start + timedelta(days=-(jan4_start.isoweekday() - 1))
    start_date = start_monday + timedelta(weeks=start_week - 1)

    jan4_end = date(end_year, 1, 4)
    end_monday = jan4_end + timedelta(days=-(jan4_end.isoweekday() - 1))
    end_date = end_monday + timedelta(weeks=end_week - 1)

    weeks = []
    current = start_date
    while current <= end_date:
        iso_year, iso_week, _ = current.isocalendar()
        weeks.append((iso_year, iso_week))
        current += timedelta(days=7)
    return weeks

# ============================================================
# 4. BUILD 2-WEEK ROLLING BLOCKS FOR EACH ITEM-STORE
# ============================================================
print("Building 2-week demand blocks...")

def build_two_week_blocks(item_data):
    """
    For a given item-store's weekly demand series, build 2-week rolling sums.
    Returns list of (year_week_label, two_week_demand).
    year_week_label is the ENDING week of the 2-week block.
    """
    item_data = item_data.sort_values(['iso_year', 'iso_week'])
    demands = item_data[['iso_year', 'iso_week', 'weekly_demand']].copy()
    demands['year_week_str'] = demands['iso_year'].astype(str) + '-W' + demands['iso_week'].astype(str).str.zfill(2)

    results = []
    for i in range(1, len(demands)):
        w1 = demands.iloc[i-1]
        w2 = demands.iloc[i]
        # Check weeks are consecutive
        if (w2['iso_year'] == w1['iso_year'] and w2['iso_week'] == w1['iso_week'] + 1) or \
           (w2['iso_year'] == w1['iso_year'] + 1 and w1['iso_week'] in [52, 53] and w2['iso_week'] == 1):
            two_week_demand = w1['weekly_demand'] + w2['weekly_demand']
            results.append({
                'end_year': w2['iso_year'],
                'end_week': w2['iso_week'],
                'two_week_demand': two_week_demand
            })
    return results

# Build blocks for each item-store
item_store_blocks = {}
for (item_id, store_code), group in weekly.groupby(['item_id', 'store_code']):
    blocks = build_two_week_blocks(group)
    if blocks:
        item_store_blocks[(item_id, store_code)] = blocks

print(f"Built 2-week blocks for {len(item_store_blocks)} item-store combos")

# ============================================================
# 5. FORECASTING FUNCTION
# ============================================================
# Target period: 2015-12-14 (Monday) to 2015-12-27 (Sunday)
# ISO weeks: 2015-W51 (Dec 14-20) and 2015-W52 (Dec 21-27)
TARGET_END_WEEK = (2015, 52)  # The ending week of the 2-week target block

def forecast_item_store(blocks, target_end_week, same_period_blocks=None):
    """
    Forecast mean and std of 2-week demand ending at target_end_week.

    Uses:
    - Recent 2-week blocks (last 6, ≈ 3 months) for trend
    - Same-period-last-year blocks for seasonality (if available)
    - Exponential weighting on recent blocks

    Returns (mu, sigma) or (None, None) if can't forecast.
    """
    if not blocks:
        return None, None

    target_year, target_week = target_end_week

    # Separate blocks into:
    # 1. Recent blocks (any blocks from the last 12 weeks before target)
    # 2. Seasonal blocks (same weeks from previous years)

    recent_demands = []
    seasonal_demands = []

    for b in blocks:
        ey, ew = b['end_year'], b['end_week']
        d = b['two_week_demand']

        # Check if this is a same-week seasonal match (same ISO week, earlier year)
        if ew == target_week and ey < target_year:
            seasonal_demands.append(d)

        # Recent blocks: within the last 12 weeks and before target
        # Calculate week distance
        week_dist = (target_year - ey) * 52 + (target_week - ew)
        if 0 < week_dist <= 12:
            recent_demands.append((week_dist, d))

    # ---- Mean forecast ----
    mu = None

    if recent_demands:
        # Exponentially weighted: weight = exp(-week_dist / decay)
        # decay = 4 means half-life of ~2.8 weeks
        decay = 4.0
        weights = [np.exp(-wd / decay) for wd, _ in recent_demands]
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_avg = sum(w * d for (_, d), w in zip(recent_demands, weights)) / total_weight
            mu = weighted_avg
        else:
            mu = np.mean([d for _, d in recent_demands])
    elif seasonal_demands:
        mu = np.mean(seasonal_demands)
    else:
        # Fallback: use all historical blocks
        all_demands = [b['two_week_demand'] for b in blocks]
        if all_demands:
            # Use last few blocks
            mu = np.mean(all_demands[-6:]) if len(all_demands) >= 6 else np.mean(all_demands)
        else:
            return None, None

    # Blend with seasonal if available
    if seasonal_demands and recent_demands:
        mu_seasonal = np.mean(seasonal_demands)
        # Blend: 70% recent, 30% seasonal
        mu = 0.7 * mu + 0.3 * mu_seasonal

    # ---- Std forecast ----
    sigma = None

    if recent_demands and len(recent_demands) >= 3:
        recent_demand_values = [d for _, d in recent_demands]
        sigma = np.std(recent_demand_values, ddof=1)
    elif len(blocks) >= 3:
        all_demands = [b['two_week_demand'] for b in blocks[-12:]]  # Last 12 blocks
        sigma = np.std(all_demands, ddof=1)
    elif len(blocks) >= 2:
        all_demands = [b['two_week_demand'] for b in blocks]
        sigma = np.std(all_demands, ddof=1)
    else:
        # Only 1 block - use sqrt(mu) * global_cv as sigma estimate
        sigma = np.sqrt(max(mu, 0.1)) * 1.5  # Conservative estimate

    # Safety checks
    if sigma is None or np.isnan(sigma) or sigma <= 0:
        sigma = np.sqrt(max(mu, 0.1)) * 1.5

    if mu is None or np.isnan(mu):
        return None, None

    return mu, sigma


# ============================================================
# 6. NEWSVENTOR OPTIMAL TARGET
# ============================================================
def newsvendor_target(mu, sigma, a, b):
    """
    Calculate optimal inventory using newsvendor model.
    Assumes normally distributed demand.
    Q* = μ + z·σ  where z = Φ⁻¹(b/(a+b))
    """
    if mu is None or sigma is None:
        return 0

    mu = max(mu, 0)
    sigma = max(sigma, 0.01)

    critical_ratio = b / (a + b)
    # Clamp critical ratio to avoid numerical issues
    critical_ratio = np.clip(critical_ratio, 0.001, 0.999)
    z = stats.norm.ppf(critical_ratio)

    target = mu + z * sigma
    target = max(0, round(target))
    return int(target)


# ============================================================
# 7. HIERARCHICAL FALLBACK FOR SPARSE ITEMS
# ============================================================
print("Building fallback statistics...")

# Item-level global stats (across all stores)
item_weekly_stats = weekly.groupby('item_id')['weekly_demand'].agg(['mean', 'std', 'count']).reset_index()
item_weekly_stats.columns = ['item_id', 'item_mean', 'item_std', 'item_count']
# Fill NaN std with mean-based estimate
item_weekly_stats['item_std'] = item_weekly_stats['item_std'].fillna(
    np.sqrt(item_weekly_stats['item_mean']) * 1.5
)

# Global stats
global_mean_weekly = weekly['weekly_demand'].mean()
global_std_weekly = weekly['weekly_demand'].std()

# Store-level scale factors (how much each store's demand is relative to "all")
store_factors = {}
for sc in ['1', '2', '3', '4', '5']:
    sc_total = weekly[weekly['store_code'] == sc]['weekly_demand'].sum()
    all_total = weekly[weekly['store_code'] == 'all']['weekly_demand'].sum()
    if all_total > 0:
        store_factors[sc] = sc_total / all_total
    else:
        store_factors[sc] = 0.2  # uniform

print(f"Store scale factors: {store_factors}")
print(f"Global weekly mean: {global_mean_weekly:.2f}, std: {global_std_weekly:.2f}")


# ============================================================
# 8. MAIN PREDICTION LOOP
# ============================================================
print("Predicting targets for all item-store combos...")

results = []

# Build a quick lookup for config
config_lookup = {}
for _, row in config.iterrows():
    config_lookup[(row['item_id'], row['store_code'])] = (row['a'], row['b'])

# Track stats for reporting
forecast_count = 0
fallback_count = 0
zero_demand_count = 0

for (item_id, store_code), (a, b) in config_lookup.items():
    key = (item_id, store_code)
    blocks = item_store_blocks.get(key, [])

    # Try primary forecast
    mu, sigma = forecast_item_store(blocks, TARGET_END_WEEK)

    if mu is not None and mu >= 0:
        forecast_count += 1
    else:
        # Fallback: use item-level stats
        fallback_count += 1
        item_stats = item_weekly_stats[item_weekly_stats['item_id'] == item_id]
        if len(item_stats) > 0:
            item_mean_w = item_stats.iloc[0]['item_mean']
            item_std_w = item_stats.iloc[0]['item_std']
        else:
            item_mean_w = global_mean_weekly
            item_std_w = global_std_weekly

        # 2-week forecast from weekly stats
        # Assuming independent weeks: mu_2w = 2 * mu_1w, sigma_2w = sqrt(2) * sigma_1w
        mu = 2 * max(item_mean_w, 0)
        sigma = np.sqrt(2) * max(item_std_w, 0.01)

        if mu == 0 and sigma < 0.1:
            zero_demand_count += 1

    # Calculate optimal target
    target = newsvendor_target(mu, sigma, a, b)
    results.append({
        'item_id': item_id,
        'store_code': store_code,
        'target': target
    })

# ============================================================
# 9. OUTPUT
# ============================================================
output_df = pd.DataFrame(results)
output_df = output_df[['item_id', 'store_code', 'target']]  # Ensure column order

output_path = 'D:/case1/case1/target_inventory.csv'
output_df.to_csv(output_path, index=False)

print(f"\n=== Results ===")
print(f"Total predictions: {len(results)}")
print(f"Primary forecasts: {forecast_count}")
print(f"Fallback forecasts: {fallback_count}")
print(f"Zero-demand items: {zero_demand_count}")
print(f"\nTarget statistics:")
print(output_df['target'].describe())
print(f"\nOutput saved to: {output_path}")

# Preview
print("\n=== Preview (first 10 rows) ===")
print(output_df.head(10).to_string())
print("\n=== Preview (sample with store_code='all') ===")
print(output_df[output_df['store_code'] == 'all'].head(10).to_string())
