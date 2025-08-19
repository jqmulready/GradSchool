import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
 
# Step 1: Compute Moments
def compute_moments(data):
    m1 = np.mean(data)
    m2 = np.mean(data ** 2)
    m3 = np.mean(data ** 3)
    m4 = np.mean(data ** 4)
    return m1, m2, m3, m4
 
# Step 2: Simulate Paths
def simulate_paths(m1, m2, m, r_min, r_max):
    r_values = np.linspace(r_min, r_max, 1000)
    paths = np.zeros((len(r_values), m))
    for i, r in enumerate(r_values):
        paths[i, :] = np.random.normal(m1, np.sqrt(m2), m)
    return r_values, paths
 
# Step 3: Compute Path Moments
def compute_path_moments(paths):
    path_m1 = np.mean(paths, axis=1)
    path_m2 = np.mean(paths ** 2, axis=1)
    path_m3 = np.mean(paths ** 3, axis=1)
    path_m4 = np.mean(paths ** 4, axis=1)
    return path_m1, path_m2, path_m3, path_m4
 
# Step 5: Compute Higher-Order Conditional Moment
def compute_higher_order_moment(m4):
    m6 = m4 - .00000003
    return m6
 
data = yf.download('TSLA', period='3y', interval='1d') 
data['Returns'] = data['Close'].pct_change()


# Sample Data
data = data['Returns']
r_min = np.min(data)
r_max = np.max(data)
 
# Step 1: Compute Moments
m1, m2, m3, m4 = compute_moments(data)
 
# Step 2: Simulate Paths
m = 1000
r_values, paths = simulate_paths(m1, m2, m, r_min, r_max)
 
# Step 3: Compute Path Moments
path_m1, path_m2, path_m3, path_m4 = compute_path_moments(paths)
 
# Step 4: Compare to Confidence Bands
median_m1 = np.median(path_m1)
percentile_10_m1 = np.percentile(path_m1, 10)
percentile_90_m1 = np.percentile(path_m1, 90)
 
median_m2 = np.median(path_m2)
percentile_10_m2 = np.percentile(path_m2, 10)
percentile_90_m2 = np.percentile(path_m2, 90)
 
median_m3 = np.median(path_m3)
percentile_10_m3 = np.percentile(path_m3, 10)
percentile_90_m3 = np.percentile(path_m3, 90)
 
median_m4 = np.median(path_m4)
percentile_10_m4 = np.percentile(path_m4, 10)
percentile_90_m4 = np.percentile(path_m4, 90)
 
# Step 5: Compute Higher-Order Conditional Moment
m6 = compute_higher_order_moment(m4)
 
# Plot Results
plt.figure(figsize=(10, 6))
 
plt.subplot(2, 1, 1)
plt.plot(r_values, path_m1, label='Path Moments M1')
plt.axhline(y=median_m1, color='r', linestyle='--', label='Median')
plt.axhline(y=percentile_10_m1, color='g', linestyle='--', label='10th Percentile')
plt.axhline(y=percentile_90_m1, color='b', linestyle='--', label='90th Percentile')
plt.title('Path Moments M1 and Confidence Bands')
plt.legend()
 
plt.subplot(2, 1, 2)
plt.plot(r_values, path_m2, label='Path Moments M2')
plt.axhline(y=median_m2, color='r', linestyle='--', label='Median')
plt.axhline(y=percentile_10_m2, color='g', linestyle='--', label='10th Percentile')
plt.axhline(y=percentile_90_m2, color='b', linestyle='--', label='90th Percentile')
plt.title('Path Moments M2 and Confidence Bands')
plt.legend()
 
plt.tight_layout()
plt.show()
 
# Step 6: Estimate Jump Size Variance
def estimate_jump_size_variance(m4, m6):
    return m6 / (5 * m4)
 
# Step 7: Estimate Jump Intensity
def estimate_jump_intensity(m4, sigma_y_sq):
    return m4 / (3 * sigma_y_sq)
 
# Compute Jump Size Variance
sigma_y_sq = estimate_jump_size_variance(m4, m6)
 
# Estimate Jump Intensity
jump_intensity = estimate_jump_intensity(m4, sigma_y_sq)
 
print("Estimated Jump Size Variance:", sigma_y_sq)
print("Estimated Jump Intensity:", jump_intensity)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Step 6: Estimate Jump Size Variance
def estimate_jump_size_variance(m4, m6):
    return m6 / (5 * m4)
 
# Step 7: Estimate Jump Intensity
def estimate_jump_intensity(m4, sigma_y_sq):
    return m4 / (3 * sigma_y_sq)
 
# Function to calculate jump intensity for a given window of returns
def calculate_rolling_jump_intensity(returns_window):
    m4 = np.mean(returns_window ** 4)
    m6 = np.mean(returns_window ** 6)
    sigma_y_sq = estimate_jump_size_variance(m4, m6)
    jump_intensity = estimate_jump_intensity(m4, sigma_y_sq)
    return jump_intensity
 
 
# 'Returns' column contains the returns of the stock
# 'Date' column is in datetime format
# Adjust the window size as needed
window_size = 90  # 3-month window
 
# Calculate rolling jump intensity
rolling_jump_intensity = data.rolling(window=window_size).apply(calculate_rolling_jump_intensity)
 
# Plot rolling jump intensity
plt.figure(figsize=(10, 6))
rolling_jump_intensity.plot(color='blue')
plt.title('Rolling Jump Intensity (3-Month Window)')
plt.xlabel('Date')
plt.ylabel('Jump Intensity')
plt.grid(True)
plt.show()
