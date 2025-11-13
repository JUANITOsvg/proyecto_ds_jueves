import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load datasets
races = pd.read_csv('F1_datasets/races.csv')
results = pd.read_csv('F1_datasets/results.csv')
sprints = pd.read_csv('F1_datasets/sprint_results.csv')
lap_times = pd.read_csv('F1_datasets/lap_times.csv')
standings = pd.read_csv('F1_datasets/driver_standings.csv')
qualifying = pd.read_csv('F1_datasets/qualifying.csv')
drivers = pd.read_csv('F1_datasets/drivers.csv')

# Feature engineering
def get_driver_features():
    # Average race position
    avg_race_pos = results.groupby('driverId')['positionOrder'].mean().rename('avg_race_pos')
    # Average sprint position
    avg_sprint_pos = sprints.groupby('driverId')['positionOrder'].mean().rename('avg_sprint_pos')
    # Average lap time (in seconds)
    lap_times['lap_time_sec'] = lap_times['milliseconds'] / 1000
    avg_lap_time = lap_times.groupby('driverId')['lap_time_sec'].mean().rename('avg_lap_time')
    # Current driver standing points
    latest_race_id = standings['raceId'].max()
    latest_standings = standings[standings['raceId'] == latest_race_id][['driverId', 'points']].set_index('driverId')
    # Average qualifying position
    avg_qual_pos = qualifying.groupby('driverId')['position'].mean().rename('avg_qual_pos')
    # Merge all features
    features = pd.concat([avg_race_pos, avg_sprint_pos, avg_lap_time, latest_standings, avg_qual_pos], axis=1).dropna()
    return features

features = get_driver_features()

# Prepare target variable: 1 if driver won a race, else 0
results['win'] = (results['positionOrder'] == 1).astype(int)
win_counts = results.groupby('driverId')['win'].sum()
features = features.join(win_counts.rename('win_count')).fillna(0)

# Prepare data for modeling
X = features[['avg_race_pos', 'avg_sprint_pos', 'avg_lap_time', 'points', 'avg_qual_pos']]
y = (features['win_count'] > 0).astype(int)  # Has the driver ever won

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Predict probability of winning for each driver
win_probs = model.predict_proba(X_scaled)[:, 1]
features['win_probability'] = win_probs

# Output probabilities
output = features[['win_probability']]
output = output.join(drivers.set_index('driverId')[['surname']])
output = output.sort_values('win_probability', ascending=False)

print("Win probabilities for each driver:")
print(output[['surname', 'win_probability']])
