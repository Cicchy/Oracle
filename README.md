#Oracle

Oracle is an AI model that predicts match outcomes based on historical match data. Oracle is capable

##Description
The model is trained on a dataset that includes information from [The Blue Alliance](https://www.thebluealliance.com/) for basic match data, such as teams and points, and [Statbotics](statbotics.io) for team performance evaluation metrics like the [EPA model](https://www.statbotics.io/blog/intro). The model implements the XGBoost algorithm to predict the score breakdown of a match.

##Set up
1.  **Clone the repository:**

    ```bash
    git clone <URL_del_repositorio>
    ```
2.  **Install dependencies:**

    ```bash
    pip install pandas scikit-learn xgboost
    ```
##How to use it?
You will find various models trained in the available dataset in the models folder. The models are classified in the following way:

[`dev`]: Model still in development or highly experimental
[`stable`]: More finished and tested model.
[`release`]: Official release used on the Oracle website.

> *We recommend you stick to the release version, but some stable versions should be fine for general use*

**General match data**
> *Match data fetched from [The Blue Alliance](https://www.thebluealliance.com/) used as a target forprediction *
[`year`]: The current year the match occurred.
[`event_key`]: The key assigned by the TBA for the event in which the match takes place.
[`comp_level`]: The match level in the competition. [`qm`] for qualification matches, [`qf`] for quarter finals, [`sf`] for semi finals, [`f`] for finals.
[`match_number`]: The number of the current match.
[`alliance_score`]:
[`alliance_auto_points`]:
[`alliance_teleop_points`]:
[`alliance_endgame_points`]:
[`alliance_rp`]: 

**Team data**
> *Individual team evaluation metrics, for better understanding of the EPA system, check out the [EPA model](https://www.statbotics.io/blog/intro).*
* [`team_epa`]: The overall Expected Points Added (EPA) for the team.
* [`team_epa_auto`]: The EPA for the team during the autonomous period.
* [`team_epa_teleop`]: The EPA for the team during the teleoperated period.
* [`team_epa_endgame`]: The EPA for the team during the endgame period.
* [`team_epa_trend`]: The trend of the team's EPA over time.
* [`team_win_rate`]: The team's overall win rate.
* [`team_win_streak`]: The team's current win streak.

**EPA data**

* [`epa_dif`]: The difference in EPA between the red and blue alliances.
* [`epa_ratio`]: The ratio of the red alliance's EPA to the blue alliance's EPA.


## Simple implementation

To use the model, load the trained model (`or_dev_2018-2025v1.1.0.pkl` or the latest release) and provide input data in the correct format. Here's a Python example:

```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Load the trained model
with open('models/or_dev_2018-2025v1.1.0.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the features
features = [
    'year', 'event_key', 'comp_level', 'match_number', 'epa_diff', 'epa_ratio',
    'red_1_win_rate', 'red_2_win_rate', 'red_3_win_rate', 'blue_1_win_rate',
    'blue_2_win_rate', 'blue_3_win_rate', 'red_1_epa_trend', 'red_2_epa_trend',
    'red_3_epa_trend', 'blue_1_epa_trend', 'blue_2_epa_trend', 'blue_3_epa_trend',
    'red_auto_points', 'blue_auto_points', 'red_teleop_points', 'blue_teleop_points'
]

# Function to predict probabilities
def predict_win_probability(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    input_df = pd.get_dummies(input_df, columns=['event_key', 'comp_level'], drop_first=True)
    # Reindex columns to match training data
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)
    imputer = SimpleImputer(strategy='mean')
    input_imputed = imputer.transform(input_df)
    scaler = StandardScaler()
    input_scaled = scaler.transform(input_imputed)
    win_probability = model.predict_proba(input_scaled)[:, 1][0]
    return win_probability

# Example input data
input_data = {
    'year': 2018, 'event_key': '2018abca', 'comp_level': 'qm', 'match_number': 6,
    'epa_diff': 100, 'epa_ratio': 1.2, 'red_1_win_rate': 0.8, 'red_2_win_rate': 0.7,
    'red_3_win_rate': 0.9, 'blue_1_win_rate': 0.6, 'blue_2_win_rate': 0.5,
    'blue_3_win_rate': 0.8, 'red_1_epa_trend': 0.1, 'red_2_epa_trend': -0.2,
    'red_3_epa_trend': 0.3, 'blue_1_epa_trend': -0.1, 'blue_2_epa_trend': 0.2,
    'blue_3_epa_trend': -0.3, 'red_auto_points': 20, 'blue_auto_points': 15,
    'red_teleop_points': 100, 'blue_teleop_points': 90
}

# Make predictions
win_probability = predict_win_probability(input_data)
print(f'Probability of Red Alliance Winning: {win_probability}')
