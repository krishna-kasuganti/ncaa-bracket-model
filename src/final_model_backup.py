### FINAL MODEL
final_model = #DECIDED ON BEST MODEL
final_predictions = final_model.predict(X_test)
final_predictions
final_accuracy = accuracy_score(y_test, final_predictions)
final_logloss = log_loss(y_test, final_model.predict_proba(X_test))
final_brier_score = brier_score_loss(y_test, final_model.predict_proba(X_test)[:,1])

final_accuracy, final_logloss, final_brier_score
## Interpreting Results
teams = pd.read_csv("../data/raw/Mteams.csv")
teams.head()
final_features = pd.read_parquet("../data/processed/final_features.parquet")


final_features = final_features[final_features['Season'] == 2024].reset_index(drop=True)
final_features = final_features.copy()
final_features.head()
final_features['Predicted'] = final_predictions
final_features['WinProbability'] = final_model.predict_proba(X_test)[:,1]
final_features['Correct'] = final_features['Predicted'] == final_features['Results']

final_features.head()
final_features = pd.merge(final_features, teams, left_on='TeamA_Id', right_on='TeamID', how='left')
final_features = pd.merge(final_features, teams, left_on='TeamB_Id', right_on='TeamID', how='left')
final_results = final_features[['Season', 'DayNum', 'TeamA_Id', 'TeamB_Id', 'Predicted', 'WinProbability', 'Correct', 'TeamName_x', 'TeamName_y']]

conditions = [
    final_results['DayNum'] == 154,
    final_results['DayNum'] == 152,
    (final_results['DayNum'] >= 145) & (final_results['DayNum'] < 152),
    (final_results['DayNum'] >= 143) & (final_results['DayNum'] < 145),
    (final_results['DayNum'] >= 138) & (final_results['DayNum'] < 143),
    final_results['DayNum'] < 138
]

values = ["Championship", "Final Four", "Elite 8", "Sweet 16", "Round of 32", "Round of 64"]

final_results['Round'] = np.select(conditions, values, default="Unknown")
#Cleaning Up
final_results = final_results.drop(columns=['DayNum', 'TeamA_Id', 'TeamB_Id'])
final_results.rename(columns={'TeamName_x': 'TeamA', 'TeamName_y': 'TeamB', 'WinProbability': 'TeamA_WinProb', 'Predicted': 'Model_Pick'}, inplace=True)
final_results = final_results[['Season', 'Round', 'TeamA', 'TeamB', 'Model_Pick', 'TeamA_WinProb', 'Correct']]
final_results
