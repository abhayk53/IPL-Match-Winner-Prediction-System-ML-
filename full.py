import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'venue']]
    df.dropna(inplace=True)
    return df


def encode_teams(df):
    all_teams = pd.concat([
        df['team1'],
        df['team2'],
        df['toss_winner'],
        df['winner']
    ]).unique()

    team_encoding = {}
    reverse_team_encoding = {}

    for index, team in enumerate(all_teams):
        team_encoding[team] = index
        reverse_team_encoding[index] = team

    df['team1'] = df['team1'].map(team_encoding)
    df['team2'] = df['team2'].map(team_encoding)
    df['toss_winner'] = df['toss_winner'].map(team_encoding)
    df['winner'] = df['winner'].map(team_encoding)

    return df, team_encoding, reverse_team_encoding


def encoded_toss_decision(df):
    toss_decision_encoded = []

    for dec in df['toss_decision']:
        if isinstance(dec, str) and dec.lower() == 'bat':
            toss_decision_encoded.append(1)
        else:
            toss_decision_encoded.append(0)

    df['toss_decision'] = toss_decision_encoded
    return df


def encode_venue(df):
    all_venues = df['venue'].unique()
    venue_encoding = {}

    for index, venue in enumerate(all_venues):
        venue_encoding[venue] = index

    df['venue'] = df['venue'].map(venue_encoding)

    return df, venue_encoding


def train_model(df):
    X = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']]
    y = df['winner']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model


def predict_winner(model, team_encoding, reverse_team_encoding, venue_encoding,
                   team1, team2, toss_winner, toss_decision, venue):

    # Safety checks
    if team1 not in team_encoding:
        return "Invalid Team1", 0, 0
    if team2 not in team_encoding:
        return "Invalid Team2", 0, 0
    if toss_winner not in team_encoding:
        return "Invalid Toss Winner", 0, 0
    if venue not in venue_encoding:
        return "Invalid Venue", 0, 0

    t1 = team_encoding[team1]
    t2 = team_encoding[team2]
    tw = team_encoding[toss_winner]
    td = 1 if toss_decision.lower() == 'bat' else 0
    v = venue_encoding[venue]

    input_test = np.array([[t1, t2, tw, td, v]])

    probs = model.predict_proba(input_test)[0]
    class_labels = model.classes_
    prob_dict = dict(zip(class_labels, probs))

    team1_prob_raw = prob_dict.get(t1, 0)
    team2_prob_raw = prob_dict.get(t2, 0)

    total = team1_prob_raw + team2_prob_raw

    if total > 0:
        team1_prob = (team1_prob_raw / total) * 100
        team2_prob = (team2_prob_raw / total) * 100
    else:
        team1_prob = 50.0
        team2_prob = 50.0

    if team1_prob > team2_prob:
        winner = team1
    else:
        winner = team2

    return winner, team1_prob, team2_prob


def main():
    df = load_data("Match_Info_Final.csv")

    df, team_encoding, reverse_team_encoding = encode_teams(df)
    df = encoded_toss_decision(df)
    df, venue_encoding = encode_venue(df)

    model = train_model(df)

    # Example test
    winner, p1, p2 = predict_winner(
        model,
        team_encoding,
        reverse_team_encoding,
        venue_encoding,
        "Mumbai Indians",
        "Chennai Super Kings",
        "Mumbai Indians",
        "bat",
        "Wankhede Stadium"
    )

    print("Winner:", winner)
    print("Team1 Prob:", p1)
    print("Team2 Prob:", p2)


if __name__ == "__main__":
    main()