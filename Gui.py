import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 
from full import load_data, encode_teams, encoded_toss_decision, encode_venue, train_model 

st.set_page_config(page_title="IPL Winner Predictor", layout="wide") 

st.markdown(
"""
<style> 
.stApp { 
    background: url("https://wallpapercave.com/wp/wp3049868.jpg"); 
    background-size: cover; 
    background-position: center; 
    background-attachment: fixed; 
    position: relative; 
} 
.stApp::before { 
    content: ""; 
    position: absolute; 
    top: 0; 
    left: 0; 
    width: 100%; 
    height: 100%; 
    background: rgba(0,0,0,0.6); 
    z-index: 0; 
} 
.stApp > div { 
    position: relative; 
    z-index: 1; 
} 
</style> 
""",
unsafe_allow_html=True
)

df = load_data("Match_Info_Final.csv") 
df, team_encoding, reverse_team_encoding = encode_teams(df) 
df = encoded_toss_decision(df) 
df, venue_encoding = encode_venue(df) 
model = train_model(df) 

col1, col2, col3 = st.columns([2.3, 2, 1]) 

with col2: 
    st.image("logos/IPL.png", width=200) 

st.markdown("""
<h1 style='text-align: center; 
font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif; 
font-size: 48px; 
background: linear-gradient(to right, #FF5733, #1E90FF); 
-webkit-background-clip: text; 
-webkit-text-fill-color: transparent; 
font-weight: 900; 
margin-bottom: 20px;'>
Cricket Match Winner Predictor 
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='text-align: center; color: #AAAAAA; font-weight: 300; margin-top: -10px;'>
Powered by Machine Learning 
</h3>
""", unsafe_allow_html=True)

teams = sorted(list(team_encoding.keys())) 
venues = sorted(list(venue_encoding.keys())) 
toss_options = ['bat', 'field'] 

col1, col2, col3 = st.columns(3) 

with col1: 
    team1 = st.selectbox("Select Team 1", teams, key='team1') 

with col2: 
    team2 = st.selectbox("Select Team 2", [team for team in teams if team != team1], key='team2') 

with col3: 
    venue = st.selectbox("Select Venue", venues) 

col4, col5 = st.columns(2) 

with col4: 
    toss_winner = st.radio("Toss Winner", [team1, team2]) 

with col5: 
    toss_decision = st.radio("Toss Decision", toss_options) 

st.markdown("---") 

col1, col2, col3 = st.columns([0.7, 0.5, 5]) 

with col1: 
    st.image(f"logos/{team1}.png", width=150) 

with col2: 
    st.markdown(
        "<h2 style='text-align: center; color: #FF5733; margin-top: 60px;'>VS</h2>",
        unsafe_allow_html=True
    ) 

with col3: 
    st.image(f"logos/{team2}.png", width=150) 

st.markdown("") 

col1, col2, col3 = st.columns([1, 2, 1]) 

with col2: 
    predict_btn = st.button(" Predict Winner     ", use_container_width=True) 

if predict_btn: 

    t1 = team_encoding[team1] 
    t2 = team_encoding[team2] 
    tw = team_encoding[toss_winner] 
    td = 1 if toss_decision.lower() == 'bat' else 0 
    v = venue_encoding[venue] 

    input_test = np.array([[t1, t2, tw, td, v]]) 
    probs = model.predict_proba(input_test)[0] 

    team1_prob_raw = probs[t1] if t1 < len(probs) else 0 
    team2_prob_raw = probs[t2] if t2 < len(probs) else 0 

    total = team1_prob_raw + team2_prob_raw 

    if total > 0: 
        team1_prob = (team1_prob_raw / total) * 100 
        team2_prob = (team2_prob_raw / total) * 100 
    else: 
        team1_prob, team2_prob = 50.0, 50.0 

    winner = team1 if team1_prob > team2_prob else team2 

    total_matches = df.shape[0] 
    team1_matches = df[(df['team1'] == t1) | (df['team2'] == t1)].shape[0] 
    team2_matches = df[(df['team1'] == t2) | (df['team2'] == t2)].shape[0] 
    team1_wins = df[df['winner'] == t1].shape[0] 
    team2_wins = df[df['winner'] == t2].shape[0] 

    h2h_matches = df[((df['team1'] == t1) & (df['team2'] == t2)) | 
                     ((df['team1'] == t2) & (df['team2'] == t1))] 

    h2h_count = h2h_matches.shape[0] 
    h2h_team1_wins = h2h_matches[h2h_matches['winner'] == t1].shape[0] 
    h2h_team2_wins = h2h_matches[h2h_matches['winner'] == t2].shape[0] 

    total_matches_at_venue = df[df['venue'] == venue_encoding[venue]].shape[0] 

    st.markdown("### Match Summary") 

    st.info(f""" 
- **Venue**: {venue} 
- **Toss Winner**: {toss_winner} ({toss_decision.upper()}) 
- **{team1}**: {team1_matches} matches, {team1_wins} wins 
- **{team2}**: {team2_matches} matches, {team2_wins} wins 
- **{team1} VS {team2}**: {h2h_count} matches 
    - {team1}: {h2h_team1_wins} wins 
    - {team2}: {h2h_team2_wins} wins 
""") 

    st.success(f"**Predicted Winner: {winner}**") 

    st.markdown("### Winning Probability") 

    col1, col2, col3 = st.columns([1, 1, 1]) 

    with col2: 
        fig, ax = plt.subplots(figsize=(5, 5)) 
        ax.bar([team1, team2], [team1_prob, team2_prob], color=['#1f77b4', '#2ca02c']) 
        ax.set_ylabel("Win Probability (%)") 
        ax.set_ylim(0, 100) 
        ax.set_title("Prediction Confidence") 
        fig.tight_layout() 
        st.pyplot(fig) 

    col_m1, col_m2, col_m3 = st.columns([1, 1, 1]) 

    with col_m1: 
        st.markdown(f"<h5 style='text-align: center;'>{team1} Win Chance</h5>", unsafe_allow_html=True) 
        st.markdown(f"<h2 style='text-align: center;'>{team1_prob:.2f}%</h2>", unsafe_allow_html=True) 

    with col_m3: 
        st.markdown(f"<h5 style='text-align: center;'>{team2} Win Chance</h5>", unsafe_allow_html=True) 
        st.markdown(f"<h2 style='text-align: center;'>{team2_prob:.2f}%</h2>", unsafe_allow_html=True) 

st.markdown("---") 

st.markdown(
"<div style='text-align: center; font-size: 14px;'>"
"Cricket Match Winner Predictor using Machine Learning"
"</div>",
unsafe_allow_html=True
)