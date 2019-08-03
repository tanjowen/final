import pandas as pd
import numpy as np
import random
random.seed(145)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine


pitching_df = pd.read_sql_table('df_pitching', 'sqlite:///db/baseball.sqlite')
batting_df = pd.read_sql_table('df_batting', 'sqlite:///db/baseball.sqlite')

def read_team(team_abb):
	df = pd.read_sql_table(f'{team_abb}_data', 'sqlite:///db/baseball.sqlite')

	return df 

def std_dfs(year, pitching_df = pitching_df, batting_df = batting_df):
	pitching_df = pitching_df[pitching_df.year == year]
	batting_df = batting_df[batting_df.year == year]

	pitching_num = pitching_df.drop(['team', 'year'], axis=1)
	batting_num = batting_df.drop(['team', 'year', 'games'], axis=1)

	teams = list(batting_df[batting_df.year == year]['team'])

	def std_cols(df):
		scaler = MinMaxScaler()
		df = df.astype(float)

		scaled_stats = scaler.fit_transform(df)
		scaled_stats = pd.DataFrame(scaled_stats, columns = df.columns, index=teams)

		return scaled_stats

	pitching_df_scaled = std_cols(pitching_num)
	batting_df_scaled = std_cols(batting_num)

	return pitching_df_scaled, batting_df_scaled

def neighbors(opponent_league, scaled_df, team_abb, opponent_abb, similar_teams=6, n_neighbors=30):
    league = {'American':['BAL','BOS','CHW','CLE','DET','HOU','KCR','LAA','MIN','NYY','OAK','SEA','TBR','TEX','TOR'],
            'National':['ARI','ATL','CHC','CIN','COL','LAD','MIA','MIL','NYM','PHI','PIT','SDP','SFG','STL','WSN']}
    league_teams = [v for k, v in league.items() if opponent_league == k]
    league_teams = [item for sublist in league_teams for item in sublist]

    neighbors = NearestNeighbors(n_neighbors = n_neighbors)
    neighbors.fit(scaled_df)

    distance, team = neighbors.kneighbors(scaled_df[scaled_df.index == team_abb], n_neighbors=n_neighbors)
    
    difference_df = pd.DataFrame()
    difference_df['team'] = [team.item(i) for i in range(team.shape[1])]
    difference_df['distance'] = [distance.item(i) for i in range(distance.shape[1])]
    
    difference_df = difference_df.sort_values(by='team')

    neighbors_df = scaled_df.copy()
    neighbors_df['distance'] = difference_df['distance'].values 

    leagues = []
    for i in neighbors_df.index:
    	for k,v in league.items():
    		if i in v:
    			leagues.append(k)

    neighbors_df['league'] = leagues
    neighbors_df = neighbors_df[(neighbors_df.league == opponent_league) & (neighbors_df.index != opponent_abb)]
    neighbors_df = neighbors_df.sort_values(by = 'distance')
    neighbors_df = neighbors_df.iloc[:similar_teams,:]

    return neighbors_df

def simulation_batting(runs_team1, runs_team2, series=1, games=1, simulations=20000):
	winner = []
	total_games = []

	for i in range(simulations):
		team1_win = 0
		team2_win = 0
		for i in range(series):
			team1_score = np.random.choice(runs_team1)
			team2_score = np.random.choice(runs_team2)
			while team1_score == team2_score:
				team1_score = np.random.choice(runs_team1)
				team2_score = np.random.choice(runs_team2)
			if team1_score > team2_score:
				team1_win += 1 
			elif team2_score > team1_score:
				team2_win += 1
			if (team1_win == games) | (team2_win == games):
				winner.append([1 if team1_win == games else 0])
				total_games.append(team1_win + team2_win)
				break 
	winner = [item for sublist in winner for item in sublist]

	return winner, total_games

def simulation_pitching(pitching_team1, pitching_team2, series=1, games=1, simulations=20000):
	winner = []
	total_games = []

	for i in range(simulations):
		team1_win = 0 
		team2_win = 0 
		for i in range(series):
			team1_pitching = np.random.choice(pitching_team1)
			team2_pitching = np.random.choice(pitching_team2)
			while team1_pitching == team2_pitching:
				team1_pitching = np.random.choice(pitching_team1)
				team2_pitching = np.random.choice(pitching_team2)

			if team1_pitching < team2_pitching:
				team1_win += 1
			elif team2_pitching < team1_pitching:
				team2_win += 1
			else: 
				continue

			if (team1_win == games) | (team2_win == games):
				winner.append([1 if team1_win == games else 0])
				total_games.append(team1_win + team2_win)
				
				break
				
	winner = [item for sublist in winner for item in sublist]

	return winner, total_games

def win_percent(pitching_sim, batting_sim):
	winner = pd.DataFrame({'pitching':pitching_sim, 'batting':batting_sim})
	winner['combined'] = winner['pitching'] + winner['batting']
	final_percent = winner.combined.sum() / (len(winner) * 2)
	return final_percent

def give_me_the_number(team1_abb, team1_league, team2_abb, team2_league):
	pitching_df = pd.read_sql_table('df_pitching', 'sqlite:///db/baseball.sqlite')
	batting_df = pd.read_sql_table('df_batting', 'sqlite:///db/baseball.sqlite')
	pitching_df_scaled, batting_df_scaled = std_dfs(2019)

	team1_data = read_team(team1_abb)
	team2_data = read_team(team2_abb)

	team1_pitching_neighbors = neighbors(team1_league, pitching_df_scaled, team1_abb, team2_abb)
	team2_pitching_neighbors = neighbors(team2_league, pitching_df_scaled, team2_abb, team1_abb)

	team1_batting_neighbors = neighbors(team1_league, batting_df_scaled, team1_abb, team2_abb)
	team2_batting_neighbors = neighbors(team2_league, batting_df_scaled, team2_abb, team1_abb)

	team1_pitching = team1_data[team1_data.opponent.isin(team2_batting_neighbors.index)]
	team2_pitching = team2_data[team2_data.opponent.isin(team1_batting_neighbors.index)]

	team1_batting = team1_data[team1_data.opponent.isin(team2_pitching_neighbors.index)]
	team2_batting = team2_data[team2_data.opponent.isin(team1_pitching_neighbors.index)]

	batting_winner, batting_totalgames = simulation_batting(team1_batting['runs_for'], team2_batting['runs_for'])
	pitching_winner, pitching_totalgames = simulation_pitching(team1_pitching['runs_against'], team2_pitching['runs_against'])

	team1_win_prob = win_percent(batting_winner, pitching_winner)
	team2_win_prob = 1-team1_win_prob

	team1_percent_print = print('Team 1: {:.0f}%'.format(team1_win_prob*100))
	team2_percent_print = print('Team 2: {:.0f}%'.format(team2_win_prob*100))

	team1_percent = 'Team 1: {:.0f}%'.format(team1_win_prob*100)
	team2_percent = 'Team 2: {:.0f}%'.format(team2_win_prob*100)

	return team1_percent, team2_percent

