import datetime
import string
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from basketball_reference_scraper.pbp import get_pbp
from basketball_reference_scraper.constants import TEAM_TO_TEAM_ABBR
from basketball_reference_scraper.seasons import get_schedule

#  Get the nba schedule (all games)
year = 2021
nba = get_schedule(year, playoffs=False)
pbps = {}

teams = [
 'PHILADELPHIA 76ERS',
 'ATLANTA HAWKS',
 'BOSTON CELTICS',
 'BROOKLYN NETS',
 'CHICAGO BULLS',
 'CHARLOTTE HORNETS',
 'CLEVELAND CAVALIERS',
 'DALLAS MAVERICKS',
 'DENVER NUGGETS',
 'DETROIT PISTONS',
 'GOLDEN STATE WARRIORS',
 'HOUSTON ROCKETS',
 'INDIANA PACERS',
 'LOS ANGELES CLIPPERS',
 'LOS ANGELES LAKERS',
 'MEMPHIS GRIZZLIES',
 'MIAMI HEAT',
 'MILWAUKEE BUCKS',
 'MINNESOTA TIMBERWOLVES',
 'NEW ORLEANS PELICANS',
 'NEW YORK KNICKS',
 'OKLAHOMA CITY THUNDER',
 'ORLANDO MAGIC',
 'PHOENIX SUNS',
 'PORTLAND TRAIL BLAZERS',
 'SACRAMENTO KINGS',
 'SAN ANTONIO SPURS',
 'TORONTO RAPTORS',
 'UTAH JAZZ',
 'WASHINGTON WIZARDS'
]
 
philly =  'PHILADELPHIA 76ERS'  # philly is separated because python doesn't like digits in the name!


#  Indexing pbp dataframe
QTR_IDX = 0
TIME_IDX = 1
HOME_IDX = 2
AWAY_IDX = 3

#  Indexing numpy array "plays"
TEAM_IDX = 0
TIME_IDX = 1
SCOREPLAY_IDX = 2
BENEFIT_IDX = 3
DETRIMENT_IDX = 4
MOMENTUM_IDX = 5

results = ['LOSS','WIN']

def shuffle_arrays(a, b, c=None):
    assert len(a) == len(b) and len(b) == len(c)
    p = np.random.permutation(len(a))
    a = [a[i] for i in p]
    b = [b[i] for i in p]
    c = [c[i] for i in p]
    
    return a, b, c

def name_to_abbr(team):
    return TEAM_TO_TEAM_ABBR[team.upper()]

def calculate_elapsed_time(remaining, quarter):
    qtr = '12:00.0' # 12-minute regulation quarters
    ot = '5:00.0'  # 5-minute overtime periods
    format = '%M:%S.%f'

    if 'OT' not in str(quarter): #  play occurs during regulation
        period = qtr
        quarter_progression = datetime.timedelta(minutes=(quarter-1)*12) 
    else: #  play occurs during overtime
        period = ot
        ot_period = int(quarter.strip(string.ascii_letters))
        quarter_progression = datetime.timedelta(minutes=(ot_period-1)*5)  
        
    return (quarter_progression + 
            datetime.datetime.strptime(period, format) - 
            datetime.datetime.strptime(remaining, format))

# def penalty_play(play):
#     results = ['foul']

def scoring_play(play):
    results = ['makes 2-pt', 'makes 3-pt']
    if any(word in play for word in results):
        return 1
    return 0
    
def benifit_play(play):
    results = ['rebound', 'foul']
    if any(word in play for word in results):
        return 1
    return 0

def detriment_play(play):
    results = ['turnover', 'miss', 'timeout']
    if any(word in play for word in results):
        return 1
    return 0
    
def process_play(time, play):
    return np.array([time, scoring_play(play), benifit_play(play), detriment_play(play)])

def calculate_momentum(play_arr):
    return (1+play_arr[SCOREPLAY_IDX]) * (1+play_arr[BENEFIT_IDX]) - 2*play_arr[DETRIMENT_IDX]
    
def momentum_checks(plays):
    playcount = plays.shape[0]
    
    for play in range(playcount):
        idx = play  # start from the current play
        while idx >= 0:
            idx -= 1
            if plays[play][TIME_IDX] - plays[idx][TIME_IDX] > 60:
                break
            plays[play][MOMENTUM_IDX] += calculate_momentum(plays[idx])

    return plays

def run(nba, model=LinearSVC(random_state=0), printouts=True):

    accuracies = []
    predicted_team_records = {team: 0 for team in teams}

    for _team in teams:
        team = _team.title() if _team != philly else 'Philadelphia 76ers'
        predicted_team_records[team] = predicted_team_records.pop(_team)
        
        #  Remove games that have not yet been played
        season = nba[~np.isnan(nba.VISITOR_PTS)]
        season = nba[~np.isnan(nba.HOME_PTS)]

        #  Select games played by a specific team if desired
        team_season = pd.concat((season[season.VISITOR == team], season[season.HOME== team]), axis=0)


        #  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        games_pbp = []
        wins = []
        game_summaries = []

        playcount_stopper = 200

        #  Extract parameters used by play-by-play functions 
        for idx,row in team_season.iterrows():
            date = row['DATE']
            away = name_to_abbr(row['VISITOR'])
            home = name_to_abbr(row['HOME'])
            home_winner = row['HOME_PTS'] > row['VISITOR_PTS']
            winner = home if home_winner else away

            summary = f'{date.date()}: {away} @ {home} | winner: {winner}'
            game_summaries.append(summary)

            wins.append(1 if winner == name_to_abbr(team) else 0)

            #  Begin processing play-by-play data on a per-game basis
            game_key = str(date) + str(home) + str(away)  # cache pbp data between uses to save time on the network
            
            if game_key not in pbps.keys():
                pbp = get_pbp(date, home, away)
                pbps[game_key] = pbp
            else:
                pbp = pbps[game_key]
                
            pbp_playcount = len(pbp)

            # Stop collecting pbp data after n plays 
            # Comment this line to use the full pbp data; 
            # this may introduce problems as teams play different number of games (reschedule, cancel, playoffs...)
            pbp_playcount = playcount_stopper


            plays = np.zeros((pbp_playcount, 6))  # [HOME?, time (secs), score, benefit, detriment, momentum_score]    

            for play in range(pbp_playcount):
                elapsed_gametime = calculate_elapsed_time(pbp.iloc[play][TIME_IDX], pbp.iloc[play][QTR_IDX])
                action_home = pbp.iloc[play][HOME_IDX]
                action_away = pbp.iloc[play][AWAY_IDX]

                if action_home is not np.nan:
                    plays[play] = np.hstack((1, process_play(elapsed_gametime.seconds, action_home.lower()), 0))
                else:
                    plays[play] = np.hstack((0, process_play(elapsed_gametime.seconds, action_away.lower()), 0))

            plays = momentum_checks(plays)
            plays = plays / plays.max(axis=0)

            games_pbp.append(plays)

        #  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        if year == 2021:  # In 2020-21, the regular seasons was capped at 72 games instead of 82.
            games_pbp = games_pbp[:72]
            wins = wins[:72]
            game_summaries = game_summaries[:72]
        
        if printouts:
                print(np.array(games_pbp).shape)     
        
        game_count = len(games_pbp)

        games_pbp, wins, game_summaries = shuffle_arrays(np.array(games_pbp), np.array(wins), game_summaries)

        tts_idx = round(game_count*0.75)  # train-test-split index

        X_train = np.array(games_pbp)[:tts_idx].reshape(tts_idx,-1)
        y_train = np.array(wins)[:tts_idx].reshape(-1,)

        X_test = np.array(games_pbp)[tts_idx:].reshape(game_count-tts_idx,-1)
        y_test = np.array(wins)[tts_idx:].reshape(-1,)

        #  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        clf = model.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)

        if printouts:
                print(f'Team: {team} accuracy = {100*acc}%')

        test_predictions = [results[i] for i in preds]

        for i in range(tts_idx, game_count): # indexing only the test data
            if preds[i-tts_idx] == 1:
                predicted_team_records[team] += 1
            if printouts:
                print(f'{game_summaries[i]} ||| Prediction: {name_to_abbr(team)} {test_predictions[i-tts_idx]}')

        #  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        
    test_games = game_count-tts_idx  # The number of games that were used for testing
    
    league_accuracy = np.mean(accuracies)*100
    league_wins_avg = np.mean(list(predicted_team_records.values()))/test_games
    
    print(f'test_games = {test_games}')
    print(f'League accuracy average: {league_accuracy:.2f}%')
    print(f'League wins average: {league_wins_avg:.3f}%')
    

    for team in range(len(teams)):
        _team = teams[team].title() if teams[team] != philly else 'Philadelphia 76ers'
        print(f'({predicted_team_records[_team]}) Win rate: {(predicted_team_records[_team]/test_games):.3f} | Correctly predicted {accuracies[team]*100:.2f}% for team {teams[team]}')

    return league_accuracy, league_wins_avg
        
