import json
from urllib.request import urlopen
from pathlib import Path

import altair as alt
# from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from scipy.stats import ttest_ind, ttest_rel, skew
from sklearn.preprocessing import RobustScaler

__all__ = [
    "n_players",
    "load_allday_data",
    "load_stats_data",
    "convert_df",
    "get_subset",
    "alt_mean_price",
    "get_metrics",
    "get_position_group",
    "cols_to_keep",
    "score_columns",
    "td_mapping",
    "all_pos",
    "offense",
    "defense",
    "team_pos",
    "pos_groups",
    "positions",
    "rarities",
    "position_type_dict",
    "agg_dict",
    "main_date_ranges",
    "play_v_player_date_ranges",
    "get_ttests",
    "stats_date_ranges",
    "load_score_data",
    "load_ttest",
    "stats_subset",
    "load_player_data",
    "load_play_v_player_data",
    "load_headshot",
    "load_challenge_data",
    "player_mapping",
    "week_timings",
    "game_timings",
    "load_challenge_player_data",
    "alt_challenge_chart",
    "get_challenge_summary",
    "get_challenge_ttests",
    "alt_challenge_game",
    "load_challenge_reward",
    "load_pack_info",
    "load_pack",
    "pack_date_ranges",
    "alt_pack_sales",
    "load_player_pack",
    "load_pack_cache",
    "series2_mint1_standard_proportions",
    "series2_mint1_premium_proportions",
    "get_pack_value",
    "mint_pack",
    "load_series2_mint1_grouped",
    "load_pack_samples",
    "get_avg_pack_metrics",
    "load_player_mint",
    "load_simulation",
    # "get_player_name",
    "xgb_feature_names",
    "load_by_marketplace_id",
    "load_by_playername",
    "marketplace_id",
    "get_xgb_data",
    "rescale_xgb",
    "rarity_map"
]

cols_to_keep = [
    "Date",
    "Datetime",
    "marketplace_id",
    "Player",
    "Team",
    "Position",
    "Series",
    "Set_Name",
    # "Position Group",
    "Play_Type",
    "Season",
    "Week",
    "Moment_Date",
    "Game Outcome",
    "won_game",
    # "Scored Touchdown?",
    "Moment_Tier",
    "Rarity",
    "Moment_Description",
    "NFLALLDAY_ASSETS_URL",
    "Total_Circulation",
    "Price",
    "tx_id",
    "scored_td_in_moment",
    "pbp_td",
    "description_td",
    "scored_td_in_game",
    "game_td",
]

score_columns = [
    "Pass",
    "Reception",
    "Rush",
    "Strip Sack",
    "Interception",
    "Fumble Recovery",  # ~50% TD
    "Blocked Kick",  # 1/4 not td
    "Punt Return",  # all TD
    "Kick Return",  # 1/6 not td
]

td_mapping = {
    "scored_td_in_moment": "Best Guess (Moment TD)",
    "pbp_td": "Conservative (Moment TD)",
    "description_td": "Description only (Moment TD)",
    "scored_td_in_game": "Best Guess: (In-game TD)",
    "game_td": "Conservative (In-game TD)",
}


all_pos = ["All"]
offense = [
    "QB",
    "WR",
    "RB",
    "TE",
    "OL",
]
defense = [
    "DB",
    "DL",
    "LB",
]
team_pos = ["Team"]
pos_groups = ["All", "Offense", "Defense", "Team"]
positions = all_pos + offense + defense + team_pos
rarities = ["COMMON", "RARE", "LEGENDARY", "ULTIMATE"]
rarity_map = dict(zip([0,1,2,3], rarities))

# #TODO: clean up date ranges
main_date_ranges = [
    "All Time",
    "2022 Full Season",
    "2022 Week 1",
    "2022 Week 2",
    "2022 Week 3",
    "2022 Week 4",
    "2022 Week 5",
    "2022 Week 6",
    "2022 Week 7",
    "2022 Week 8",
]
play_v_player_date_ranges = [
    "All dates",
    "Since 2022 preseason",
    "Since 2022 Week 1",
    "Since 2022 Week 2",
    "Since 2022 Week 3",
    "Since 2022 Week 4",
    "Since 2022 Week 5",
    "Since 2022 Week 6",
    "Since 2022 Week 7",
    "Since 2022 Week 8",
]
stats_date_ranges = [
    "2022 Full Season",
    "2022 Week 1",
    "2022 Week 2",
    "2022 Week 3",
    "2022 Week 4",
    "2022 Week 5",
    "2022 Week 6",
    "2022 Week 7",
    "2022 Week 8",
]

position_type_dict = {
    "By Position": ("Position", positions),
    "By Group": ("Position Group", pos_groups),
    "By Rarity": ("Moment_Tier", rarities),
}

agg_dict = {
    "Player": "first",
    "Team": "first",
    "Position": "first",
    "Position Group": "first",
    "Play_Type": "first",
    "Season": "first",
    "Week": "first",
    "Moment_Date": "first",
    "Game Outcome": "first",
    "won_game": "first",
    "Series": "first",
    "Set_Name": "first",
    # "Scored Touchdown?": "first",
    "Description only (Moment TD)": "first",
    "Conservative (In-game TD)": "first",
    "Conservative (Moment TD)": "first",
    "Best Guess: (In-game TD)": "first",
    "Best Guess (Moment TD)": "first",
    "Moment_Tier": "first",
    "Rarity": "first",
    "Moment_Description": "first",
    "NFLALLDAY_ASSETS_URL": "first",
    "Total_Circulation": "first",
    "Price": "mean",
    "tx_id": "count",
}

stats_subset = [
    "player_id",
    "player_display_name",
    "position",
    "team",
    "headshot_url",
    "season",
    # "week",
    "fantasy_points_ppr",
    "passing_tds",
    "passing_yards",
    "receiving_tds",
    "receiving_yards",
    "rushing_tds",
    "rushing_yards",
]

player_mapping = {
    "Jakeem Grant Sr.": "Jakeem Grant",
    "Gardner Minshew II": "Gardner Minshew",
    "Cedrick Wilson Jr.": "Cedrick Wilson",
    "Michael Pittman Jr.": "Michael Pittman",
    "Odell Beckham Jr.": "Odell Beckham",
    "AJ Dillon": "A.J. Dillon",
    "Marvin Jones Jr.": "Marvin Jones",
    "Steve Smith Sr.": "Steve Smith",
    "Allen Robinson II": "Allen Robinson",
    "Laviska Shenault Jr.": "Laviska Shenault",
    "Gabriel Davis": "Gabe Davis",
    "Scotty Miller": "Scott Miller",
    "Patrick Mahomes II": "Patrick Mahomes",
    "DJ Chark Jr.": "D.J. Chark",
    "DJ Moore": "D.J. Moore",
    "Melvin Gordon III": "Melvin Gordon",
}

week_timings = {
    1: ("2022-09-08", "2022-09-15"),
    2: ("2022-09-15", "2022-09-22"),
    3: ("2022-09-22", "2022-09-29"),
    4: ("2022-09-29", "2022-10-06"),
    5: ("2022-10-06", "2022-10-13"),
    6: ("2022-10-13", "2022-10-20"),
    7: ("2022-10-20", "2022-10-27"),
    8: ("2022-10-27", "2022-11-03"),
}

game_timings = {
    1: {
        "thursday": (
            pd.Timestamp("2022-09-08 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-08 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-09-11 13:00:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-11 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-09-11 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-11 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-09-12 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-12 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-09-08 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-12 23:00:00-0400", tz="US/Eastern"),
        ),
    },
    2: {
        "thursday": (
            pd.Timestamp("2022-09-15 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-15 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-09-18 13:00:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-18 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-09-18 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-18 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-09-19 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-19 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-09-15 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-19 23:00:00-0400", tz="US/Eastern"),
        ),
    },
    3: {
        "thursday": (
            pd.Timestamp("2022-09-22 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-22 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-09-25 09:30:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-25 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-09-25 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-25 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-09-26 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-26 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-09-22 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-26 23:00:00-0400", tz="US/Eastern"),
        ),
    },
    4: {
        "thursday": (
            pd.Timestamp("2022-09-29 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-09-29 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-10-02 09:30:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-02 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-10-02 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-02 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-10-03 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-03 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-09-29 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-03 23:00:00-0400", tz="US/Eastern"),
        ),
        "bills_ravens": (
            pd.Timestamp("2022-10-02 13:00:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-02 16:15:00-0400", tz="US/Eastern"),
        ),
        "49ers_rams": (
            pd.Timestamp("2022-10-03 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-03 23:00:00-0400", tz="US/Eastern"),
        ),
    },
    5: {  # #TODO: update for all challenges
        "thursday": (
            pd.Timestamp("2022-10-06 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-06 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-10-09 09:30:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-09 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-10-09 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-09 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-10-10 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-10 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-10-06 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-10 23:00:00-0400", tz="US/Eastern"),
        ),
    },
    6: {  # #TODO: update for all challenges
        "thursday": (
            pd.Timestamp("2022-10-13 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-13 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-10-16 09:30:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-16 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-10-16 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-16 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-10-17 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-17 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-10-13 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-17 23:00:00-0400", tz="US/Eastern"),
        ),
    },
    7: {  # #TODO: update for all challenges
        "thursday": (
            pd.Timestamp("2022-10-20 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-20 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-10-23 09:30:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-23 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-10-23 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-23 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-10-24 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-24 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-10-20 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-24 23:00:00-0400", tz="US/Eastern"),
        ),
    },
    8: {  # #TODO: update for all challenges
        "thursday": (
            pd.Timestamp("2022-10-27 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-27 23:00:00-0400", tz="US/Eastern"),
        ),
        "slate": (
            pd.Timestamp("2022-10-30 09:30:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-30 19:30:00-0400", tz="US/Eastern"),
        ),
        "sunday_night": (
            pd.Timestamp("2022-10-30 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-30 23:00:00-0400", tz="US/Eastern"),
        ),
        "monday": (
            pd.Timestamp("2022-10-31 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-31 23:00:00-0400", tz="US/Eastern"),
        ),
        "weekly": (
            pd.Timestamp("2022-10-27 20:15:00-0400", tz="US/Eastern"),
            pd.Timestamp("2022-10-31 23:00:00-0400", tz="US/Eastern"),
        ),
    },
}

pack_date_ranges = [
    ("2021-12-10 06:00", "2021-12-13 23:59"),
    ("2021-12-17 06:00", "2021-12-21 08:00"),
    ("2022-01-07 06:00", "2022-01-07 23:59"),
    ("2022-01-14 06:00", "2022-01-14 23:59"),
    ("2022-02-16 06:00", "2022-02-18 23:59"),
    ("2022-02-25 06:00", "2022-02-25 23:59"),
    ("2022-03-02 06:00", "2022-03-02 23:59"),
    ("2022-03-04 06:00", "2022-03-04 23:59"),
    ("2022-03-11 06:00", "2022-03-11 23:59"),
    ("2022-09-27 06:00", "2022-09-27 23:59"),
    ("2022-10-11 06:00", "2022-10-11 23:59"),
    # #TODO need to update
]

player_pack_cols = [
    "Datetime",
    "Date",
    "tx_id",
    "Price",
    "Buyer",
    "Seller",
    "Player",
    "Team",
    "Position",
    "Season",
    "Week",
    "Play_Type",
    "Moment_Date",
    "Moment_Tier",
    "NFLALLDAY_ID",
    "Serial_Number",
    "Away_Team_Name",
    "Away_Team_Score",
    "Home_Team_Name",
    "Home_Team_Score",
    "Player_ID",
    "Player_Number",
    "Classification",
    "Total_Circulation",
    "NFT_ID",
    "Series",
    "Set_Name",
    "unique_id",
    "marketplace_id",
    "site",
    "Rarity",
    "Datetime_Reveal",
    "Moments_In_Pack",
    "Datetime_Pack",
    "Pack_Price",
    "Pack_Buyer",
    "Pack Type",
]


series2_mint1_standard_proportions = {
    "COMMON": 18182 / 22222,
    "RARE": 4000 / 22222,
    "LEGENDARY": 40 / 22222,
}
series2_mint1_premium_proportions = {
    "RARE": 3480 / 3800,
    "LEGENDARY": 320 / 3800,
}
n_players = 40

xgb_feature_names = [
    "Rarity",
    "Sales_Count",
    "Resell_Number",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "sack_fumbles_lost",
    "passing_epa",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "rushing_epa",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_fumbles_lost",
    "receiving_epa",
    "fantasy_points_ppr",
    "completions_season",
    "attempts_season",
    "passing_yards_season",
    "passing_tds_season",
    "interceptions_season",
    "sacks_season",
    "sack_fumbles_lost_season",
    "passing_epa_season",
    "carries_season",
    "rushing_yards_season",
    "rushing_tds_season",
    "rushing_fumbles_lost_season",
    "rushing_epa_season",
    "receptions_season",
    "targets_season",
    "receiving_yards_season",
    "receiving_tds_season",
    "receiving_fumbles_lost_season",
    "receiving_epa_season",
    "fantasy_points_ppr_season",
    "completions_career",
    "attempts_career",
    "passing_yards_career",
    "passing_tds_career",
    "interceptions_career",
    "sacks_career",
    "sack_fumbles_lost_career",
    "passing_epa_career",
    "carries_career",
    "rushing_yards_career",
    "rushing_tds_career",
    "rushing_fumbles_lost_career",
    "rushing_epa_career",
    "receptions_career",
    "targets_career",
    "receiving_yards_career",
    "receiving_tds_career",
    "receiving_fumbles_lost_career",
    "receiving_epa_career",
    "fantasy_points_ppr_career",
    "Series_Historical",
    "Series_Series 1",
    "Series_Series 2",
    "Game Outcome_Loss",
    "Game Outcome_Win",
    "Set_Name_importance_Base",
    "Set_Name_importance_Gridiron",
    "Set_Name_importance_Move the Chains",
    "Set_Name_importance_Opening Acts",
    "Set_Name_importance_Other",
]

marketplace_id = [
    "1001--Calvin_Johnson.csv.gzip",
    "1002--Tony_Gonzalez.csv.gzip",
    "1003--Jarvis_Landry.csv.gzip",
    "1007--Anquan_Boldin.csv.gzip",
    "1009--DeSean_Jackson.csv.gzip",
    "1010--Torry_Holt.csv.gzip",
    "1011--Julio_Jones.csv.gzip",
    "1012--Chad_Johnson.csv.gzip",
    "1013--Tony_Gonzalez.csv.gzip",
    "1015--Stefon_Diggs.csv.gzip",
    "1017--Odell_Beckham.csv.gzip",
    "1018--James_Conner.csv.gzip",
    "1019--Aaron_Rodgers.csv.gzip",
    "1020--Baker_Mayfield.csv.gzip",
    "1022--Kyler_Murray.csv.gzip",
    "1026--Damien_Harris.csv.gzip",
    "1029--Carson_Wentz.csv.gzip",
    "1031--Rashod_Bateman.csv.gzip",
    "1032--Justin_Fields.csv.gzip",
    "1033--D'Andre_Swift.csv.gzip",
    "1034--Clyde_Edwards-Helaire.csv.gzip",
    "1036--Justin_Jefferson.csv.gzip",
    "1037--Michael_Thomas.csv.gzip",
    "1038--Kyler_Murray.csv.gzip",
    "1039--Drake_London.csv.gzip",
    "1040--Kareem_Hunt.csv.gzip",
    "1042--Tony_Pollard.csv.gzip",
    "1043--Allen_Robinson.csv.gzip",
    "1044--Garrett_Wilson.csv.gzip",
    "1045--Geno_Smith.csv.gzip",
    "1046--Mike_Evans.csv.gzip",
    "1051--Sammy_Watkins.csv.gzip",
    "1052--Najee_Harris.csv.gzip",
    "1053--Sterling_Shepard.csv.gzip",
    "1056--Cooper_Rush.csv.gzip",
    "1059--Nick_Chubb.csv.gzip",
    "1060--Amon-Ra_St._Brown.csv.gzip",
    "1061--James_Robinson.csv.gzip",
    "1062--Saquon_Barkley.csv.gzip",
    "1063--Diontae_Johnson.csv.gzip",
    "1064--Jimmy_Garoppolo.csv.gzip",
    "1065--Devin_Duvernay.csv.gzip",
    "1070--Nelson_Agholor.csv.gzip",
    "1071--Davis_Mills.csv.gzip",
    "1074--Lamar_Jackson.csv.gzip",
    "1076--Tua_Tagovailoa.csv.gzip",
    "1080--Breece_Hall.csv.gzip",
    "1081--Deebo_Samuel.csv.gzip",
    "1083--Mike_Williams.csv.gzip",
    "1085--Stefon_Diggs.csv.gzip",
    "1086--Amari_Cooper.csv.gzip",
    "1088--Jameis_Winston.csv.gzip",
    "1091--Laviska_Shenault.csv.gzip",
    "1092--Jacoby_Brissett.csv.gzip",
    "1093--Jamaal_Williams.csv.gzip",
    "1094--Christian_Watson.csv.gzip",
    "1095--Matt_Ryan.csv.gzip",
    "1096--Cam_Akers.csv.gzip",
    "1097--Chris_Olave.csv.gzip",
    "1099--Kenny_Pickett.csv.gzip",
    "1100--Rashaad_Penny.csv.gzip",
    "1101--Leonard_Fournette.csv.gzip",
    "1107--Tyler_Allgeier.csv.gzip",
    "1109--Teddy_Bridgewater.csv.gzip",
    "1110--Tyler_Boyd.csv.gzip",
    "1111--Noah_Gray.csv.gzip",
    "1114--Jelani_Woods.csv.gzip",
    "1115--Jamal_Agnew.csv.gzip",
    "1116--Alexander_Mattison.csv.gzip",
    "1117--Christian_McCaffrey.csv.gzip",
    "1118--Khalil_Herbert.csv.gzip",
    "1119--Josh_Jacobs.csv.gzip",
    "1120--Mack_Hollins.csv.gzip",
    "1121--DeVante_Parker.csv.gzip",
    "1122--Daniel_Jones.csv.gzip",
    "1127--T.J._Hockenson.csv.gzip",
    "1128--DeVonta_Smith.csv.gzip",
    "1129--Dameon_Pierce.csv.gzip",
    "1130--George_Pickens.csv.gzip",
    "1131--CeeDee_Lamb.csv.gzip",
    "1132--Miles_Sanders.csv.gzip",
    "1134--Steve_McNair.csv.gzip",
    "1136--Brandon_Aiyuk.csv.gzip",
    "1137--Geno_Smith.csv.gzip",
    "353--Olamide_Zaccheaus.csv.gzip",
    "354--Mark_Andrews.csv.gzip",
    "355--Jerry_Jeudy.csv.gzip",
    "356--Brock_Wright.csv.gzip",
    "358--Ashton_Dulin.csv.gzip",
    "363--Foster_Moreau.csv.gzip",
    "364--Ben_Skowronek.csv.gzip",
    "366--Mack_Hollins.csv.gzip",
    "368--K.J._Osborn.csv.gzip",
    "369--Gardner_Minshew.csv.gzip",
    "370--Miles_Sanders.csv.gzip",
    "371--Chase_Claypool.csv.gzip",
    "372--Adrian_Peterson.csv.gzip",
    "373--Russell_Wilson.csv.gzip",
    "377--Kyler_Murray.csv.gzip",
    "378--Tee_Higgins.csv.gzip",
    "379--Michael_Gallup.csv.gzip",
    "380--Cooper_Kupp.csv.gzip",
    "381--Logan_Thomas.csv.gzip",
    "382--James_Conner.csv.gzip",
    "386--George_Kittle.csv.gzip",
    "387--Chris_Godwin.csv.gzip",
    "394--Cordarrelle_Patterson.csv.gzip",
    "395--Amari_Cooper.csv.gzip",
    "396--Michael_Pittman.csv.gzip",
    "397--Darrel_Williams.csv.gzip",
    "398--Taysom_Hill.csv.gzip",
    "400--Tony_Pollard.csv.gzip",
    "402--Damien_Harris.csv.gzip",
    "403--Braxton_Berrios.csv.gzip",
    "410--Diontae_Johnson.csv.gzip",
    "415--Justin_Herbert.csv.gzip",
    "416--Mike_Williams.csv.gzip",
    "417--DK_Metcalf.csv.gzip",
    "418--Odell_Beckham.csv.gzip",
    "419--Jakeem_Grant.csv.gzip",
    "421--Travis_Homer.csv.gzip",
    "423--Christian_Kirk.csv.gzip",
    "424--Russell_Gage.csv.gzip",
    "425--Josh_Allen.csv.gzip",
    "426--Cam_Newton.csv.gzip",
    "427--Robby_Anderson.csv.gzip",
    "428--Damiere_Byrd.csv.gzip",
    "429--Joe_Burrow.csv.gzip",
    "430--Jarvis_Landry.csv.gzip",
    "431--Albert_Okwuegbunam.csv.gzip",
    "433--Javonte_Williams.csv.gzip",
    "435--Laquon_Treadwell.csv.gzip",
    "437--Josh_Gordon.csv.gzip",
    "438--Hunter_Renfrow.csv.gzip",
    "439--Josh_Palmer.csv.gzip",
    "440--Alvin_Kamara.csv.gzip",
    "442--Saquon_Barkley.csv.gzip",
    "443--Keelan_Cole.csv.gzip",
    "444--Zach_Wilson.csv.gzip",
    "446--Tom_Brady.csv.gzip",
    "447--Mark_Andrews.csv.gzip",
    "450--Marquez_Callaway.csv.gzip",
    "451--James_Washington.csv.gzip",
    "452--George_Kittle.csv.gzip",
    "453--Rashaad_Penny.csv.gzip",
    "454--Cam_Sims.csv.gzip",
    "455--Davante_Adams.csv.gzip",
    "456--Justin_Herbert.csv.gzip",
    "458--Dalvin_Cook.csv.gzip",
    "459--Breshad_Perriman.csv.gzip",
    "460--Mike_Evans.csv.gzip",
    "470--Tyler_Huntley.csv.gzip",
    "471--D.J._Moore.csv.gzip",
    "472--Matthew_Stafford.csv.gzip",
    "473--Kyle_Rudolph.csv.gzip",
    "475--Jakeem_Grant.csv.gzip",
    "476--Derrick_Gore.csv.gzip",
    "477--Taysom_Hill.csv.gzip",
    "478--Leonard_Fournette.csv.gzip",
    "485--Patrick_Mahomes.csv.gzip",
    "486--Van_Jefferson.csv.gzip",
    "487--Kirk_Cousins.csv.gzip",
    "488--Tyler_Lockett.csv.gzip",
    "489--Aaron_Rodgers.csv.gzip",
    "490--Rashod_Bateman.csv.gzip",
    "492--Devin_Singletary.csv.gzip",
    "493--Gabe_Davis.csv.gzip",
    "494--Stefon_Diggs.csv.gzip",
    "496--Chuba_Hubbard.csv.gzip",
    "499--Nick_Chubb.csv.gzip",
    "501--Josh_Reynolds.csv.gzip",
    "502--Marquez_Valdes-Scantling.csv.gzip",
    "504--Cooper_Kupp.csv.gzip",
    "505--DeVante_Parker.csv.gzip",
    "506--Myles_Gaskin.csv.gzip",
    "507--Justin_Jefferson.csv.gzip",
    "508--Mac_Jones.csv.gzip",
    "509--Kenny_Golladay.csv.gzip",
    "511--Jalen_Hurts.csv.gzip",
    "516--A.J._Green.csv.gzip",
    "517--Tyler_Boyd.csv.gzip",
    "518--Javonte_Williams.csv.gzip",
    "519--Jared_Goff.csv.gzip",
    "520--Brandin_Cooks.csv.gzip",
    "521--Bryan_Edwards.csv.gzip",
    "522--Jonathan_Taylor.csv.gzip",
    "523--Travis_Kelce.csv.gzip",
    "524--Tyreek_Hill.csv.gzip",
    "527--Saquon_Barkley.csv.gzip",
    "532--Kyler_Murray.csv.gzip",
    "533--Davis_Mills.csv.gzip",
    "534--Brandon_Aiyuk.csv.gzip",
    "544--Matt_Ryan.csv.gzip",
    "545--Tua_Tagovailoa.csv.gzip",
    "546--N'Keal_Harry.csv.gzip",
    "547--Dallas_Goedert.csv.gzip",
    "548--Terry_McLaurin.csv.gzip",
    "551--Kyle_Pitts.csv.gzip",
    "553--Hayden_Hurst.csv.gzip",
    "554--Josh_Johnson.csv.gzip",
    "555--Isaiah_McKenzie.csv.gzip",
    "556--Khalil_Herbert.csv.gzip",
    "557--Ja'Marr_Chase.csv.gzip",
    "558--D'Ernest_Johnson.csv.gzip",
    "559--Jerry_Jeudy.csv.gzip",
    "561--Chris_Conley.csv.gzip",
    "562--Mo_Alie-Cox.csv.gzip",
    "563--Nyheim_Hines.csv.gzip",
    "564--T.Y._Hilton.csv.gzip",
    "565--Byron_Pringle.csv.gzip",
    "566--Clyde_Edwards-Helaire.csv.gzip",
    "568--Lil'Jordan_Humphrey.csv.gzip",
    "569--Evan_Engram.csv.gzip",
    "570--Braxton_Berrios.csv.gzip",
    "573--Diontae_Johnson.csv.gzip",
    "574--Cyril_Grayson.csv.gzip",
    "575--Dyami_Brown.csv.gzip",
    "576--Malik_Turner.csv.gzip",
    "578--Amon-Ra_St._Brown.csv.gzip",
    "579--Allen_Lazard.csv.gzip",
    "580--Hunter_Renfrow.csv.gzip",
    "582--Damien_Harris.csv.gzip",
    "583--Najee_Harris.csv.gzip",
    "584--Joe_Burrow.csv.gzip",
    "585--Tee_Higgins.csv.gzip",
    "586--Dak_Prescott.csv.gzip",
    "587--Aaron_Rodgers.csv.gzip",
    "595--Ryan_Tannehill.csv.gzip",
    "596--Kyler_Murray.csv.gzip",
    "597--Brandon_Powell.csv.gzip",
    "598--Deebo_Samuel.csv.gzip",
    "599--Ke'Shawn_Vaughn.csv.gzip",
    "606--Russell_Wilson.csv.gzip",
    "607--A.J._Brown.csv.gzip",
    "608--Jaylen_Waddle.csv.gzip",
    "609--Zach_Wilson.csv.gzip",
    "610--DeVonta_Smith.csv.gzip",
    "612--Antoine_Wesley.csv.gzip",
    "615--Devonta_Freeman.csv.gzip",
    "618--Darnell_Mooney.csv.gzip",
    "619--Baker_Mayfield.csv.gzip",
    "620--David_Njoku.csv.gzip",
    "622--A.J._Dillon.csv.gzip",
    "623--Dare_Ogunbowale.csv.gzip",
    "624--Marvin_Jones.csv.gzip",
    "626--Demarcus_Robinson.csv.gzip",
    "627--Derek_Carr.csv.gzip",
    "628--Zay_Jones.csv.gzip",
    "629--Keenan_Allen.csv.gzip",
    "632--Jakobi_Meyers.csv.gzip",
    "634--Boston_Scott.csv.gzip",
    "635--George_Kittle.csv.gzip",
    "636--D'Onta_Foreman.csv.gzip",
    "637--Kyle_Pitts.csv.gzip",
    "639--Austin_Ekeler.csv.gzip",
    "640--Mike_Williams.csv.gzip",
    "641--Tyler_Higbee.csv.gzip",
    "642--Alvin_Kamara.csv.gzip",
    "643--Michael_Carter.csv.gzip",
    "644--T.Y._Hilton.csv.gzip",
    "645--Ben_Roethlisberger.csv.gzip",
    "647--DK_Metcalf.csv.gzip",
    "652--Cedrick_Wilson.csv.gzip",
    "653--Noah_Fant.csv.gzip",
    "654--Aaron_Jones.csv.gzip",
    "655--Patrick_Mahomes.csv.gzip",
    "656--Elijah_Mitchell.csv.gzip",
    "658--Andre_Roberts.csv.gzip",
    "659--Kendrick_Bourne.csv.gzip",
    "660--Rashaad_Penny.csv.gzip",
    "668--Courtland_Sutton.csv.gzip",
    "669--Trevor_Lawrence.csv.gzip",
    "670--Mecole_Hardman.csv.gzip",
    "671--Travis_Kelce.csv.gzip",
    "674--Ja'Marr_Chase.csv.gzip",
    "675--Amon-Ra_St._Brown.csv.gzip",
    "676--Najee_Harris.csv.gzip",
    "677--Trey_Lance.csv.gzip",
    "680--Andy_Dalton.csv.gzip",
    "681--CeeDee_Lamb.csv.gzip",
    "682--Dak_Prescott.csv.gzip",
    "683--Drew_Lock.csv.gzip",
    "684--Amon-Ra_St._Brown.csv.gzip",
    "685--D'Andre_Swift.csv.gzip",
    "686--Allen_Lazard.csv.gzip",
    "687--Josiah_Deguara.csv.gzip",
    "689--Austin_Ekeler.csv.gzip",
    "690--Jared_Cook.csv.gzip",
    "691--Tyler_Higbee.csv.gzip",
    "692--Tua_Tagovailoa.csv.gzip",
    "693--Ihmir_Smith-Marsette.csv.gzip",
    "694--Hunter_Henry.csv.gzip",
    "695--Darius_Slayton.csv.gzip",
    "699--Scott_Miller.csv.gzip",
    "700--Julio_Jones.csv.gzip",
    "701--Antonio_Gibson.csv.gzip",
    "703--Russell_Gage.csv.gzip",
    "704--Stefon_Diggs.csv.gzip",
    "705--K.J._Osborn.csv.gzip",
    "706--Kirk_Cousins.csv.gzip",
    "707--Jakobi_Meyers.csv.gzip",
    "708--Tre'Quan_Smith.csv.gzip",
    "709--Quez_Watkins.csv.gzip",
    "710--Mike_Evans.csv.gzip",
    "711--Ryan_Tannehill.csv.gzip",
    "712--Cooper_Kupp.csv.gzip",
    "714--Rashaad_Penny.csv.gzip",
    "715--Russell_Wilson.csv.gzip",
    "724--A.J._Green.csv.gzip",
    "725--Devin_Singletary.csv.gzip",
    "726--Nick_Chubb.csv.gzip",
    "727--DK_Metcalf.csv.gzip",
    "728--Latavius_Murray.csv.gzip",
    "729--Melvin_Gordon.csv.gzip",
    "738--Jaylen_Waddle.csv.gzip",
    "740--Deebo_Samuel.csv.gzip",
    "741--Davis_Mills.csv.gzip",
    "742--Trevor_Lawrence.csv.gzip",
    "746--Dawson_Knox.csv.gzip",
    "747--Devin_Singletary.csv.gzip",
    "749--C.J._Uzomah.csv.gzip",
    "753--Matthew_Stafford.csv.gzip",
    "754--Cooper_Kupp.csv.gzip",
    "755--Mac_Jones.csv.gzip",
    "756--Ben_Roethlisberger.csv.gzip",
    "758--Brandon_Aiyuk.csv.gzip",
    "759--Elijah_Mitchell.csv.gzip",
    "760--Rob_Gronkowski.csv.gzip",
    "762--Joe_Burrow.csv.gzip",
    "763--Travis_Kelce.csv.gzip",
    "766--Deebo_Samuel.csv.gzip",
    "768--Mike_Evans.csv.gzip",
    "770--Josh_Allen.csv.gzip",
    "771--Patrick_Mahomes.csv.gzip",
    "772--Odell_Beckham.csv.gzip",
    "773--Jerick_McKinnon.csv.gzip",
    "774--Josh_Allen.csv.gzip",
    "775--Stefon_Diggs.csv.gzip",
    "776--Ja'Marr_Chase.csv.gzip",
    "777--Joe_Mixon.csv.gzip",
    "778--Aaron_Jones.csv.gzip",
    "779--Mecole_Hardman.csv.gzip",
    "780--Patrick_Mahomes.csv.gzip",
    "781--Travis_Kelce.csv.gzip",
    "783--George_Kittle.csv.gzip",
    "786--A.J._Brown.csv.gzip",
    "788--Tyreek_Hill.csv.gzip",
    "789--Cooper_Kupp.csv.gzip",
    "791--Tom_Brady.csv.gzip",
    "793--Gabe_Davis.csv.gzip",
    "798--Joe_Burrow.csv.gzip",
    "800--Tee_Higgins.csv.gzip",
    "801--Tyreek_Hill.csv.gzip",
    "802--Kendall_Blanton.csv.gzip",
    "803--Odell_Beckham.csv.gzip",
    "804--Deebo_Samuel.csv.gzip",
    "805--Jimmy_Garoppolo.csv.gzip",
    "806--Ja'Marr_Chase.csv.gzip",
    "807--Samaje_Perine.csv.gzip",
    "808--Patrick_Mahomes.csv.gzip",
    "809--Matthew_Stafford.csv.gzip",
    "812--Kyle_Pitts.csv.gzip",
    "813--Jaylen_Waddle.csv.gzip",
    "814--Mac_Jones.csv.gzip",
    "815--DeVonta_Smith.csv.gzip",
    "816--Najee_Harris.csv.gzip",
    "817--Ja'Marr_Chase.csv.gzip",
    "821--Joe_Mixon.csv.gzip",
    "823--Darrell_Henderson.csv.gzip",
    "824--Matthew_Stafford.csv.gzip",
    "825--Odell_Beckham.csv.gzip",
    "826--Ja'Marr_Chase.csv.gzip",
    "829--Cooper_Kupp.csv.gzip",
    "831--Javon_Walker.csv.gzip",
    "835--Marvin_Harrison.csv.gzip",
    "836--Ricky_Williams.csv.gzip",
    "837--Cris_Carter.csv.gzip",
    "838--Daunte_Culpepper.csv.gzip",
    "840--Thomas_Jones.csv.gzip",
    "842--Shaun_Alexander.csv.gzip",
    "845--Mark_Brunell.csv.gzip",
    "849--Matt_Hasselbeck.csv.gzip",
    "852--Laveranues_Coles.csv.gzip",
    "853--Byron_Leftwich.csv.gzip",
    "856--Roy_Williams.csv.gzip",
    "857--Donald_Driver.csv.gzip",
    "859--Brett_Favre.csv.gzip",
    "860--Ricky_Williams.csv.gzip",
    "863--Matt_Hasselbeck.csv.gzip",
    "866--Donald_Driver.csv.gzip",
    "867--Greg_Jennings.csv.gzip",
    "868--Marvin_Harrison.csv.gzip",
    "869--Jerricho_Cotchery.csv.gzip",
    "873--Ahman_Green.csv.gzip",
    "877--Cris_Carter.csv.gzip",
    "879--Brett_Favre.csv.gzip",
    "880--Antonio_Freeman.csv.gzip",
    "881--Brett_Favre.csv.gzip",
    "882--Courtland_Sutton.csv.gzip",
    "883--James_Robinson.csv.gzip",
    "884--Darren_Waller.csv.gzip",
    "885--Josh_Jacobs.csv.gzip",
    "886--Justin_Herbert.csv.gzip",
    "887--Kadarius_Toney.csv.gzip",
    "888--Elijah_Moore.csv.gzip",
    "889--Kenneth_Gainwell.csv.gzip",
    "890--Pat_Freiermuth.csv.gzip",
    "891--Derrick_Henry.csv.gzip",
    "892--Taylor_Heinicke.csv.gzip",
    "893--Cordarrelle_Patterson.csv.gzip",
    "894--CeeDee_Lamb.csv.gzip",
    "895--Aaron_Rodgers.csv.gzip",
    "896--Michael_Pittman.csv.gzip",
    "897--Dalvin_Cook.csv.gzip",
    "900--Jalen_Hurts.csv.gzip",
    "901--Joe_Mixon.csv.gzip",
    "902--Austin_Ekeler.csv.gzip",
    "903--Justin_Jefferson.csv.gzip",
    "908--Michael_Gallup.csv.gzip",
    "909--Carson_Wentz.csv.gzip",
    "910--Michael_Carter.csv.gzip",
    "911--Terry_McLaurin.csv.gzip",
    "912--Darnell_Mooney.csv.gzip",
    "913--Ezekiel_Elliott.csv.gzip",
    "917--D.J._Moore.csv.gzip",
    "918--Davante_Adams.csv.gzip",
    "919--Keenan_Allen.csv.gzip",
    "921--Mac_Jones.csv.gzip",
    "922--Marquise_Brown.csv.gzip",
    "923--Robert_Woods.csv.gzip",
    "925--Christian_McCaffrey.csv.gzip",
    "926--Elijah_Mitchell.csv.gzip",
    "927--Chase_Edmonds.csv.gzip",
    "928--Lamar_Jackson.csv.gzip",
    "929--Justin_Fields.csv.gzip",
    "930--Melvin_Gordon.csv.gzip",
    "931--Jonathan_Taylor.csv.gzip",
    "932--D.J._Chark.csv.gzip",
    "933--Adam_Thielen.csv.gzip",
    "934--Jalen_Reagor.csv.gzip",
    "936--DeVonta_Smith.csv.gzip",
    "937--Chase_Claypool.csv.gzip",
    "938--Chris_Godwin.csv.gzip",
    "939--Mark_Andrews.csv.gzip",
    "940--Amari_Cooper.csv.gzip",
    "941--Leonard_Fournette.csv.gzip",
    "946--Donovan_Peoples-Jones.csv.gzip",
    "947--T.J._Hockenson.csv.gzip",
    "948--Tyler_Lockett.csv.gzip",
    "950--Jamal_Agnew.csv.gzip",
    "966--Rondale_Moore.csv.gzip",
    "967--David_Montgomery.csv.gzip",
    "968--Jordan_Love.csv.gzip",
    "969--Trevor_Lawrence.csv.gzip",
    "970--Mike_Gesicki.csv.gzip",
    "971--Kendrick_Bourne.csv.gzip",
    "972--Jameis_Winston.csv.gzip",
    "973--JuJu_Smith-Schuster.csv.gzip",
    "974--Derek_Carr.csv.gzip",
    "975--Zach_Wilson.csv.gzip",
    "977--Nick_Chubb.csv.gzip",
    "978--Mike_Williams.csv.gzip",
    "979--Derrick_Henry.csv.gzip",
    "985--Rhamondre_Stevenson.csv.gzip",
    "986--Daniel_Jones.csv.gzip",
    "989--Jonathan_Taylor.csv.gzip",
    "991--Javonte_Williams.csv.gzip",
    "992--Tom_Brady.csv.gzip",
    "993--Larry_Fitzgerald.csv.gzip",
    "994--Chad_Johnson.csv.gzip",
    "995--Michael_Thomas.csv.gzip",
    "997--Keyshawn_Johnson.csv.gzip",
    "998--Steve_Smith.csv.gzip",
    "999--Davante_Adams.csv.gzip",
]


skewed_feats = [
    "Price",
    "Rarity",
    "Resell_Number",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "sacks",
    "sack_fumbles_lost",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_fumbles_lost",
    "receiving_tds",
    "receiving_fumbles_lost",
    "completions_season",
    "attempts_season",
    "passing_yards_season",
    "passing_tds_season",
    "interceptions_season",
    "sacks_season",
    "sack_fumbles_lost_season",
    "passing_epa_season",
    "carries_season",
    "rushing_yards_season",
    "rushing_tds_season",
    "rushing_fumbles_lost_season",
    "receiving_tds_season",
    "receiving_fumbles_lost_season",
    "receiving_epa_season",
    "completions_career",
    "attempts_career",
    "passing_yards_career",
    "passing_tds_career",
    "interceptions_career",
    "sacks_career",
    "sack_fumbles_lost_career",
    "passing_epa_career",
    "carries_career",
    "rushing_yards_career",
    "rushing_tds_career",
    "rushing_fumbles_lost_career",
    "rushing_epa_career",
    "receiving_tds_career",
    "receiving_fumbles_lost_career",
    "receiving_epa_career",
]


@st.cache(ttl=3600 * 24, allow_output_mutation=True)
def load_allday_data(cols=None):
    if cols is not None:
        df = pd.read_csv("data/current_allday_data.csv.gz", usecols=cols)
    else:
        df = pd.read_csv("data/current_allday_data.csv.gz")
    datecols = ["Datetime", "Date"]
    df[datecols] = df[datecols].apply(pd.to_datetime)
    return df


@st.cache(ttl=3600 * 24, allow_output_mutation=True)
def load_stats_data(years=None, subset=False):
    # NOTE: now returning less data

    weekly_df = pd.read_csv("data/weekly_data.csv")
    season_df = pd.read_csv("data/season_data.csv")
    roster_df = pd.read_csv("data/roster_data.csv")
    # team_df = pd.read_csv("data/team_desc.csv")
    season_df = season_df.merge(
        roster_df[
            ["player_id", "player_name", "position", "team", "headshot_url", "season"]
        ],
        on=["player_id", "season"],
    ).rename(columns={"player_name": "player_display_name"})
    if subset:
        season_df = season_df[stats_subset]

    weekly_df["team"] = weekly_df["recent_team"]
    if subset:
        weekly_df = weekly_df[
            [
                "player_id",
                "player_display_name",
                "position",
                "team",
                "headshot_url",
                "season",
                "week",
                "fantasy_points_ppr",
                "passing_tds",
                "passing_yards",
                "receiving_tds",
                "receiving_yards",
                "rushing_tds",
                "rushing_yards",
            ]
        ]

    if years is None:
        return weekly_df, season_df  # , roster_df, team_df
    elif type(years) == int:
        return (
            weekly_df[weekly_df.season == years],
            season_df[season_df.season == years],
            # roster_df[roster_df.season == years],
            # team_df,
        )
    else:
        return (
            weekly_df[weekly_df.season.isin(years)],
            season_df[season_df.season.isin(years)],
            # roster_df[roster_df.season.isin(years)],
            # team_df,
        )


@st.cache(ttl=3600 * 24)
def convert_df(df):
    """From: https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-pandas-dataframe-csv"""
    return df.to_csv().encode("utf-8")


def get_subset(df, col, val, n=n_players):
    return (
        df[df[col] == val]
        .sort_values("mean", ascending=False)
        .reset_index(drop=True)
        .iloc[:n]
    )


def get_position_group(x):
    if x in offense:
        return "Offense"
    if x in defense:
        return "Defense"
    if x in team_pos:
        return "Team"


@st.cache(ttl=3600 * 24 * 7)
def load_headshot(headshot_url):
    basewidth = 200
    try:
        image = Image.open(urlopen(headshot_url))
        wpercent = basewidth / float(image.size[0])
        hsize = int((float(image.size[1]) * float(wpercent)))
        image = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    except:
        image = Image.new("RGB", (basewidth, 142))
        img = ImageDraw.Draw(image)
        img.rectangle([(0, 0), (200, 200)], fill="#6c706d")
    return image


def alt_mean_price(
    df, col, color_col="Position", y_title="Mean Price ($)", y_labels=True
):
    chart = (
        (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(col, title=col.replace("_", " "), sort="-y"),
                y=alt.Y("mean", title=y_title, axis=alt.Axis(labels=y_labels)),
                tooltip=[
                    alt.Tooltip(col, title=col.replace("_", " ")),
                    alt.Tooltip(f"{color_col}:N", title=color_col.replace("_", " ")),
                    alt.Tooltip("mean", title="Mean Price ($)", format=".2f"),
                    alt.Tooltip("count", title="Total Sales", format=","),
                ],
                color=alt.Color(
                    f"{color_col}:N",
                    sort=alt.EncodingSortField(color_col, op="max", order="ascending"),
                    scale=alt.Scale(
                        scheme="paired",
                    ),
                ),
            )
        )
        .interactive()
        .properties(height=600)
    )
    return chart


def get_metrics(
    df,
    cols,
    metric,
    positions,
    short_form,
    pos_column="Position",
    agg_column="Price",
    summary=False,
):
    ntests = 1000  # approximate, for play types * positions * metrics * dates
    alpha = 0.05 / ntests  # Bonferroni correction for number tests
    for i, x in enumerate(positions):
        if x == "All":
            pos_data = df
        else:
            pos_data = df[df[pos_column] == x]

        if type(metric) == str:
            pos_metric = pos_data[pos_data[metric] == True]
            pos_no_metric = pos_data[pos_data[metric] == False]
        else:
            pos_metric = pos_data[pos_data[metric[0]] == True]
            pos_no_metric = pos_data[pos_data[metric[1]] == False]

        if summary:
            cols.metric(
                f"{short_form} Percentage: {x}", f"{len(pos_metric)/len(pos_data):.2%}"
            )
        else:
            pos_metric_agg = pos_metric[agg_column].values
            pos_no_metric_agg = pos_no_metric[agg_column].values

            pval = ttest_ind(pos_metric_agg, pos_no_metric_agg, equal_var=False).pvalue
            metric_mean = pos_metric_agg.mean()
            no_metric_mean = pos_no_metric_agg.mean()

            comp = (
                f"${metric_mean:,.2f} vs ${no_metric_mean:,.2f}"
                if agg_column == "Price"
                else f"{metric_mean:,.2f} vs {no_metric_mean:,.2f}"
            )

            if pd.isna(pval):
                sig = ""
                if pd.isna(metric_mean) and pd.isna(no_metric_mean):
                    comp = ""
                elif pd.isna(metric_mean):
                    comp = (
                        f"No {short_form}: ${no_metric_mean:,.2f}"
                        if agg_column == "Price"
                        else f"No {short_form}: {no_metric_mean:,.2f}"
                    )
                elif pd.isna(no_metric_mean):
                    comp = (
                        f"{short_form}: ${metric_mean:,.2f}"
                        if agg_column == "Price"
                        else f"{short_form}: {metric_mean:,.2f}"
                    )
            elif pval < alpha:
                if metric_mean > no_metric_mean:
                    sig = f"+ {short_form} HIGHER 📈"
                else:
                    sig = f"- {short_form} LOWER 📉"
            else:
                # sig = "- No Significant Difference"
                sig = ""

            if type(metric) != str:
                if len(pos_no_metric) == 0:
                    percentage = f"(No {short_form})"
                else:
                    percentage = f"({len(pos_metric)/len(pos_no_metric):,.2f} BG: Desc)"
            else:
                if len(pos_data) == 0:
                    percentage = f"(No {short_form})"
                else:
                    percentage = f"({len(pos_metric)/len(pos_data):.2%} {short_form})"

            cols[i % len(cols)].metric(
                f"Position: {x} {percentage}",
                comp,
                sig,
            )


def get_ttests(
    df,
    metric,
    positions,
    short_form,
    pos_column="Position",
    agg_column="Price",
):
    ntests = 1000  # approximate, for play types * positions * metrics * dates
    alpha = 0.05 / ntests  # Bonferroni correction for number tests
    vals = []
    for x in positions:
        if x == "All":
            pos_data = df
        else:
            pos_data = df[df[pos_column] == x]

        if type(metric) == str:
            pos_metric = pos_data[pos_data[metric] == True]
            pos_no_metric = pos_data[pos_data[metric] == False]
        else:
            pos_metric = pos_data[pos_data[metric[0]] == True]
            pos_no_metric = pos_data[pos_data[metric[1]] == False]

        pos_metric_agg = pos_metric[agg_column].values
        pos_no_metric_agg = pos_no_metric[agg_column].values

        pval = ttest_ind(pos_metric_agg, pos_no_metric_agg, equal_var=False).pvalue
        metric_mean = pos_metric_agg.mean()
        no_metric_mean = pos_no_metric_agg.mean()

        comp = (
            f"${metric_mean:,.2f} vs ${no_metric_mean:,.2f}"
            if agg_column == "Price"
            else f"{metric_mean:,.2f} vs {no_metric_mean:,.2f}"
        )

        if pd.isna(pval):
            sig = ""
            if pd.isna(metric_mean) and pd.isna(no_metric_mean):
                comp = ""
            elif pd.isna(metric_mean):
                comp = (
                    f"No {short_form}: ${no_metric_mean:,.2f}"
                    if agg_column == "Price"
                    else f"No {short_form}: {no_metric_mean:,.2f}"
                )
            elif pd.isna(no_metric_mean):
                comp = (
                    f"{short_form}: ${metric_mean:,.2f}"
                    if agg_column == "Price"
                    else f"{short_form}: {metric_mean:,.2f}"
                )
        elif pval < alpha:
            if metric_mean > no_metric_mean:
                sig = f"+ {short_form} HIGHER 📈"
            else:
                sig = f"- {short_form} LOWER 📉"
        else:
            # sig = "- No Significant Difference"
            sig = ""

        if type(metric) != str:
            if len(pos_no_metric) == 0:
                percentage = f"(No {short_form})"
            else:
                percentage = f"({len(pos_metric)/len(pos_no_metric):,.2f} BG: Desc)"
        else:
            if len(pos_data) == 0:
                percentage = f"(No {short_form})"
            else:
                percentage = f"({len(pos_metric)/len(pos_data):.2%} {short_form})"

        label = f"Position: {x} {percentage}"
        vals.append((label, comp, sig))

    return vals


@st.cache(ttl=3600 * 24)
def load_score_data(date_range, how_scores, play_type):
    date_str = date_range.replace(" ", "_")
    df = pd.read_csv(f"data/cache/{date_str}--grouped.csv")
    df["Scored Touchdown?"] = df[how_scores]
    if play_type != "All":
        df = df[df.Play_Type == play_type]
    return df


@st.cache(ttl=3600 * 24)
def load_ttest(
    date_range,
    play_type,
    how_scores,
    agg_metric,
    position_type,
    metric,
    short_form,
):
    with open("data/cache/score_ttest_results.json") as f:
        data = json.load(f)
    key = (
        f"{date_range}--{play_type}--{how_scores}--{agg_metric}--{position_type}--{metric}--{short_form}".replace(
            " ", "_"
        )
        .replace(")", "")
        .replace("(", "")
    )
    return data[key]


@st.cache(ttl=3600 * 24)
def load_player_data(date_range, agg_metric):
    date_str = date_range.replace(" ", "_")
    df = pd.read_csv(f"data/cache/player-{date_str}-{agg_metric}--grouped.csv")
    return df


@st.cache(ttl=3600 * 24)
def load_play_v_player_data(date_range):
    date_str = date_range.replace(" ", "_")
    play_type_price_data = pd.read_csv(
        f"data/cache/play_v_player-play_type-{date_str}--grouped.csv"
    )
    play_type_tier_price_data = pd.read_csv(
        f"data/cache/play_v_player-play_type_tier-{date_str}--grouped.csv"
    )
    player_tier_price_data = pd.read_csv(
        f"data/cache/play_v_player-player_tier-{date_str}--grouped.csv"
    )
    topN_player_data = pd.read_csv(
        f"data/cache/play_v_player-topN_player-{date_str}--grouped.csv"
    )
    return (
        play_type_price_data,
        play_type_tier_price_data,
        player_tier_price_data,
        topN_player_data,
    )


@st.experimental_memo(ttl=3600 * 24)
def load_challenge_data():
    challenges = pd.read_csv("data/NFLALLDAY_Challenges-Challenges.csv")
    datecols = ["Start Time (EDT)", "End Time (EDT)"]
    challenges[datecols] = challenges[datecols].apply(pd.to_datetime)
    challenges["Index"] = challenges.index
    challenges["Moments Needed"] = challenges["What You'll Need"].apply(
        lambda x: ", ".join(json.loads(x))
    )
    return challenges


@st.experimental_memo(ttl=3600 * 24)
def load_challenge_player_data(challenge):
    challenge_df = pd.read_csv(f"data/challenges/{challenge}.csv.gz")

    challenge_data_points = len(challenge_df)
    challenge_chart_df = challenge_df.astype(str)
    if challenge_data_points > 10000:
        frac = 10000 / challenge_data_points
        weights = 1 / challenge_chart_df.groupby("Display")["Display"].transform(
            "count"
        )
        challenge_chart_df = challenge_chart_df.sample(
            frac=frac,
            weights=weights,
            random_state=1234,
        )

    return challenge_df, challenge_chart_df


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def alt_challenge_chart(challenge_chart_df, time_df, shape_col):
    chart = (
        alt.Chart(challenge_chart_df)
        .mark_point(size=50, filled=True)
        .encode(
            x=alt.X("yearmonthdatehoursminutes(Datetime):T", title=None),
            y=alt.Y("Price:Q", scale=alt.Scale(type="log")),
            color=alt.Color(
                "Display",
                title="Player",
                scale=alt.Scale(
                    scheme="category20",
                ),
                sort=["Other"],
            ),
            tooltip=[
                alt.Tooltip("yearmonthdatehoursminutes(Datetime):T", title="Date"),
                "Player",
                alt.Tooltip("Display", title="Player Display"),
                "Moment_Tier",
                "marketplace_id",
                "Serial_Number",
                "Price:Q",
                "Wildcard",
            ],
            shape=alt.Shape(
                shape_col,
                scale=alt.Scale(
                    domain=[
                        "True",
                        "False",
                    ],
                    range=[
                        "circle",
                        "triangle",
                    ],
                ),
            ),
            href="site",
        )
        .interactive()
    )
    date_rules = (
        alt.Chart(
            time_df,
        )
        .mark_rule(strokeDash=[10, 4], opacity=0.7)
        .encode(
            x="yearmonthdatehoursminutes(Datetime):T",
            tooltip=[
                alt.Tooltip("yearmonthdatehoursminutes(Datetime):T", title="Date"),
                alt.Tooltip("Description"),
            ],
            color=alt.Color("Color:N", scale=None),
            strokeWidth=alt.value(3),
        )
    )
    combined = (chart + date_rules).properties(height=600, width=800)
    return combined


@st.experimental_memo(ttl=3600 * 24)
def get_challenge_summary(challenge_df, metric, summary=True):
    sub = challenge_df[challenge_df[metric]]

    if summary:
        sub_n = len(sub)
        sales_count = f"{sub_n}"
        median_price = f"${sub.Price.median():.2f}"
        floor_price = f"${sub.Price.min():.2f}"
        return (
            ("Total Sales Count", sales_count),
            ("Median Price", median_price),
            ("Floor Price", floor_price),
        )
    else:
        grouped_by_nft = sub.groupby("marketplace_id").Price
        sales_count_nft_median = f"{grouped_by_nft.count().median():.2f}"
        mean_price_nft_median = f"${grouped_by_nft.mean().median():.2f}"
        floor_price_nft_median = f"${grouped_by_nft.min().median():.2f}"

        return (
            ("Median Sales Count, per Moment", sales_count_nft_median),
            ("Median Price, per Moment", mean_price_nft_median),
            ("Median Floor Price, per Moment", floor_price_nft_median),
        )


def get_challenge_ttests(challenge_df, use_cache=False, update_cache=False):
    if use_cache and not update_cache:
        with open("data/challenge_ttests.json") as f:
            results = json.load(f)
            return results
    else:
        pre_game = challenge_df[challenge_df["pre_game_1d"]]
        during_game = challenge_df[challenge_df["during_game"]]
        post_game = challenge_df[challenge_df["post_game_1d"]]
        pre_challenge = challenge_df[challenge_df["pre_challenge_1d"]]
        during_challenge = challenge_df[challenge_df["during_challenge"]]
        post_challenge = challenge_df[challenge_df["post_challenge_1d"]]

        pre_game_group_price = pre_game.groupby("marketplace_id").Price.mean()
        during_game_group_price = during_game.groupby("marketplace_id").Price.mean()
        post_game_group_price = post_game.groupby("marketplace_id").Price.mean()
        pre_challenge_group_price = pre_challenge.groupby("marketplace_id").Price.mean()
        during_challenge_group_price = during_challenge.groupby(
            "marketplace_id"
        ).Price.mean()
        post_challenge_group_price = post_challenge.groupby(
            "marketplace_id"
        ).Price.mean()

        pre_game_group_count = pre_game.groupby("marketplace_id").Price.count()
        during_game_group_count = during_game.groupby("marketplace_id").Price.count()
        post_game_group_count = post_game.groupby("marketplace_id").Price.count()
        pre_challenge_group_count = pre_challenge.groupby(
            "marketplace_id"
        ).Price.count()
        during_challenge_group_count = during_challenge.groupby(
            "marketplace_id"
        ).Price.count()
        post_challenge_group_count = post_challenge.groupby(
            "marketplace_id"
        ).Price.count()

        pre_challenge_group_floor = pre_challenge.groupby("marketplace_id").Price.min()
        during_challenge_group_floor = during_challenge.groupby(
            "marketplace_id"
        ).Price.min()
        post_challenge_group_floor = post_challenge.groupby(
            "marketplace_id"
        ).Price.min()

        Before_Game_vs_During_Game_Price = pd.concat(
            [pre_game_group_price, during_game_group_price], axis=1
        )
        Before_Game_vs_During_Game_Count = pd.concat(
            [pre_game_group_count, during_game_group_count], axis=1
        )

        During_Game_vs_During_Challenge_Price = pd.concat(
            [during_game_group_price, during_challenge_group_price], axis=1
        )
        During_Game_vs_During_Challenge_Count = pd.concat(
            [during_game_group_count, during_challenge_group_count], axis=1
        )

        During_Challenge_vs_After_Challenge_Price = pd.concat(
            [during_challenge_group_price, post_challenge_group_price], axis=1
        )
        During_Challenge_vs_After_Challenge_Count = pd.concat(
            [during_challenge_group_count, post_challenge_group_count], axis=1
        )

        Before_Game_vs_After_Game_Price = pd.concat(
            [pre_game_group_price, post_game_group_price], axis=1
        )
        Before_Game_vs_After_Game_Count = pd.concat(
            [pre_game_group_count, post_game_group_count], axis=1
        )

        Before_Challenge_vs_After_Challenge_Price = pd.concat(
            [pre_challenge_group_price, post_challenge_group_price], axis=1
        )
        Before_Challenge_vs_After_Challenge_Count = pd.concat(
            [pre_challenge_group_count, post_challenge_group_count], axis=1
        )

        Before_Challenge_vs_During_Challenge_Count = pd.concat(
            [pre_challenge_group_count, during_challenge_group_count], axis=1
        )
        Before_Challenge_vs_During_Challenge_Price = pd.concat(
            [pre_challenge_group_price, during_challenge_group_price], axis=1
        )

        Before_Challenge_vs_After_Challenge_Floor = pd.concat(
            [pre_challenge_group_floor, post_challenge_group_floor], axis=1
        )
        Before_Challenge_vs_During_Challenge_Floor = pd.concat(
            [pre_challenge_group_floor, during_challenge_group_floor], axis=1
        )

        dfs = {
            "Before_Game_vs_During_Game_Price": Before_Game_vs_During_Game_Price,
            "Before_Game_vs_During_Game_Count": Before_Game_vs_During_Game_Count,
            "During_Game_vs_During_Challenge_Price": During_Game_vs_During_Challenge_Price,
            "During_Game_vs_During_Challenge_Count": During_Game_vs_During_Challenge_Count,
            "During_Challenge_vs_After_Challenge_Price": During_Challenge_vs_After_Challenge_Price,
            "During_Challenge_vs_After_Challenge_Count": During_Challenge_vs_After_Challenge_Count,
            "Before_Game_vs_After_Game_Price": Before_Game_vs_After_Game_Price,
            "Before_Game_vs_After_Game_Count": Before_Game_vs_After_Game_Count,
            "Before_Challenge_vs_After_Challenge_Price": Before_Challenge_vs_After_Challenge_Price,
            "Before_Challenge_vs_After_Challenge_Count": Before_Challenge_vs_After_Challenge_Count,
            "Before_Challenge_vs_During_Challenge_Count": Before_Challenge_vs_During_Challenge_Count,
            "Before_Challenge_vs_During_Challenge_Price": Before_Challenge_vs_During_Challenge_Price,
            "Before_Challenge_vs_After_Challenge_Floor": Before_Challenge_vs_After_Challenge_Floor,
            "Before_Challenge_vs_During_Challenge_Floor": Before_Challenge_vs_During_Challenge_Floor,
        }

        results = {}
        for k, v in dfs.items():
            t = ttest_rel(v.iloc[:, 0], v.iloc[:, 1], nan_policy="omit")
            results[k] = t.pvalue
        if update_cache:
            with open("data/challenge_ttests.json", "w") as f:
                json.dump(results, f)

    return results


def alt_challenge_game(
    df,
    xval,
    yval,
):
    base = alt.Chart(df)
    chart = (
        base.mark_point(filled=True, size=100)
        .encode(
            x=alt.X(
                "jitter:Q",
                title=None,
                axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
                scale=alt.Scale(),
            ),
            y=alt.Y(f"{yval}:Q", scale=alt.Scale(type="log", zero=False, nice=False)),
            tooltip=[
                "Player:N",
                "Team:N",
                "Position:N",
                "Moment_Tier:N",
                "Price (USD):Q",
                "Count:Q",
                "Floor Price (USD):Q",
            ],
            color=alt.Color(
                "Display:N",
                title="Player",
                scale=alt.Scale(
                    scheme="category20",
                ),
                sort=["Other"],
            ),
            shape=alt.Shape(
                "Moment_Tier",
                scale=alt.Scale(
                    domain=[
                        "True",
                        "False",
                    ],
                    range=[
                        "circle",
                        "triangle",
                    ],
                ),
            ),
            href="site",
        )
        .transform_calculate(
            # Generate Gaussian jitter with a Box-Muller transform
            jitter="sqrt(-2*log(random()))*cos(2*PI*random())"
        )
        .interactive()
        .properties(height=800, width=125)
    )
    box = base.mark_boxplot(color="#004D40", outliers=False, size=25).encode(
        y=alt.Y(f"{yval}:Q"),
    )
    combined_chart = (
        alt.layer(box, chart)
        .facet(
            column=alt.Column(
                f"{xval}:N",
                title=None,
                header=alt.Header(
                    labelAngle=-90,
                    titleOrient="top",
                    labelOrient="bottom",
                    labelAlign="right",
                    labelPadding=3,
                ),
                # sort=position_type_dict[position_type][1],
            ),
            title=f'{xval.title().replace("_", " ")}: {yval}',
        )
        .configure_facet(spacing=0)
    )
    return combined_chart


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_challenge_reward():
    return pd.read_csv("data/NFLALLDAY_Challenges-RewardBreakdown.csv")


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_pack_info(summary=True):
    pack_info = pd.read_csv("data/packs.csv")
    pack_info["Datetime"] = pd.to_datetime(pack_info["Datetime"]).dt.tz_convert(
        "US/Eastern"
    )
    if summary:
        pack_info = pack_info[
            ["Number", "Name", "Site", "Type", "Series", "Cost", "Supply"]
        ]
    return pack_info


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_pack():
    pack_df = pd.read_csv("data/pack_combined.csv.gz")
    pack_df["Datetime_Reveal"] = pd.to_datetime(
        pack_df["Datetime_Reveal"]
    ).dt.tz_localize("US/Eastern")
    pack_df["Datetime_Pack"] = pd.to_datetime(pack_df["Datetime_Pack"]).dt.tz_localize(
        "US/Eastern"
    )

    pack_df["Reveal_Lag"] = pack_df["Datetime_Reveal"] - pack_df["Datetime_Pack"]
    # pack_df["Reveal_Lag_Seconds"] = (
    #     pack_df["Datetime_Reveal"] - pack_df["Datetime_Pack"]
    # ).dt.total_seconds()
    grouped = (
        pack_df.groupby(
            [
                pd.Grouper(key="Datetime_Pack", axis=0, freq="min"),
                "Pack Type",
            ]
        )
        .agg(
            Sales_Count=("tx_id_Pack", "nunique"),
            Moments_In_Pack=("Moments_In_Pack", "mean"),
            Pack_Price=("Pack_Price", "mean"),
            Reveal_Time_Avg=("Datetime_Reveal", "mean"),
            Reveal_Time_Earliest=("Datetime_Reveal", "min"),
            Reveal_Time_Latest=("Datetime_Reveal", "max"),
            Reveal_Lag_Avg=("Reveal_Lag", "mean"),
            Reveal_Lag_Earliest=("Reveal_Lag", "min"),
            Reveal_Lag_Latest=("Reveal_Lag", "max"),
        )
        .reset_index()
    )
    for x in [
        "Reveal_Lag_Avg",
        "Reveal_Lag_Earliest",
        "Reveal_Lag_Latest",
    ]:
        grouped[f"{x}_Seconds"] = grouped[x].dt.total_seconds()
        grouped[x] = grouped[x].apply(lambda x: str(x.floor("s")))
    # pack_df["Reveal_Lag"] = pack_df["Reveal_Lag"].apply(lambda x: str(x.floor("s")))

    return pack_df, grouped


def alt_pack_sales(df):
    max_range = df.Sales_Count.max()
    pts = alt.selection(type="interval", encodings=["x"])
    scatter = (
        alt.Chart(title="Pack Sales")
        .mark_point(filled=True)
        .encode(
            x=alt.X("yearmonthdatehoursminutes(Datetime_Pack):T", title=None),
            y=alt.Y(
                "Sales_Count",
                title="Sales per Minute",
                scale=alt.Scale(domain=[0, max_range]),
            ),
            color="Pack Type",
            opacity=alt.condition(pts, alt.value(1), alt.value(0.05)),
            tooltip=[
                alt.Tooltip(
                    "yearmonthdatehoursminutes(Datetime_Pack):T",
                    title="Pack Purchase Datetime",
                ),
                "Pack Type",
                alt.Tooltip("Sales_Count", title="Sales per Minute"),
                alt.Tooltip("Pack_Price", title="Pack Price ($)", format=".0f"),
                alt.Tooltip("Moments_In_Pack", title="Moments In Pack", format=".0f"),
                alt.Tooltip(
                    "yearmonthdatehoursminutes(Reveal_Time_Avg):T",
                    title="Average Reveal Datetime",
                ),
                alt.Tooltip(
                    "yearmonthdatehoursminutes(Reveal_Time_Earliest):T",
                    title="Earliest Reveal Datetime",
                ),
                alt.Tooltip(
                    "yearmonthdatehoursminutes(Reveal_Time_Latest):T",
                    title="Latest Reveal Datetime",
                ),
                alt.Tooltip("Reveal_Lag_Avg", title="Average Reveal Lag"),
                alt.Tooltip("Reveal_Lag_Earliest", title="Earliest Reveal Lag"),
                alt.Tooltip("Reveal_Lag_Latest", title="Latest Reveal Lag"),
            ],
        )
        .properties(height=600, width=500)
        .add_selection(pts)
        # .interactive()
    )
    hist = (
        (
            alt.Chart(title="Count of Sales per Minute")
            .mark_bar()
            .encode(
                x=alt.X("count()", title="Count (Sales per Minute)"),
                y=alt.Y(
                    "mbin:Q",
                    title=None,
                    axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
                    scale=alt.Scale(domain=[0, max_range]),
                ),
                color="Pack Type",
                tooltip=[
                    alt.Tooltip("count()", title="Number of timepoints"),
                    alt.Tooltip("mbin:Q", title="Minimum Sales per Minute"),
                ],
            )
            .transform_filter(pts)
        ).properties(height=600, width=200)
        # .interactive()
    )
    chart = (
        alt.hconcat(scatter, hist, data=df)
        .transform_bin("mbin", field="Sales_Count", bin=alt.Bin(maxbins=50))
        .resolve_scale(y="shared")
        .properties(spacing=0)
    )
    return chart


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_player_pack():
    df = pd.read_csv(
        "data/current_allday_data_pack.csv.gz", usecols=player_pack_cols
    ).dropna(
        subset=[
            "Player",
            "Datetime_Reveal",
            "Datetime_Pack",
        ]
    )
    for x in [
        "Datetime",
        "Date",
        "Datetime_Reveal",
        "Datetime_Pack",
    ]:
        df[x] = pd.to_datetime(df[x]).dt.tz_localize("US/Eastern")
    df["Mint_Date"] = pd.to_datetime(df.Datetime_Pack.dt.date).dt.tz_localize(
        "US/Eastern"
    )
    return df


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_pack_cache(date_range):
    if type(date_range) == str:
        date_str = date_range.replace(" ", "_")
    else:
        date_str = date_range[0].split(" ")[0]
    df = pd.read_csv(f"data/cache/pack_data-{date_str}--grouped.csv.gz")
    return df


def get_pack_value(player_pack_data, proporiton_dict, pack_type, i):
    common = player_pack_data[player_pack_data.Moment_Tier == "COMMON"]
    rare = player_pack_data[player_pack_data.Moment_Tier == "RARE"]
    legendary = player_pack_data[player_pack_data.Moment_Tier == "LEGENDARY"]

    tier = np.random.choice(
        list(proporiton_dict.keys()), p=list(proporiton_dict.values())
    )
    if tier == "COMMON":
        players = common.sample(4)
    elif tier == "RARE":
        if pack_type == "Standard":
            players = pd.concat([common.sample(3), rare.sample(1)])
        elif pack_type == "Premium":
            players = pd.concat([common.sample(6), rare.sample(2)])
    elif tier == "LEGENDARY":
        if pack_type == "Standard":
            players = pd.concat([common.sample(3), legendary.sample(1)])
        elif pack_type == "Premium":
            players = pd.concat([common.sample(6), rare.sample(1), legendary.sample(1)])
    else:
        raise ValueError

    players = players[["Player", "site", "Moment_Tier", "Price"]]
    players["pack_type"] = pack_type
    players["pack_tier"] = tier
    players["idx"] = i
    return players


def mint_pack(sample_df, pack_type):
    if pack_type == "Standard":
        n = 10000
    elif pack_type == "Premium":
        n = 5000
    else:
        raise ValueError
    randn = np.random.randint(0, n)
    players = sample_df[(sample_df.idx == randn) & (sample_df.pack_type == pack_type)]
    return players


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_series2_mint1_grouped():
    return pd.read_csv("data/cache/series2_mint1_grouped.csv")


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_pack_samples():
    return pd.read_csv("data/cache/sample_packs.csv.gz")


def get_avg_pack_metrics(data):
    vals = {"Price": {}, "Count": {}}
    for tier in ["COMMON", "RARE", "LEGENDARY"]:
        df = data[data["Moment_Tier"] == tier]
        total_spent = (df.Price * df.Count).sum()
        total_sold = df.Count.sum()
        vals["Price"][tier] = total_spent / total_sold
        vals["Count"][tier] = df.Count.mean()
    return vals


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_player_mint(date):
    df = pd.read_csv(
        f"data/cache/player_mint-{date}--grouped.csv.gz",
    )
    for x in [
        "Datetime",
        "Date",
    ]:
        df[x] = pd.to_datetime(df[x]).dt.tz_convert("US/Eastern")

    return df


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_simulation():
    return pd.read_csv("data/simulation.csv")


# #TODO: figure this out, maybe as easter egg?
# def load_players():
#     return pd.read_csv("data/players.csv")


# def get_random_player():
#     r = requests.get("http://bflfootball.com/Player.aspx")
#     html = r.text
#     parsed_html = BeautifulSoup(html)
#     data = []

#     rows = parsed_html.find_all("tr")
#     for row in rows:
#         cols = row.find_all("td")
#         cols = [ele.text.strip() for ele in cols]
#         cols = [ele for ele in cols if ele]
#         data.append(cols)  # Get rid of empty values

#     player_dict = {}
#     for x in data:
#         if "Name:" in x:
#             player_dict["Name"] = x[-1]
#     return player_dict


# def get_player_name():
#     if np.random.randint(0, 2):
#         players = load_players()
#         player_name = players.iloc[np.random.randint(0, len(players))].values[0]
#         real = True
#     else:
#         player_info = get_random_player()
#         player_name = player_info["Name"]
#         real = False
#     return player_name, real


# ---
@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def load_by_playername(player):
    data = Path("data/ml")
    player_files = [pd.read_csv(x) for x in data.glob(f"*{player.replace(' ', '_')}*")]
    df = pd.concat(list(player_files))
    return df


@st.experimental_memo(ttl=3600, suppress_st_warning=True)
def load_by_marketplace_id(data_file):
    # data = Path("data/ml")
    # data_file = list(data.glob(f"{marketplace_id}*"))[0]
    df = pd.read_csv(Path("data/ml", data_file), compression="gzip")
    return df


@st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def get_xgb_data(train_dataset):
    categorical_features = [
        # "Position",
        # "Team",
        # "player_display_name",
        "Series",
        "Game Outcome",
        # "Play_Type_importance",
        "Set_Name_importance",
        # "Rarity"
    ]
    y_cols = ["Price"]
    info_cols = [
        "player_display_name",
        "player_id",
        "site",
        "Team",
        "Position",
        "season",
        "week",
        "marketplace_id",
        "scored_td_in_moment",
        "Play_Type",
        "Set_Name",
        "Play_Type_importance",
        # "Game Outcome",
        # "Sales_Count",
        # "Resell_Number",
    ]


    every_column_non_categorical = [
        col
        for col in train_dataset.columns
        if col not in categorical_features + info_cols
    ]
    numeric_feats = (
        train_dataset[every_column_non_categorical]
        .dtypes[train_dataset.dtypes != "object"]
        .index
    )
    numeric_no_price = numeric_feats.drop("Price")

    train_dataset[skewed_feats] = np.log1p(train_dataset[skewed_feats])
    train_dataset = train_dataset.replace(-1 * np.inf, 0)
    train_dataset = pd.get_dummies(train_dataset, columns=categorical_features)
    feature_cols = [
        col
        for col in train_dataset.columns
        if col not in y_cols + info_cols and col not in ["Game Outcome_Tie"]
    ]
    X = train_dataset[feature_cols]
    y = train_dataset[y_cols]
    scaler = RobustScaler()
    X[numeric_no_price] = scaler.fit_transform(X[numeric_no_price])

    for f in xgb_feature_names:
        if f not in X.columns:
            X[f] = 0

    return X[xgb_feature_names], y, scaler, numeric_no_price

# @st.experimental_memo(ttl=3600 * 24, suppress_st_warning=True)
def rescale_xgb(X,y, _scaler:RobustScaler, numeric_feats):
    unscaled = X.copy()
    unscaled[numeric_feats] = _scaler.inverse_transform(unscaled[numeric_feats])

    unscaled = pd.concat([unscaled, y], axis=1)
    unscaled[skewed_feats] = np.expm1(unscaled[skewed_feats])

    return unscaled