#!/usr/bin/env python3
import datetime
import json

import numpy as np
import pandas as pd

from utils import *


def get_score_data(score_data, date_range):
    # #TODO: clean up date ranges
    if date_range == "All Time":
        df = score_data
    elif date_range == "2022 Full Season":
        start = week_timings[1][0]
        df = score_data[score_data.Date >= start]
    elif date_range == "2022 Week 1":
        start, end = week_timings[1]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 2":
        start, end = week_timings[2]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 3":
        start, end = week_timings[3]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 4":
        start, end = week_timings[4]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 5":
        start, end = week_timings[5]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 6":
        start, end = week_timings[6]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 7":
        start, end = week_timings[7]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 8":
        start, end = week_timings[8]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]

    grouped = df.groupby(["marketplace_id"]).agg(agg_dict).reset_index()
    grouped["Week"] = grouped.Week.astype(str)
    grouped["site"] = grouped.marketplace_id.apply(
        lambda x: f"https://nflallday.com/listing/moment/{x}"
    )

    grouped_all = grouped.copy()
    grouped_all["Position"] = "All"
    grouped_all["Position Group"] = "All"
    grouped = pd.concat([grouped, grouped_all]).reset_index(drop=True)
    del grouped_all

    return df, grouped


def get_player_data(main_data, date_range, agg_metric):
    # #TODO: clean up date ranges
    if date_range == "All Time":
        df = main_data
    elif date_range == "2022 Full Season":
        start = week_timings[1][0]
        df = score_data[score_data.Date >= start]
    elif date_range == "2022 Week 1":
        start, end = week_timings[1]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 2":
        start, end = week_timings[2]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 3":
        start, end = week_timings[3]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 4":
        start, end = week_timings[4]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 5":
        start, end = week_timings[5]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 6":
        start, end = week_timings[6]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 7":
        start, end = week_timings[7]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]
    elif date_range == "2022 Week 8":
        start, end = week_timings[8]
        df = score_data[(score_data.Date >= start) & (score_data.Date < end)]

    grouped = (
        df.groupby(["Date", "Player", "Position", "Team"])
        .Price.agg(agg_metric)
        .reset_index()
    )
    video_url = (
        df.groupby(["Date", "Player", "Position", "Team"])
        .NFLALLDAY_ASSETS_URL.first()
        .reset_index()
    )
    grouped = grouped.merge(video_url, on=["Date", "Player", "Position", "Team"])
    grouped["Date"] = grouped.Date.dt.tz_localize("US/Pacific")

    return grouped


def get_play_v_player_data(main_data, date_range):
    # #TODO: clean up date ranges
    if date_range == "Since 2022 preseason":
        df = main_data[main_data.Date >= "2022-08-04"]
    elif date_range == "Since 2022 Week 1":
        start = week_timings[1][0]
        df = main_data[main_data.Date >= start]
    elif date_range == "Since 2022 Week 2":
        start = week_timings[2][0]
        df = main_data[main_data.Date >= start]
    elif date_range == "Since 2022 Week 3":
        start = week_timings[3][0]
        df = main_data[main_data.Date >= start]
    elif date_range == "Since 2022 Week 4":
        start = week_timings[4][0]
        df = main_data[main_data.Date >= start]
    elif date_range == "Since 2022 Week 5":
        start = week_timings[5][0]
        df = main_data[main_data.Date >= start]
    elif date_range == "Since 2022 Week 6":
        start = week_timings[6][0]
        df = main_data[main_data.Date >= start]
    elif date_range == "Since 2022 Week 7":
        start = week_timings[7][0]
        df = main_data[main_data.Date >= start]
    elif date_range == "Since 2022 Week 8":
        start = week_timings[8][0]
        df = main_data[main_data.Date >= start]
    else:
        df = main_data

    play_type_price_data = (
        df.groupby(
            [
                "Play_Type",
            ]
        )["Price"]
        .agg(["mean", "count"])
        .reset_index()
    )
    play_type_price_data["Position"] = "N/A"
    play_type_tier_price_data = (
        df.groupby(
            [
                "Play_Type",
                "Moment_Tier",
            ]
        )["Price"]
        .agg(["mean", "count"])
        .reset_index()
    )
    play_type_tier_price_data["Position"] = "N/A"

    player_price_data = (
        df.groupby(["Player", "Position"])["Price"].agg(["mean", "count"]).reset_index()
    )
    player_tier_price_data = (
        df.groupby(["Player", "Moment_Tier", "Position"])["Price"]
        .agg(["mean", "count"])
        .reset_index()
    )
    topN_player_data = (
        player_price_data.sort_values("mean", ascending=False)
        .reset_index(drop=True)
        .iloc[:n_players]
    )

    return (
        play_type_price_data,
        play_type_tier_price_data,
        player_tier_price_data,
        topN_player_data,
    )


def get_pack_data(grouped_pack, date_range_choice, date_range):
    if date_range_choice == "By Date Range":
        if date_range == "Since 2022 preseason":
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= "2022-08-04"]
        elif date_range == "Since 2022 Week 1":
            start = week_timings[1][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        elif date_range == "Since 2022 Week 2":
            start = week_timings[2][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        elif date_range == "Since 2022 Week 3":
            start = week_timings[3][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        elif date_range == "Since 2022 Week 4":
            start = week_timings[4][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        elif date_range == "Since 2022 Week 5":
            start = week_timings[5][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        elif date_range == "Since 2022 Week 6":
            start = week_timings[6][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        elif date_range == "Since 2022 Week 7":
            start = week_timings[7][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        elif date_range == "Since 2022 Week 8":
            start = week_timings[8][0]
            df = grouped_pack[grouped_pack["Datetime_Pack"] >= start]
        else:
            df = grouped_pack
    elif date_range_choice == "By Selected Drop":
        start, end = date_range
        df = grouped_pack[
            (grouped_pack["Datetime_Pack"] >= start)
            & (grouped_pack["Datetime_Pack"] <= end)
        ]
    return df


if __name__ == "__main__":
    save_full = False
    main_data = load_allday_data(cols_to_keep)
    main_data["Position Group"] = main_data.Position.apply(get_position_group)

    score_data = main_data[main_data.Play_Type.isin(score_columns)].reset_index(
        drop=True
    )
    score_data = score_data.rename(columns=td_mapping)
    score_ttest_results = {}
    for date_range in main_date_ranges:
        date_str = date_range.replace(" ", "_")
        df, grouped = get_score_data(score_data, date_range)
        if save_full:
            df.to_csv(
                f"data/cache/score-{date_str}--df.csv.gz",
                compression="gzip",
                index=False,
            )
        grouped.to_csv(
            f"data/cache/{date_str}--grouped.csv",
            index=False,
        )
        for play_type in ["All"] + score_columns:
            for how_scores in td_mapping.values():
                # substr = (
                #     f"{date_range}--{play_type}--{how_scores}".replace(" ", "_")
                #     .replace(")", "")
                #     .replace("(", "")
                # )
                # print(f"#@# Working on: {substr}")

                df["Scored Touchdown?"] = df[how_scores]
                if play_type != "All":
                    df = df[df.Play_Type == play_type]
                for agg_metric in ["Average Sales Price ($)", "Sales Count"]:
                    for position_type in position_type_dict.keys():
                        if position_type == "By Position":
                            pos_subset = [
                                x
                                for x in positions
                                if x in ["All"] + df.Position.unique().tolist()
                            ]
                            pos_column = position_type_dict[position_type][0]
                        else:
                            pos_subset = position_type_dict[position_type][1]
                            pos_column = position_type_dict[position_type][0]
                        for metric, short_form in [
                            (
                                how_scores,
                                "TDs",
                            ),
                            (
                                "won_game",
                                "Winners",
                            ),
                            (
                                [
                                    "Best Guess (Moment TD)",
                                    "Description only (Moment TD)",
                                ],
                                "Best Guess Moment",
                            ),
                            (
                                [
                                    "Best Guess: (In-game TD)",
                                    "Description only (Moment TD)",
                                ],
                                "Best Guess Game",
                            ),
                        ]:
                            if agg_metric == "Sales Count" and type(metric) == str:
                                ttest_df = grouped
                                agg_column = "tx_id"
                            else:
                                ttest_df = df
                                agg_column = "Price"
                            ttest_res = get_ttests(
                                ttest_df,
                                metric,
                                pos_subset,
                                short_form,
                                pos_column,
                                agg_column,
                            )
                            substr = (
                                f"{date_range}--{play_type}--{how_scores}--{agg_metric}--{position_type}--{metric}--{short_form}".replace(
                                    " ", "_"
                                )
                                .replace(")", "")
                                .replace("(", "")
                            )
                            print(f"#@# Working on: {substr}")
                            score_ttest_results[substr] = ttest_res

    with open("data/cache/score_ttest_results.json", "w") as f:
        json.dump(score_ttest_results, f)

    for date_range in main_date_ranges:
        date_str = date_range.replace(" ", "_")
        for agg_metric in ["median", "mean", "count"]:
            grouped = get_player_data(main_data, date_range, agg_metric)
            grouped.to_csv(
                f"data/cache/player-{date_str}-{agg_metric}--grouped.csv",
                index=False,
            )

    del score_data
    _, grouped_pack = load_pack()
    for date_range in play_v_player_date_ranges:
        date_str = date_range.replace(" ", "_")
        (
            play_type_price_data,
            play_type_tier_price_data,
            player_tier_price_data,
            topN_player_data,
        ) = get_play_v_player_data(main_data, date_range)
        play_type_price_data.to_csv(
            f"data/cache/play_v_player-play_type-{date_str}--grouped.csv",
            index=False,
        )
        play_type_tier_price_data.to_csv(
            f"data/cache/play_v_player-play_type_tier-{date_str}--grouped.csv",
            index=False,
        )
        player_tier_price_data.to_csv(
            f"data/cache/play_v_player-player_tier-{date_str}--grouped.csv",
            index=False,
        )
        topN_player_data.to_csv(
            f"data/cache/play_v_player-topN_player-{date_str}--grouped.csv",
            index=False,
        )

        pack_df = get_pack_data(grouped_pack, "By Date Range", date_range)
        pack_df.to_csv(
            f"data/cache/pack_data-{date_str}--grouped.csv.gz",
            index=False,
            compression="gzip",
        )
    del main_data
    for date_range in pack_date_ranges:
        date_str = date_range[0].split(" ")[0]
        pack_df = get_pack_data(grouped_pack, "By Selected Drop", date_range)
        pack_df.to_csv(
            f"data/cache/pack_data-{date_str}--grouped.csv.gz",
            index=False,
            compression="gzip",
        )

    del grouped_pack
    player_pack_data = load_player_pack()
    series2_mint1 = player_pack_data[
        (player_pack_data.Mint_Date >= "2022-09-27 00:00:00-04:00")
        & (player_pack_data.Mint_Date < "2022-10-08 00:00:00-04:00")
    ]
    series2_mint1_grouped = (
        series2_mint1.groupby(
            [
                "marketplace_id",
                "Player",
                "Team",
                "Position",
                "Moment_Tier",
                "Moment_Date",
                "Total_Circulation",
                "site",
                # "Pack Type",  # not doing for now
            ]
        )
        .agg(
            Price=("Price", "mean"),
            Max_Price=("Price", "max"),
            Min_Price=("Price", "min"),
            Count=("tx_id", "count"),
        )
        .reset_index()
    )
    series2_mint1_grouped.to_csv(
        f"data/cache/series2_mint1_grouped.csv",
        index=False,
    )

    samples = []
    for i in range(10000):
        samp = get_pack_value(
            series2_mint1, series2_mint1_standard_proportions, "Standard", i
        )
        samples.append(samp)
    for i in range(5000):
        samp = get_pack_value(
            series2_mint1, series2_mint1_premium_proportions, "Premium", i
        )
        samples.append(samp)

    roster_df = pd.read_csv("data/roster_data.csv")
    roster_df = roster_df[roster_df.season == 2022]
    sample_df = pd.concat(samples)
    sample_df = sample_df.merge(
        roster_df[["player_name", "headshot_url"]],
        left_on="Player",
        right_on="player_name",
        how="left",
    ).drop(columns="player_name")
    sample_df.to_csv(
        f"data/cache/sample_packs.csv.gz",
        index=False,
        compression="gzip",
    )

    for i, x in enumerate(pack_date_ranges):
        start = pd.to_datetime(x[0]).tz_localize("US/Eastern")
        end = pd.to_datetime(x[1]).tz_localize("US/Eastern")
        try:
            next_start = pd.to_datetime(pack_date_ranges[i + 1][0]).tz_localize(
                "US/Eastern"
            )
            next_end = pd.to_datetime(pack_date_ranges[i + 1][1]).tz_localize(
                "US/Eastern"
            )
        except IndexError:
            next_start = pd.to_datetime(datetime.datetime.now()).tz_localize(
                "US/Eastern"
            )
            next_end = pd.to_datetime(datetime.datetime.now()).tz_localize("US/Eastern")
        df = player_pack_data[
            (player_pack_data.Datetime >= start)
            & (player_pack_data.Datetime < end + pd.Timedelta("1d"))
        ]
        df = df[
            [
                "Datetime",
                "Date",
                "Price",
                "Player",
                "Team",
                "Position",
                "Play_Type",
                "Moment_Date",
                "Moment_Tier",
                "Series",
                "Set_Name",
                "marketplace_id",
                "site",
                "Datetime_Reveal",
                "Moments_In_Pack",
                "Datetime_Pack",
                "Pack_Price",
                "Pack_Buyer",
                "Pack Type",
                "Mint_Date",
            ]
        ]
        minted = df[(df.Mint_Date.dt.date >= start.date()) & (df.Mint_Date.dt.date < next_start.date())]
        df["minted_moment"] = False
        df["minted_player"] = False
        df['player_not_moment'] = False
        df['player_moment'] = False
        df.loc[df.Player.isin(minted.Player.unique()), "minted_player"] = True
        df.loc[df.marketplace_id.isin(minted.marketplace_id.unique()), "minted_moment"] = True
        df.loc[(df.minted_player) & ~(df.minted_moment), "player_not_moment"] = True
        df.loc[(df.minted_player) & (df.minted_moment), "player_moment"] = True
        print(f"#@# {x}: {len(df)}")

        date_str = start.date()
        df.to_csv(
            f"data/cache/player_mint-{date_str}--grouped.csv.gz",
            index=False,
            compression="gzip",
        )
