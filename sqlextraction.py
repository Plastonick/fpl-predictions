import numpy as np
import psycopg2
from itertools import chain


def build_training_data(form_size=5) -> tuple[np.ndarray, np.ndarray]:
    records_by_player = build_records_per_player()

    X = []
    y = []
    for player_id in records_by_player:
        records = records_by_player[player_id]

        # look_ahead represents how many fixtures ahead the game we're predicting is
        for look_ahead in range(10):
            for i in range(len(records) - form_size - look_ahead):
                # include the number of games look_ahead we're including in the context to let ML know that
                X.append(list(chain.from_iterable(records[i - look_ahead: i + form_size - look_ahead])) + [look_ahead])
                y.append([records[i + form_size][0]])

    return np.asarray(X), np.asarray(y)


def build_records_per_player():
    cursor = get_cursor()
    sql = f"""
SELECT 
       pp.player_id,
       pp.kickoff_time,
       pp.total_points,
       CASE WHEN pp.was_home THEN 1 ELSE 0 END,
       COALESCE(f.team_h_difficulty, 0),
       COALESCE(f.team_a_difficulty, 0),
       pp.minutes,
       p.position_id
FROM player_performances pp
         INNER JOIN fixtures f ON pp.fixture_id = f.fixture_id
         INNER JOIN player_season_positions psp ON (f.season_id = psp.season_id AND pp.player_id = psp.player_id)
         INNER JOIN positions p ON psp.position_id = p.position_id
ORDER BY pp.player_id, pp.kickoff_time DESC;
    """
    cursor.execute(sql)
    records = cursor.fetchall()
    records_by_player = {}
    for (
            player_id,
            kickoff_time,
            total_points,
            was_home,
            home_difficulty,
            away_difficulty,
            minutes,
            position,
    ) in records:
        if player_id not in records_by_player:
            records_by_player[player_id] = []

        records_by_player[player_id].append(
            [
                total_points,
                was_home,
                home_difficulty,
                away_difficulty,
                minutes,
                position,
            ]
        )

    return records_by_player


def get_cursor():
    connection = psycopg2.connect(
        user="fantasy-user",
        password="fantasy-pwd",
        host="192.168.1.106",
        port="5432",
        database="fantasy-db",
    )
    cursor = connection.cursor()

    return cursor


def get_context(season: int):
    players = get_player_and_team_ids_for_season(2021)


    # get all un-played fixtures
    # generate context for all un-played fixtures and for all players
    # return to allow predictions

    return 1


def get_player_and_team_ids_for_season(season: int) -> list[tuple[int, int]]:
    sql = f"""
SELECT DISTINCT pp.player_id,
                ( SELECT team_id
                  FROM player_performances pp2
                  WHERE pp2.player_id = pp.player_id
                  ORDER BY pp2.kickoff_time DESC
                  LIMIT 1 ) AS player_team_id
FROM player_performances pp
         INNER JOIN fixtures f ON pp.fixture_id = f.fixture_id
         INNER JOIN seasons s ON f.season_id = s.season_id
WHERE s.start_year = {season};
"""

    cursor = get_cursor()
    cursor.execute(sql)

    return cursor.fetchall()


def get_un_played_fixtures(season: int) -> list[tuple[int, int]]:
    sql = f"""
SELECT home_team_id,
    away_team_id,
    team_h_difficulty,
    team_a_difficulty,
    kickoff_time,
FROM fixtures f
         LEFT JOIN player_performances pp ON f.fixture_id = pp.fixture_id
         INNER JOIN seasons s ON f.season_id = s.season_id
WHERE s.start_year = {season}
    AND pp.player_performance_id IS NULL;
"""

    cursor = get_cursor()
    cursor.execute(sql)

    fixtures_data = {}
    for record in cursor.fetchall():


    return cursor.fetchall()


# players = get_player_and_team_ids_for_season(2021)
#
# print(players)

X, y = build_training_data(form_size=5)

print(X)

a = 1