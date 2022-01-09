import numpy as np
import psycopg2
from itertools import chain


def build_training_data(lookback=5) -> tuple[np.ndarray, np.ndarray]:
    connection = psycopg2.connect(
        user="fantasy-user",
        password="fantasy-pwd",
        host="192.168.1.106",
        port="5432",
        database="fantasy-db"
    )

    cursor = connection.cursor()

    sql = 'select player_id, fixture_id from player_performances limit 10'

    cursor.execute(sql)
    records = cursor.fetchall()

    for record in records:
        context = get_context(record[0], record[1])
        a = 1

    return X, y


def get_context(player_id, fixture_id):
    connection = psycopg2.connect(
        user="fantasy-user",
        password="fantasy-pwd",
        host="192.168.1.106",
        port="5432",
        database="fantasy-db"
    )

    cursor = connection.cursor()

    sql = f"""
SELECT pp.total_points,
       CASE WHEN pp.was_home THEN 1 ELSE 0 END,
       COALESCE(f.team_h_difficulty, 0),
       COALESCE(f.team_a_difficulty, 0),
       pp.minutes,
       p.position_id
FROM player_performances pp
         INNER JOIN fixtures f ON pp.fixture_id = f.fixture_id
         INNER JOIN player_season_positions psp ON (f.season_id = psp.season_id AND pp.player_id = psp.player_id)
         INNER JOIN positions p ON psp.position_id = p.position_id
WHERE pp.kickoff_time < ( SELECT kickoff_time FROM fixtures WHERE fixture_id = {fixture_id} )
  AND pp.player_id = {player_id}
ORDER BY pp.kickoff_time DESC
LIMIT 5;
"""

    cursor.execute(sql)
    records = cursor.fetchall()

    return list(chain.from_iterable(records))

build_training_data()
