import numpy as np
import psycopg2
from itertools import chain


class Extractor:
    def __init__(self, lookahead: int = 10, form_size: int = 5):
        # lookahead is how many games in advance we'd like to generate training data for. i.e. "what did the player get
        # <lookahead> matches after the last data we have for them". This is useful when we want to predict a players
        # score <lookahead> matches in the future.
        self.lookahead = lookahead

        # this represents how many finished matches we'll consider for this player.
        self.form_size = form_size

    def build_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Builds training data for our fantasy prediction

        Parameters:
        argument1 (int): Description of arg1

        Returns:
        int:Returning value

       """
        records_by_player = self.build_records_per_player()

        X = []
        y = []
        for player_id in records_by_player:
            records = records_by_player[player_id]

            # look_ahead represents how many fixtures ahead the game we're predicting is
            # we generate data for up to lookahead
            for look_ahead in range(self.lookahead):
                for i in range(look_ahead, len(records) - self.form_size - look_ahead):
                    # include the number of games look_ahead we're including in the context to let ML know that
                    previous_fixture_data = list(
                        chain.from_iterable(records[i - look_ahead: i + self.form_size - look_ahead])
                    )
                    upcoming_fixture_context = list(records[i + self.form_size][2:])
                    upcoming_fixture_points = records[i + self.form_size][0]

                    context = previous_fixture_data + upcoming_fixture_context + [look_ahead]

                    X.append(context)
                    y.append([upcoming_fixture_points])

        return np.asarray(X), np.asarray(y)

    def build_records_per_player(self):
        cursor = self.get_cursor()
        sql = f"""
    SELECT 
           pp.player_id,
           pp.kickoff_time,
           pp.total_points,
           CASE WHEN pp.was_home THEN 1 ELSE 0 END,
           COALESCE(f.team_h_difficulty, 0),
           COALESCE(f.team_a_difficulty, 0),
           pp.minutes,
           psp.position_id
    FROM player_performances pp
             INNER JOIN fixtures f ON pp.fixture_id = f.fixture_id
             INNER JOIN player_season_positions psp ON (f.season_id = psp.season_id AND pp.player_id = psp.player_id)
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
                # initialise our player records with some blank records
                records_by_player[player_id] = [[0 for _ in range(6)] for _ in range(self.form_size + self.lookahead)]

            records_by_player[player_id].append(
                [
                    total_points,
                    minutes,
                    was_home,
                    home_difficulty,
                    away_difficulty,
                    position,
                ]
            )

        return records_by_player

    def get_cursor(self):
        connection = psycopg2.connect(
            user="fantasy-user",
            password="fantasy-pwd",
            host="192.168.1.106",
            port="5432",
            database="fantasy-db",
        )
        cursor = connection.cursor()

        return cursor

    def get_context(self, season: int):
        players = self.get_player_static_context(season)

        X = []
        context = []

        # get all un-played fixtures
        unplayed_fixtures = self.get_un_played_fixtures_by_team(season=season)

        # generate context for all un-played fixtures and for all players
        # for each player, get their last "form_size" records
        records_by_player = self.build_records_per_player()
        for (player_id, last_team_id, position_id) in players:
            if player_id not in records_by_player:
                continue

            records = records_by_player[player_id]
            form = list(chain.from_iterable(records[-self.form_size:]))
            team_unplayed_fixtures = unplayed_fixtures[last_team_id]
            lookahead = 0
            for (fixture_id, was_home, home_diff, away_diff) in team_unplayed_fixtures:
                X.append(form + [was_home, home_diff, away_diff] + [position_id] + [lookahead])
                context.append((player_id, fixture_id))
                lookahead += 1

        return X, context

    def get_player_static_context(self, season: int) -> list[tuple[int, int, int]]:
        sql = f"""
    SELECT p.player_id, p.last_team_id, psp.position_id
    FROM players p
             INNER JOIN player_season_positions psp ON p.player_id = psp.player_id
             INNER JOIN seasons s ON psp.season_id = s.season_id
    WHERE s.start_year = {season};
    """

        cursor = self.get_cursor()
        cursor.execute(sql)

        return cursor.fetchall()

    def get_un_played_fixtures_by_team(self, season: int) -> dict[int, tuple[int, int, int]]:
        sql = f"""
    SELECT 
        fixture_id,
        home_team_id,
        away_team_id,
        team_h_difficulty,
        team_a_difficulty,
        kickoff_time
    FROM fixtures f
             INNER JOIN seasons s ON f.season_id = s.season_id
    WHERE s.start_year = {season}
      AND f.finished_provisional = FALSE
      AND f.kickoff_time IS NOT NULL
    ORDER BY f.kickoff_time ASC;
    """

        cursor = self.get_cursor()
        cursor.execute(sql)

        fixtures_by_team = {}
        for (
                fixture_id,
                home_team_id,
                away_team_id,
                team_h_difficulty,
                team_a_difficulty,
                kickoff_time
        ) in cursor.fetchall():
            if home_team_id not in fixtures_by_team:
                fixtures_by_team[home_team_id] = []
            if away_team_id not in fixtures_by_team:
                fixtures_by_team[away_team_id] = []

            fixtures_by_team[home_team_id].append(
                (
                    fixture_id,
                    1,  # was_home
                    team_h_difficulty,
                    team_a_difficulty
                )
            )

            fixtures_by_team[away_team_id].append(
                (
                    fixture_id,
                    0,  # was_home
                    team_h_difficulty,
                    team_a_difficulty
                )
            )

        return fixtures_by_team
