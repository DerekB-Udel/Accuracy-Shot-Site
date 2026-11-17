"""Basketball Shot Predictor.

Uses simple projectile physics (NumPy) to predict whether a basketball
shot goes in, and shows the ball's path on a graph.

Extras: play on other planets, a coach hint button, ghost trails of earlier
shots on the graph, and a five-shot challenge mode.
"""
import math
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # draw graphs without opening a window (needed on GitHub and safe in Thonny)
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import numpy as np  # noqa: E402
from drafter import (Button, CheckBox, Header, MatPlotLibPlot, Page,  # noqa: E402
                     SelectBox, Table, TextBox, assert_equal, route,
                     set_site_information, start_server)

# ---------------------------------------------------------------------------
# STEP 1: Constants (allowed, because they never change)
# ---------------------------------------------------------------------------
GRAVITY = 9.80665          # m/s^2 (same value SciPy provides)
HOOP_HEIGHT = 3.05         # meters (regulation rim)
HOOP_TOLERANCE = 0.20      # meters of wiggle room that still counts as a swish
MAX_DEFENDER_HEIGHT = 1e100    # not a real limit: only stops the computer's numbers overflowing
MAX_DEFENDER_DISTANCE = 1e100  # same idea: defender can stand as far away as you want
MAX_DISTANCE = 1_000_000   # basically "as far as you want" (1000 km)
MAX_SPEED = 1e100          # not a real limit: only stops the computer's numbers overflowing
CUSTOM_CHOICE = "Custom"
DISTANCE_OPTIONS = {       # label shown to user -> distance in meters
    "Free throw (4.6 m)": 4.6,
    "Mid-range (6.0 m)": 6.0,
    "Three-pointer (7.2 m)": 7.2,
    "Half court (14.0 m)": 14.0,
}
PLANETS = {                # planet name -> gravity in m/s^2
    "Earth": GRAVITY,
    "Moon": 1.62,
    "Mars": 3.72,
    "Jupiter": 24.79,
}
GHOST_LIMIT = 5            # how many earlier shots are drawn faintly on the graph
ANGLE_STEP = 0.1           # the coach tries every angle this many degrees apart
CHALLENGE_SHOTS = 5        # shots in one challenge
CHALLENGE_DISTANCES = [4.6, 6.0, 7.2, 9.0, 14.0]  # random distances for each round
DEFENDER_CHANCE = 0.6      # chance a challenge round has a defender


# ---------------------------------------------------------------------------
# STEP 2: Dataclasses (the "nouns" of the program)
# ---------------------------------------------------------------------------
@dataclass
class ShotResult:
    """One past shot: its outcome, how it flew, and where it was fired."""
    outcome: str           # "Swish!", "Short", "Long", or "Blocked"
    peak_height: float     # highest point of the ball, in meters
    distance_from: float   # meters from the hoop at hoop height (0.0 = scored)
    points: int            # points earned (0 if the shot missed)
    angle: float           # degrees (kept so the graph can redraw the shot)
    speed: float           # meters per second
    height: float          # release height, meters
    distance: float        # meters from the hoop
    gravity: float         # gravity of the planet it was fired on
    planet: str            # name of that planet


@dataclass
class State:
    """Everything the website remembers between pages."""
    current_angle: float            # degrees
    current_speed: float            # meters per second
    current_height: float           # meters (release height)
    current_distance: float         # meters from the hoop
    defender_on: bool               # is the defender on the court?
    defender_height: float          # how tall the defender is, meters
    defender_distance: float        # how far the defender stands from you, meters
    shot_history: list[ShotResult]  # every shot fired so far
    message: str                    # feedback shown on the simulator page
    planet: str                     # which planet we are playing on
    hint: str                       # the coach's latest advice
    challenge_on: bool              # are we in challenge mode?
    challenge_shots_left: int       # shots remaining in this challenge
    challenge_score: int            # points scored in this challenge
    challenge_best: int             # best challenge score so far


# ---------------------------------------------------------------------------
# STEP 3: Physics helper functions
# ---------------------------------------------------------------------------
def parse_number(text: str) -> float:
    """Convert text to a float, returning -1.0 if it is not a usable number.

    ADVANCED TECHNIQUE: try/except catches the error Python raises when
    float("abc") fails, so the site doesn't crash on bad input.
    math.isfinite also rejects "nan" and "inf", which float() accepts.
    """
    try:
        number = float(text)
    except ValueError:
        return -1.0
    if not math.isfinite(number):
        return -1.0
    return number


def get_distance(choice: str, custom_text: str) -> float:
    """Return the distance in meters from the SelectBox / custom TextBox."""
    if choice == CUSTOM_CHOICE:
        return parse_number(custom_text)
    return DISTANCE_OPTIONS[choice]


def distance_label(distance: float) -> str:
    """Return the SelectBox label that matches a distance, else "Custom"."""
    for label, meters in DISTANCE_OPTIONS.items():
        if meters == distance:
            return label
    return CUSTOM_CHOICE


def point_value(distance: float) -> int:
    """Return how many points a made shot from this distance is worth."""
    if distance < 5.5:
        return 1   # free throw range
    if distance < 7.0:
        return 2   # mid-range
    if distance < 12.0:
        return 3   # three-pointer range
    return 5       # bonus for a really long shot


def peak_height(angle: float, speed: float, height: float,
                gravity: float = GRAVITY) -> float:
    """Return the highest point the ball reaches, in meters."""
    vertical_speed = speed * np.sin(np.radians(angle))
    # Physics: extra height gained = vy^2 / (2g)
    return float(height + vertical_speed ** 2 / (2 * gravity))


def ball_height_at(x: float, angle: float, speed: float, height: float,
                   gravity: float = GRAVITY) -> float:
    """Return the ball's height (m) when it is x meters from the shooter."""
    horizontal_speed = speed * np.cos(np.radians(angle))
    vertical_speed = speed * np.sin(np.radians(angle))
    time = x / horizontal_speed
    return float(height + vertical_speed * time - 0.5 * gravity * time ** 2)


def time_at_hoop(angle: float, speed: float, height: float,
                 gravity: float = GRAVITY) -> float:
    """Return the time the ball falls through hoop height, or -1.0 if never.

    We solve 0.5*g*t^2 - vy*t + (HOOP_HEIGHT - height) = 0 with the
    quadratic formula and take the LARGER root (the ball is coming down).
    """
    vertical_speed = speed * np.sin(np.radians(angle))
    discriminant = vertical_speed ** 2 - 2 * gravity * (HOOP_HEIGHT - height)
    if discriminant < 0:  # ball never climbs high enough
        return -1.0
    return float((vertical_speed + np.sqrt(discriminant)) / gravity)


def miss_amount(angle: float, speed: float, height: float,
                distance: float, gravity: float = GRAVITY) -> float:
    """Return how far (m) the ball lands past the hoop (negative = short)."""
    time = time_at_hoop(angle, speed, height, gravity)
    if time < 0:
        return -distance  # never reached the rim at all
    landing_x = speed * np.cos(np.radians(angle)) * time
    return float(landing_x - distance)


def defender_blocks(angle: float, speed: float, height: float, distance: float,
                    defender_distance: float, defender_height: float,
                    gravity: float = GRAVITY) -> bool:
    """Return True if the ball hits the defender on its way to the hoop."""
    if defender_distance >= distance:  # defender is behind the hoop
        return False
    ball_height = ball_height_at(defender_distance, angle, speed, height,
                                 gravity)
    # The ball must still be in the air (>= 0) and not above the defender
    return 0 <= ball_height <= defender_height


def judge_shot(angle: float, speed: float, height: float, distance: float,
               defender_on: bool, defender_height: float,
               defender_distance: float, gravity: float = GRAVITY) -> str:
    """Return "Blocked", "Short", "Long", or "Swish!" for a shot."""
    # IF 1: Defender clearance check (uses the defender's height and spot)
    if defender_on and defender_blocks(angle, speed, height, distance,
                                       defender_distance, defender_height,
                                       gravity):
        return "Blocked"
    # IF 2: Angle check (ball never rises to rim height)
    if time_at_hoop(angle, speed, height, gravity) < 0:
        return "Short"
    miss = miss_amount(angle, speed, height, distance, gravity)
    # IF 3: Scored, short, or long
    if abs(miss) <= HOOP_TOLERANCE:
        return "Swish!"
    if miss < 0:
        return "Short"
    return "Long"


def trajectory_points(angle: float, speed: float, height: float,
                      gravity: float = GRAVITY
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return NumPy arrays (x_values, y_values) for the ball's whole flight."""
    vx = speed * np.cos(np.radians(angle))
    vy = speed * np.sin(np.radians(angle))
    # Time the ball hits the floor (y = 0)
    ground_time = (vy + np.sqrt(vy ** 2 + 2 * gravity * height)) / gravity
    times = np.linspace(0, ground_time, 100)  # 100 evenly spaced moments
    return vx * times, height + vy * times - 0.5 * gravity * times ** 2


def validate_inputs(angle: float, speed: float, height: float,
                    distance: float, defender_height: float,
                    defender_distance: float) -> str:
    """Return an error message, or "" if all inputs are usable."""
    if not 1 <= angle <= 89:
        return "Angle must be a number from 1 to 89 degrees."
    if not 0 < speed <= MAX_SPEED:
        return "Speed must be a number above 0."
    if not 0 < height <= 3:
        return "Starting height must be a number between 0 and 3 m."
    if not 0 < distance <= MAX_DISTANCE:
        return f"Distance must be a number above 0 (up to {MAX_DISTANCE} m)."
    if not 0 < defender_height <= MAX_DEFENDER_HEIGHT:
        return "Defender height must be a number above 0."
    if not 0 < defender_distance <= MAX_DEFENDER_DISTANCE:
        return "Defender distance must be a number above 0."
    return ""


# ---------------------------------------------------------------------------
# STEP 3b: Coach, shot recording, and challenge helpers
# ---------------------------------------------------------------------------
def swish_angles(speed: float, height: float, distance: float, gravity: float,
                 defender_on: bool, defender_height: float,
                 defender_distance: float) -> list[float]:
    """Return every launch angle (0.1 degree steps) that scores a swish."""
    good = []
    # LOOP 3: try each angle from 1 to 89 degrees
    for angle in np.arange(1, 89 + ANGLE_STEP / 2, ANGLE_STEP):
        rounded = round(float(angle), 1)
        if judge_shot(rounded, speed, height, distance, defender_on,
                      defender_height, defender_distance,
                      gravity) == "Swish!":
            good.append(rounded)
    return good


def angle_ranges(angles: list[float]) -> list[list[float]]:
    """Group a sorted list of angles into [low, high] runs that touch."""
    ranges = []
    for angle in angles:
        # Angles within 1.5 steps of the last one belong to the same run
        if len(ranges) > 0 and angle - ranges[-1][1] <= ANGLE_STEP * 1.5:
            ranges[-1][1] = angle
        else:
            ranges.append([angle, angle])
    return ranges


def make_hint(speed: float, height: float, distance: float, gravity: float,
              defender_on: bool, defender_height: float,
              defender_distance: float) -> str:
    """Return the coach's advice as a sentence."""
    ranges = angle_ranges(swish_angles(speed, height, distance, gravity,
                                       defender_on, defender_height,
                                       defender_distance))
    if len(ranges) == 0:
        return ("Coach: no angle scores at this speed. "
                "Try changing your speed, height, or defender.")
    parts = []
    for low, high in ranges:
        if low == high:
            parts.append(f"{low}°")
        else:
            parts.append(f"{low}° to {high}°")
    return "Coach: try " + " or ".join(parts)


def record_shot(state: State) -> None:
    """Judge the shot described by state's current fields and save it."""
    gravity = PLANETS[state.planet]
    outcome = judge_shot(state.current_angle, state.current_speed,
                         state.current_height, state.current_distance,
                         state.defender_on, state.defender_height,
                         state.defender_distance, gravity)
    distance_from = 0.0
    points = 0
    if outcome == "Swish!":
        points = point_value(state.current_distance)
    else:
        distance_from = round(abs(miss_amount(
            state.current_angle, state.current_speed, state.current_height,
            state.current_distance, gravity)), 2)
    state.shot_history.append(ShotResult(
        outcome,
        round(peak_height(state.current_angle, state.current_speed,
                          state.current_height, gravity), 2),
        distance_from, points, state.current_angle, state.current_speed,
        state.current_height, state.current_distance, gravity, state.planet))


def new_round(state: State) -> None:
    """Pick a random distance and defender for the next challenge shot."""
    state.current_distance = random.choice(CHALLENGE_DISTANCES)
    state.defender_on = random.random() < DEFENDER_CHANCE
    state.defender_height = round(random.uniform(1.8, 2.3), 2)
    # Always in front of the hoop, so the defender can actually matter
    state.defender_distance = round(
        random.uniform(1.0, state.current_distance * 0.7), 1)
    state.message = ""
    state.hint = ""


# ---------------------------------------------------------------------------
# STEP 4: Routes (each one is a page or an action)
# ---------------------------------------------------------------------------
@route
def index(state: State) -> Page:
    """Simulator page: the input controls (or the challenge controls)."""
    if state.challenge_on:
        shot_number = CHALLENGE_SHOTS - state.challenge_shots_left + 1
        defender_text = "No defender this round."
        if state.defender_on:
            defender_text = (f"Defender: {state.defender_height} m tall, "
                             f"{state.defender_distance} m from you.")
        return Page(state, [
            Header("Challenge Mode"),
            f"Shot {shot_number} of {CHALLENGE_SHOTS}  |  "
            f"Score: {state.challenge_score}",
            f"Planet: {state.planet}",
            f"Distance to hoop: {state.current_distance} m "
            f"({point_value(state.current_distance)} points if you score)",
            defender_text,
            state.message,
            state.hint,
            "Launch angle (1-89 degrees):",
            TextBox("angle_text", str(state.current_angle)),
            "Ball speed (m/s, no limit):",
            TextBox("speed_text", str(state.current_speed)),
            "Starting height (m):",
            TextBox("height_text", str(state.current_height)),
            Button("Fire Shot", challenge_shot),
            Button("Coach Hint", challenge_hint),
            Button("Quit Challenge", end_challenge),
        ])
    return Page(state, [
        Header("Basketball Shot Predictor"),
        "Enter your shot and see if it goes in!",
        state.message,
        state.hint,
        "Planet:",
        SelectBox("planet_choice", list(PLANETS), state.planet),
        "Launch angle (1-89 degrees):",
        TextBox("angle_text", str(state.current_angle)),
        "Ball speed (m/s, no limit):",
        TextBox("speed_text", str(state.current_speed)),
        "Starting height (m):",
        TextBox("height_text", str(state.current_height)),
        "Distance from hoop:",
        SelectBox("distance_choice",
                  list(DISTANCE_OPTIONS) + [CUSTOM_CHOICE],
                  distance_label(state.current_distance)),
        "Custom distance in meters (only used if 'Custom' is chosen):",
        TextBox("custom_text", str(state.current_distance)),
        CheckBox("defender", state.defender_on),
        "Add a defender",
        "Defender height (m):",
        TextBox("defender_height_text", str(state.defender_height)),
        "Defender distance from you (m):",
        TextBox("defender_distance_text", str(state.defender_distance)),
        Button("Fire Shot", fire_shot),
        Button("Coach Hint", coach_hint),
        Button("Start Challenge", start_challenge),
        Button("Shot History", show_history),
        Button("Stats", show_stats),
    ])


@route
def fire_shot(state: State, angle_text: str, speed_text: str,
              height_text: str, distance_choice: str, custom_text: str,
              planet_choice: str, defender: bool, defender_height_text: str,
              defender_distance_text: str) -> Page:
    """Read inputs, update the state, judge the shot, save it to history."""
    angle = parse_number(angle_text)
    speed = parse_number(speed_text)
    height = parse_number(height_text)
    distance = get_distance(distance_choice, custom_text)
    defender_height = parse_number(defender_height_text)
    defender_distance = parse_number(defender_distance_text)

    error = validate_inputs(angle, speed, height, distance,
                            defender_height, defender_distance)
    if error != "":
        state.message = error
        state.hint = ""
        return index(state)

    # These lines modify state fields (a project requirement)
    state.current_angle = angle
    state.current_speed = speed
    state.current_height = height
    state.current_distance = distance
    state.defender_on = defender
    state.defender_height = defender_height
    state.defender_distance = defender_distance
    state.planet = planet_choice
    state.message = ""
    state.hint = ""

    record_shot(state)
    return show_result(state)


@route
def coach_hint(state: State, angle_text: str, speed_text: str,
               height_text: str, distance_choice: str, custom_text: str,
               planet_choice: str, defender: bool, defender_height_text: str,
               defender_distance_text: str) -> Page:
    """Tell the player which angles would swish with their current setup."""
    angle = parse_number(angle_text)
    speed = parse_number(speed_text)
    height = parse_number(height_text)
    distance = get_distance(distance_choice, custom_text)
    defender_height = parse_number(defender_height_text)
    defender_distance = parse_number(defender_distance_text)

    error = validate_inputs(angle, speed, height, distance,
                            defender_height, defender_distance)
    if error != "":
        state.message = error
        state.hint = ""
        return index(state)

    # Keep what the player typed so the boxes don't reset
    state.current_angle = angle
    state.current_speed = speed
    state.current_height = height
    state.current_distance = distance
    state.defender_on = defender
    state.defender_height = defender_height
    state.defender_distance = defender_distance
    state.planet = planet_choice
    state.message = ""
    state.hint = make_hint(speed, height, distance, PLANETS[planet_choice],
                           defender, defender_height, defender_distance)
    return index(state)


@route
def start_challenge(state: State, planet_choice: str) -> Page:
    """Begin a new challenge on the chosen planet."""
    state.planet = planet_choice
    state.challenge_on = True
    state.challenge_shots_left = CHALLENGE_SHOTS
    state.challenge_score = 0
    new_round(state)
    return index(state)


@route
def challenge_shot(state: State, angle_text: str, speed_text: str,
                   height_text: str) -> Page:
    """Fire one challenge shot at this round's distance and defender."""
    angle = parse_number(angle_text)
    speed = parse_number(speed_text)
    height = parse_number(height_text)

    error = validate_inputs(angle, speed, height, state.current_distance,
                            state.defender_height, state.defender_distance)
    if error != "":
        state.message = error
        state.hint = ""
        return index(state)

    state.current_angle = angle
    state.current_speed = speed
    state.current_height = height
    state.message = ""
    state.hint = ""

    record_shot(state)
    state.challenge_score += state.shot_history[-1].points
    state.challenge_shots_left -= 1
    return show_result(state)


@route
def challenge_hint(state: State, angle_text: str, speed_text: str,
                   height_text: str) -> Page:
    """Coach advice for the current challenge round."""
    angle = parse_number(angle_text)
    speed = parse_number(speed_text)
    height = parse_number(height_text)

    error = validate_inputs(angle, speed, height, state.current_distance,
                            state.defender_height, state.defender_distance)
    if error != "":
        state.message = error
        state.hint = ""
        return index(state)

    state.current_angle = angle
    state.current_speed = speed
    state.current_height = height
    state.message = ""
    state.hint = make_hint(speed, height, state.current_distance,
                           PLANETS[state.planet], state.defender_on,
                           state.defender_height, state.defender_distance)
    return index(state)


@route
def next_challenge_shot(state: State) -> Page:
    """Set up the next challenge round and go back to the controls."""
    new_round(state)
    return index(state)


@route
def challenge_over(state: State) -> Page:
    """Final score page for a finished challenge."""
    new_best = state.challenge_score > state.challenge_best
    if new_best:
        state.challenge_best = state.challenge_score
    summary = f"Best score: {state.challenge_best}"
    if new_best:
        summary = "New personal best!"
    return Page(state, [
        Header("Challenge Complete!"),
        f"Final score: {state.challenge_score}",
        summary,
        "Planet for the next challenge:",
        SelectBox("planet_choice", list(PLANETS), state.planet),
        Button("Play Again", start_challenge),
        Button("Back to Simulator", end_challenge),
    ])


@route
def end_challenge(state: State) -> Page:
    """Leave challenge mode and go back to the free-play simulator."""
    state.challenge_on = False
    state.message = ""
    state.hint = ""
    return index(state)


@route
def show_result(state: State) -> Page:
    """Result page: the outcome of the latest shot."""
    latest = state.shot_history[-1]
    content = [
        Header("Result: " + latest.outcome),
        f"Planet: {latest.planet} (gravity {latest.gravity} m/s²)",
        f"Peak height: {latest.peak_height} m",
        f"Missed the hoop by: {latest.distance_from} m",
        f"Points earned: {latest.points}",
    ]
    if state.challenge_on:
        content.append(f"Challenge score: {state.challenge_score}")
    content.append(Button("See Graph", show_graph))
    if not state.challenge_on:
        content.append(Button("Shoot Again", index))
    elif state.challenge_shots_left == 0:
        content.append(Button("Final Score", challenge_over))
    else:
        content.append(Button("Next Shot", next_challenge_shot))
    return Page(state, content)


@route
def show_graph(state: State) -> Page:
    """Graph page: the latest shot's path, with faint ghosts of earlier ones."""
    latest = state.shot_history[-1]
    earlier = state.shot_history[-1 - GHOST_LIMIT:-1]
    plt.close("all")  # don't pile up old figures in memory
    plt.figure()
    # LOOP 4: draw each earlier shot faintly (older shots are fainter)
    for position, ghost in enumerate(earlier):
        ghost_x, ghost_y = trajectory_points(
            ghost.angle, ghost.speed, ghost.height, ghost.gravity)
        fade = 0.15 + 0.45 * (position + 1) / len(earlier)
        label = "Earlier shots" if position == 0 else "_nolegend_"
        plt.plot(ghost_x, ghost_y, color="gray", linestyle="--",
                 alpha=fade, label=label)
    x_values, y_values = trajectory_points(
        latest.angle, latest.speed, latest.height, latest.gravity)
    plt.plot(x_values, y_values, color="tab:blue", linewidth=2,
             label="Latest shot")
    plt.scatter([latest.distance], [HOOP_HEIGHT],
                color="red", label="Hoop")
    right_edge = max(float(x_values.max()), latest.distance)
    if state.defender_on:
        plt.plot([state.defender_distance, state.defender_distance],
                 [0, state.defender_height],
                 color="black", linewidth=4, label="Defender")
        right_edge = max(right_edge, state.defender_distance)
    plt.ylim(bottom=0)
    plt.xlim(0, right_edge * 1.05)  # widen the x-axis so everything is visible
    plt.xlabel("Distance from shooter (m)")
    plt.ylabel("Height (m)")
    plt.title("Shot Trajectory on " + latest.planet)
    plt.legend()
    return Page(state, [
        Header("Shot Trajectory"),
        MatPlotLibPlot(),
        Button("Back to Result", show_result),
    ])


@route
def show_history(state: State) -> Page:
    """History page: a table of every past shot."""
    state.message = ""
    state.hint = ""
    rows = [["Shot #", "Planet", "Outcome", "Peak height (m)",
             "Missed by (m)", "Points"]]
    # LOOP 1: build one table row per ShotResult in the history
    for number, shot in enumerate(state.shot_history, start=1):
        rows.append([str(number), shot.planet, shot.outcome,
                     str(shot.peak_height), str(shot.distance_from),
                     str(shot.points)])
    return Page(state, [
        Header("Shot History"),
        Table(rows),
        Button("Clear History", reset_history),
        Button("Back to Simulator", index),
    ])


@route
def show_stats(state: State) -> Page:
    """Stats page: shooting percentage, points, and average peak height."""
    state.message = ""
    state.hint = ""
    made = 0
    total_peak = 0.0
    total_points = 0
    # LOOP 2: count made shots and add up peak heights and points
    for shot in state.shot_history:
        if shot.outcome == "Swish!":
            made += 1
        total_peak += shot.peak_height
        total_points += shot.points
    attempts = len(state.shot_history)
    percentage = 0.0
    average_peak = 0.0
    if attempts > 0:  # avoid dividing by zero
        percentage = round(made / attempts * 100, 1)
        average_peak = round(total_peak / attempts, 2)
    return Page(state, [
        Header("Stats & Analytics"),
        f"Shots taken: {attempts}",
        f"Shots made: {made}",
        f"Shooting percentage: {percentage}%",
        f"Total points: {total_points}",
        f"Average peak height: {average_peak} m",
        f"Best challenge score: {state.challenge_best}",
        Button("Back to Simulator", index),
    ])


@route
def reset_history(state: State) -> Page:
    """Empty the shot history list."""
    state.shot_history = []
    state.message = "History cleared."
    return index(state)


# ---------------------------------------------------------------------------
# STEP 5: Unit tests (run when the file starts)
# ---------------------------------------------------------------------------
def make_test_state() -> State:
    """Build a fresh State for tests."""
    return State(52.0, 7.6, 2.0, 4.6, False, 2.1, 1.5, [], "",
                 "Earth", "", False, 0, 0, 0)


MOON = PLANETS["Moon"]
JUPITER = PLANETS["Jupiter"]

assert_equal(parse_number("12.5"), 12.5)
assert_equal(parse_number("abc"), -1.0)
assert_equal(parse_number("inf"), -1.0)
assert_equal(get_distance("Free throw (4.6 m)", ""), 4.6)
assert_equal(get_distance(CUSTOM_CHOICE, "9"), 9.0)
assert_equal(distance_label(6.0), "Mid-range (6.0 m)")
assert_equal(distance_label(123.0), CUSTOM_CHOICE)
assert_equal(round(peak_height(90, 9.80665, 2.0), 2), 6.9)
assert_equal(time_at_hoop(10, 5, 2.0), -1.0)  # too flat and slow
assert_equal(judge_shot(52, 7.6, 2.0, 4.6, False, 2.1, 1.5), "Swish!")
assert_equal(judge_shot(52, 6.0, 2.0, 4.6, False, 2.1, 1.5), "Short")
assert_equal(judge_shot(52, 9.0, 2.0, 4.6, False, 2.1, 1.5), "Long")
# A flat 5 degree shot is about 2 m high at the defender: a 2.1 m defender
# blocks it, a 0.5 m one does not, and a defender behind the hoop does not
assert_equal(judge_shot(5, 8.0, 2.0, 4.6, True, 2.1, 1.5), "Blocked")
assert_equal(judge_shot(5, 8.0, 2.0, 4.6, True, 0.5, 1.5), "Short")
assert_equal(judge_shot(5, 8.0, 2.0, 4.6, True, 2.1, 10.0), "Short")
# Defender standing where the ball has already landed does not block it
assert_equal(defender_blocks(5, 8.0, 2.0, 100.0, 50.0, 2.1), False)
# A hoop 500 m away is allowed; this shot just falls short
assert_equal(judge_shot(52, 7.6, 2.0, 500.0, False, 2.1, 1.5), "Short")
# Speed has no cap: a huge speed sends the ball far past a close hoop
assert_equal(judge_shot(52, 5000.0, 2.0, 4.6, False, 2.1, 1.5), "Long")
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 2.1, 1.5), "")
assert_equal(validate_inputs(52, 5000.0, 2.0, 500.0, 2.1, 300.0), "")
assert_equal(validate_inputs(0, 7.6, 2.0, 4.6, 2.1, 1.5) != "", True)
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 0.0, 1.5) != "", True)
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 2.1, 0.0) != "", True)
# Defender height and distance have no real cap
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 5000.0, 9e9), "")

# Planet tests: the same shot behaves very differently with other gravity
assert_equal(judge_shot(52, 7.6, 2.0, 4.6, False, 2.1, 1.5, MOON), "Long")
assert_equal(judge_shot(52, 7.6, 2.0, 4.6, False, 2.1, 1.5, JUPITER), "Short")
assert_equal(time_at_hoop(52, 7.6, 2.0, JUPITER), -1.0)  # never reaches the rim
assert_equal(round(peak_height(90, 1.62, 2.0, MOON), 2), 2.81)
moon_x, moon_y = trajectory_points(52, 7.6, 2.0, MOON)
assert_equal(round(float(moon_y[-1]), 6), 0.0)  # ends on the floor
assert_equal(float(moon_x.max()) > 30.0, True)  # and flies much farther

# Points
assert_equal(point_value(4.6), 1)
assert_equal(point_value(6.0), 2)
assert_equal(point_value(7.2), 3)
assert_equal(point_value(9.0), 3)
assert_equal(point_value(14.0), 5)

# Coach tests
assert_equal(52.0 in swish_angles(7.6, 2.0, 4.6, GRAVITY, False, 2.1, 1.5),
             True)
assert_equal(swish_angles(7.6, 2.0, 4.6, JUPITER, False, 2.1, 1.5), [])
assert_equal(angle_ranges([50.0, 50.1, 50.2, 52.0]),
             [[50.0, 50.2], [52.0, 52.0]])
assert_equal(angle_ranges([]), [])
assert_equal(make_hint(7.6, 2.0, 4.6, GRAVITY, False, 2.1, 1.5)
             .startswith("Coach: try"), True)
assert_equal(make_hint(7.6, 2.0, 4.6, JUPITER, False, 2.1, 1.5)
             .startswith("Coach: no angle"), True)

# Route tests: fire_shot returns a Page whose .state holds our changes
fired = fire_shot(make_test_state(), "52", "7.6", "2.0",
                  "Free throw (4.6 m)", "", "Earth", False, "2.1", "1.5")
assert_equal(len(fired.state.shot_history), 1)
assert_equal(fired.state.shot_history[0].outcome, "Swish!")
assert_equal(fired.state.shot_history[0].points, 1)
tall = fire_shot(make_test_state(), "52", "7.6", "2.0",
                 "Free throw (4.6 m)", "", "Earth", True, "3.0", "2.5")
assert_equal(tall.state.defender_height, 3.0)
assert_equal(tall.state.defender_distance, 2.5)
far = fire_shot(make_test_state(), "52", "7.6", "2.0",
                CUSTOM_CHOICE, "250", "Earth", False, "2.1", "1.5")
assert_equal(far.state.current_distance, 250.0)
fast = fire_shot(make_test_state(), "52", "5000", "2.0",
                 "Free throw (4.6 m)", "", "Earth", False, "2.1", "1.5")
assert_equal(fast.state.current_speed, 5000.0)
on_moon = fire_shot(make_test_state(), "52", "7.6", "2.0",
                    "Free throw (4.6 m)", "", "Moon", False, "2.1", "1.5")
assert_equal(on_moon.state.planet, "Moon")
assert_equal(on_moon.state.shot_history[0].outcome, "Long")
assert_equal(on_moon.state.shot_history[0].gravity, MOON)
bad = fire_shot(make_test_state(), "abc", "7.6", "2.0",
                "Free throw (4.6 m)", "", "Earth", False, "2.1", "1.5")
assert_equal(len(bad.state.shot_history), 0)
assert_equal(bad.state.message != "", True)
assert_equal(reset_history(fired.state).state.shot_history, [])
assert_equal(show_history(make_test_state()).state.shot_history, [])
assert_equal(show_stats(make_test_state()).state.shot_history, [])

# Coach hint route: saves the typed inputs and writes a hint
hinted = coach_hint(make_test_state(), "40", "7.6", "2.0",
                    "Free throw (4.6 m)", "", "Earth", False, "2.1", "1.5")
assert_equal(hinted.state.hint.startswith("Coach: try"), True)
assert_equal(hinted.state.current_angle, 40.0)
assert_equal(len(hinted.state.shot_history), 0)  # a hint is not a shot

# Challenge mode tests
started = start_challenge(make_test_state(), "Mars")
assert_equal(started.state.planet, "Mars")
assert_equal(started.state.challenge_on, True)
assert_equal(started.state.challenge_shots_left, CHALLENGE_SHOTS)
assert_equal(started.state.challenge_score, 0)
assert_equal(started.state.current_distance in CHALLENGE_DISTANCES, True)
round_state = make_test_state()
# LOOP 5: a defender must always stand in front of the hoop
for _ in range(50):
    new_round(round_state)
    assert_equal(0 < round_state.defender_distance
                 < round_state.current_distance, True)

# A fixed challenge: a swish worth 1 point, then a miss, then the final score
challenge = make_test_state()
challenge.challenge_on = True
challenge.challenge_shots_left = 2
first = challenge_shot(challenge, "52", "7.6", "2.0")
assert_equal(first.state.shot_history[-1].outcome, "Swish!")
assert_equal(first.state.challenge_score, 1)
assert_equal(first.state.challenge_shots_left, 1)
second = challenge_shot(first.state, "52", "6.0", "2.0")
assert_equal(second.state.shot_history[-1].outcome, "Short")
assert_equal(second.state.challenge_score, 1)
assert_equal(second.state.challenge_shots_left, 0)
final = challenge_over(second.state)
assert_equal(final.state.challenge_best, 1)
assert_equal(end_challenge(final.state).state.challenge_on, False)
bad_challenge = make_test_state()
bad_challenge.challenge_on = True
bad_challenge.challenge_shots_left = 3
rejected = challenge_shot(bad_challenge, "0", "7.6", "2.0")
assert_equal(rejected.state.challenge_shots_left, 3)  # bad input costs no shot
challenge_hinted = challenge_hint(make_test_state(), "52", "7.6", "2.0")
assert_equal(challenge_hinted.state.hint.startswith("Coach: try"), True)
# index, show_result, and show_graph only display data (show_graph also draws
# a picture), so they are checked by running the site by hand.


# ---------------------------------------------------------------------------
# STEP 6: Start the website
# ---------------------------------------------------------------------------
set_site_information(
    author="Derek Bellin",
    description="Predicts whether a basketball shot goes in using physics.",
    sources=["https://en.wikipedia.org/wiki/Projectile_motion","LLM Documentation (2).pdf"],
    planning=["Copy of CS1 Final Project 1_ Design.pdf",
              "Copy of CS1 Final Project 0_ Pick a Problem.pdf"],
    links=["https://derekb-udel.github.io/Accuracy-Shot-Site/"],
)
start_server(State(52.0, 7.6, 2.0, 4.6, False, 2.1, 1.5, [], "",
                   "Earth", "", False, 0, 0, 0))
