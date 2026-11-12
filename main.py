"""Basketball Shot Predictor.

Uses simple projectile physics (NumPy) to predict whether a basketball
shot goes in, and shows the ball's path on a graph.
"""
import math
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


# ---------------------------------------------------------------------------
# STEP 2: Dataclasses (the "nouns" of the program)
# ---------------------------------------------------------------------------
@dataclass
class ShotResult:
    """One past shot: its outcome, peak height, and how far it missed by."""
    outcome: str           # "Swish!", "Short", "Long", or "Blocked"
    peak_height: float     # highest point of the ball, in meters
    distance_from: float   # meters from the hoop at hoop height (0.0 = scored)


@dataclass
class State:
    """Everything the website remembers between pages."""
    current_angle: float            # degrees
    current_speed: float            # meters per second
    current_height: float           # meters (release height)
    current_distance: float         # meters from the hoop
    defender_on: bool               # is the defender on the court?
    defender_height: float          # how tall the defender is, meters
    defender_distance: float        # NEW: how far the defender stands from you, meters
    shot_history: list[ShotResult]  # every shot fired so far
    message: str                    # feedback shown on the simulator page


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


def peak_height(angle: float, speed: float, height: float) -> float:
    """Return the highest point the ball reaches, in meters."""
    vertical_speed = speed * np.sin(np.radians(angle))
    # Physics: extra height gained = vy^2 / (2g)
    return float(height + vertical_speed ** 2 / (2 * GRAVITY))


def ball_height_at(x: float, angle: float, speed: float, height: float) -> float:
    """Return the ball's height (m) when it is x meters from the shooter."""
    horizontal_speed = speed * np.cos(np.radians(angle))
    vertical_speed = speed * np.sin(np.radians(angle))
    time = x / horizontal_speed
    return float(height + vertical_speed * time - 0.5 * GRAVITY * time ** 2)


def time_at_hoop(angle: float, speed: float, height: float) -> float:
    """Return the time the ball falls through hoop height, or -1.0 if never.

    We solve 0.5*g*t^2 - vy*t + (HOOP_HEIGHT - height) = 0 with the
    quadratic formula and take the LARGER root (the ball is coming down).
    """
    vertical_speed = speed * np.sin(np.radians(angle))
    discriminant = vertical_speed ** 2 - 2 * GRAVITY * (HOOP_HEIGHT - height)
    if discriminant < 0:  # ball never climbs high enough
        return -1.0
    return float((vertical_speed + np.sqrt(discriminant)) / GRAVITY)


def miss_amount(angle: float, speed: float, height: float,
                distance: float) -> float:
    """Return how far (m) the ball lands past the hoop (negative = short)."""
    time = time_at_hoop(angle, speed, height)
    if time < 0:
        return -distance  # never reached the rim at all
    landing_x = speed * np.cos(np.radians(angle)) * time
    return float(landing_x - distance)


def defender_blocks(angle: float, speed: float, height: float, distance: float,
                    defender_distance: float, defender_height: float) -> bool:
    """Return True if the ball hits the defender on its way to the hoop."""
    if defender_distance >= distance:  # defender is behind the hoop
        return False
    ball_height = ball_height_at(defender_distance, angle, speed, height)
    # The ball must still be in the air (>= 0) and not above the defender
    return 0 <= ball_height <= defender_height


def judge_shot(angle: float, speed: float, height: float, distance: float,
               defender_on: bool, defender_height: float,
               defender_distance: float) -> str:
    """Return "Blocked", "Short", "Long", or "Swish!" for a shot."""
    # IF 1: Defender clearance check (uses the defender's height and spot)
    if defender_on and defender_blocks(angle, speed, height, distance,
                                       defender_distance, defender_height):
        return "Blocked"
    # IF 2: Angle check (ball never rises to rim height)
    if time_at_hoop(angle, speed, height) < 0:
        return "Short"
    miss = miss_amount(angle, speed, height, distance)
    # IF 3: Scored, short, or long
    if abs(miss) <= HOOP_TOLERANCE:
        return "Swish!"
    if miss < 0:
        return "Short"
    return "Long"


def trajectory_points(angle: float, speed: float,
                      height: float) -> tuple[np.ndarray, np.ndarray]:
    """Return NumPy arrays (x_values, y_values) for the ball's whole flight."""
    vx = speed * np.cos(np.radians(angle))
    vy = speed * np.sin(np.radians(angle))
    # Time the ball hits the floor (y = 0)
    ground_time = (vy + np.sqrt(vy ** 2 + 2 * GRAVITY * height)) / GRAVITY
    times = np.linspace(0, ground_time, 100)  # 100 evenly spaced moments
    return vx * times, height + vy * times - 0.5 * GRAVITY * times ** 2


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
# STEP 4: Routes (each one is a page or an action)
# ---------------------------------------------------------------------------
@route
def index(state: State) -> Page:
    """Simulator page: the input controls."""
    return Page(state, [
        Header("Basketball Shot Predictor"),
        "Enter your shot and see if it goes in!",
        state.message,
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
        Button("Shot History", show_history),
        Button("Stats", show_stats),
    ])


@route
def fire_shot(state: State, angle_text: str, speed_text: str,
              height_text: str, distance_choice: str, custom_text: str,
              defender: bool, defender_height_text: str,
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
        return index(state)

    # These lines modify state fields (a project requirement)
    state.current_angle = angle
    state.current_speed = speed
    state.current_height = height
    state.current_distance = distance
    state.defender_on = defender
    state.defender_height = defender_height
    state.defender_distance = defender_distance
    state.message = ""

    outcome = judge_shot(angle, speed, height, distance, defender,
                         defender_height, defender_distance)
    distance_from = 0.0
    if outcome != "Swish!":
        distance_from = round(abs(miss_amount(angle, speed, height, distance)), 2)
    state.shot_history.append(
        ShotResult(outcome, round(peak_height(angle, speed, height), 2),
                   distance_from))
    return show_result(state)


@route
def show_result(state: State) -> Page:
    """Final page: the outcome of the latest shot."""
    latest = state.shot_history[-1]
    return Page(state, [
        Header("Result: " + latest.outcome),
        f"Peak height: {latest.peak_height} m",
        f"Missed the hoop by: {latest.distance_from} m",
        Button("See Graph", show_graph),
        Button("Shoot Again", index),
    ])


@route
def show_graph(state: State) -> Page:
    """Graph page: plot the latest shot's path using NumPy arrays."""
    x_values, y_values = trajectory_points(
        state.current_angle, state.current_speed, state.current_height)
    plt.figure()
    plt.plot(x_values, y_values, label="Ball path")
    plt.scatter([state.current_distance], [HOOP_HEIGHT],
                color="red", label="Hoop")
    right_edge = max(float(x_values.max()), state.current_distance)
    if state.defender_on:
        plt.plot([state.defender_distance, state.defender_distance],
                 [0, state.defender_height],
                 color="black", linewidth=4, label="Defender")
        right_edge = max(right_edge, state.defender_distance)
    plt.ylim(bottom=0)
    plt.xlim(0, right_edge * 1.05)  # widen the x-axis so everything is visible
    plt.xlabel("Distance from shooter (m)")
    plt.ylabel("Height (m)")
    plt.title("Shot Trajectory")
    plt.legend()
    return Page(state, [
        Header("Shot Trajectory"),
        MatPlotLibPlot(),
        Button("Back to Simulator", index),
    ])


@route
def show_history(state: State) -> Page:
    """History page: a table of every past shot."""
    rows = [["Shot #", "Outcome", "Peak height (m)", "Missed by (m)"]]
    # LOOP 1: build one table row per ShotResult in the history
    for number, shot in enumerate(state.shot_history, start=1):
        rows.append([str(number), shot.outcome, str(shot.peak_height),
                     str(shot.distance_from)])
    return Page(state, [
        Header("Shot History"),
        Table(rows),
        Button("Clear History", reset_history),
        Button("Back to Simulator", index),
    ])


@route
def show_stats(state: State) -> Page:
    """Stats page: shooting percentage and average peak height."""
    made = 0
    total_peak = 0.0
    # LOOP 2: count made shots and add up peak heights
    for shot in state.shot_history:
        if shot.outcome == "Swish!":
            made += 1
        total_peak += shot.peak_height
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
        f"Average peak height: {average_peak} m",
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
    return State(52.0, 7.6, 2.0, 4.6, False, 2.1, 1.5, [], "")


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

# Route tests: fire_shot returns a Page whose .state holds our changes
fired = fire_shot(make_test_state(), "52", "7.6", "2.0",
                  "Free throw (4.6 m)", "", False, "2.1", "1.5")
assert_equal(len(fired.state.shot_history), 1)
assert_equal(fired.state.shot_history[0].outcome, "Swish!")
tall = fire_shot(make_test_state(), "52", "7.6", "2.0",
                 "Free throw (4.6 m)", "", True, "3.0", "2.5")
assert_equal(tall.state.defender_height, 3.0)
assert_equal(tall.state.defender_distance, 2.5)
far = fire_shot(make_test_state(), "52", "7.6", "2.0",
                CUSTOM_CHOICE, "250", False, "2.1", "1.5")
assert_equal(far.state.current_distance, 250.0)
fast = fire_shot(make_test_state(), "52", "5000", "2.0",
                 "Free throw (4.6 m)", "", False, "2.1", "1.5")
assert_equal(fast.state.current_speed, 5000.0)
assert_equal(reset_history(fired.state).state.shot_history, [])
assert_equal(show_history(make_test_state()).state.shot_history, [])
assert_equal(show_stats(make_test_state()).state.shot_history, [])
# index, show_result, and show_graph only display data (show_graph also draws
# a picture), so they are checked by running the site by hand.


# ---------------------------------------------------------------------------
# STEP 6: Start the website
# ---------------------------------------------------------------------------
set_site_information(
    author="Derek Bellin",
    description="Predicts whether a basketball shot goes in using physics.",
    sources=["https://en.wikipedia.org/wiki/Projectile_motion"],
    planning=["Copy of CS1 Final Project 1_ Design.pdf",
              "Copy of CS1 Final Project 0_ Pick a Problem.pdf"],
    links=["https://derekb-udel.github.io/Accuracy-Shot-Site/"],
)
start_server(State(52.0, 7.6, 2.0, 4.6, False, 2.1, 1.5, [], ""))
