"""Basketball Shot Predictor.

Uses simple projectile physics (NumPy) to predict whether a basketball
shot goes in, and shows the ball's path on a graph.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt  # used only to draw the trajectory graph
import numpy as np
from drafter import (Button, CheckBox, Header, MatPlotLibPlot, Page,
                     SelectBox, Table, TextBox, assert_equal, route,
                     set_site_information, start_server)

# ---------------------------------------------------------------------------
# STEP 1: Constants (allowed, because they never change)
# ---------------------------------------------------------------------------
GRAVITY = 9.80665         # m/s^2 (same value SciPy provides)
HOOP_HEIGHT = 3.05        # meters (regulation rim)
HOOP_TOLERANCE = 0.20     # meters of wiggle room that still counts as a swish
DEFENDER_HEIGHT = 2.1     # meters
DEFENDER_X = 1.5          # meters in front of the shooter
CUSTOM_CHOICE = "Custom"
DISTANCE_OPTIONS = {      # label shown to user -> distance in meters
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
    shot_history: list[ShotResult]  # every shot fired so far
    message: str                    # feedback shown on the simulator page


# ---------------------------------------------------------------------------
# STEP 3: Physics helper functions
# ---------------------------------------------------------------------------
def parse_number(text: str) -> float:
    """Convert text to a float, returning -1.0 if it is not a valid number.

    ADVANCED TECHNIQUE: try/except catches the error Python raises when
    float("abc") fails, so the site doesn't crash on bad input.
    Not unit tested with a route because it is tested directly below.
    """
    try:
        return float(text)
    except ValueError:
        return -1.0


def get_distance(choice: str, custom_text: str) -> float:
    """Return the distance in meters from the SelectBox / custom TextBox."""
    if choice == CUSTOM_CHOICE:
        return parse_number(custom_text)
    return DISTANCE_OPTIONS[choice]


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


def judge_shot(angle: float, speed: float, height: float,
               distance: float, defender_on: bool) -> str:
    """Return "Blocked", "Short", "Long", or "Swish!" for a shot."""
    # IF 1: Defender clearance check
    if defender_on and DEFENDER_X < distance:
        if ball_height_at(DEFENDER_X, angle, speed, height) <= DEFENDER_HEIGHT:
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
                    distance: float) -> str:
    """Return an error message, or "" if all inputs are usable."""
    if not 1 <= angle <= 89:
        return "Angle must be a number from 1 to 89 degrees."
    if speed <= 0 or speed > 30:
        return "Speed must be a number between 0 and 30 m/s."
    if height <= 0 or height > 3:
        return "Starting height must be a number between 0 and 3 m."
    if distance <= 0 or distance > 30:
        return "Distance must be a number between 0 and 30 m."
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
        "Ball speed (m/s):",
        TextBox("speed_text", str(state.current_speed)),
        "Starting height (m):",
        TextBox("height_text", str(state.current_height)),
        "Distance from hoop:",
        SelectBox("distance_choice",
                  list(DISTANCE_OPTIONS) + [CUSTOM_CHOICE],
                  list(DISTANCE_OPTIONS)[0]),
        "Custom distance in meters (only used if 'Custom' is chosen):",
        TextBox("custom_text", str(state.current_distance)),
        CheckBox("defender", state.defender_on),
        "Add a defender (2.1 m tall, 1.5 m in front of you)",
        Button("Fire Shot", fire_shot),
        Button("Shot History", show_history),
        Button("Stats", show_stats),
    ])


@route
def fire_shot(state: State, angle_text: str, speed_text: str,
              height_text: str, distance_choice: str, custom_text: str,
              defender: bool) -> Page:
    """Read inputs, update the state, judge the shot, save it to history."""
    angle = parse_number(angle_text)
    speed = parse_number(speed_text)
    height = parse_number(height_text)
    distance = get_distance(distance_choice, custom_text)

    error = validate_inputs(angle, speed, height, distance)
    if error != "":
        state.message = error
        return index(state)

    # These lines modify state fields (a project requirement)
    state.current_angle = angle
    state.current_speed = speed
    state.current_height = height
    state.current_distance = distance
    state.defender_on = defender
    state.message = ""

    outcome = judge_shot(angle, speed, height, distance, defender)
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
    if state.defender_on:
        plt.plot([DEFENDER_X, DEFENDER_X], [0, DEFENDER_HEIGHT],
                 color="black", linewidth=4, label="Defender")
    plt.ylim(bottom=0)
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
    return State(52.0, 7.6, 2.0, 4.6, False, [], "")


assert_equal(parse_number("12.5"), 12.5)
assert_equal(parse_number("abc"), -1.0)
assert_equal(get_distance("Free throw (4.6 m)", ""), 4.6)
assert_equal(get_distance(CUSTOM_CHOICE, "9"), 9.0)
assert_equal(round(peak_height(90, 9.80665, 2.0), 2), 6.9)  # FIXED value
assert_equal(time_at_hoop(10, 5, 2.0), -1.0)  # too flat and slow
assert_equal(judge_shot(52, 7.6, 2.0, 4.6, False), "Swish!")
assert_equal(judge_shot(52, 6.0, 2.0, 4.6, False), "Short")
assert_equal(judge_shot(52, 9.0, 2.0, 4.6, False), "Long")
# FIXED: a flat 5 degree shot hits the defender; a 52 degree arc goes over him
assert_equal(judge_shot(5, 8.0, 2.0, 4.6, True), "Blocked")
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6), "")
assert_equal(validate_inputs(0, 7.6, 2.0, 4.6) != "", True)

# Route tests: fire_shot returns a Page whose .state holds our changes
fired = fire_shot(make_test_state(), "52", "7.6", "2.0",
                  "Free throw (4.6 m)", "", False)
assert_equal(len(fired.state.shot_history), 1)
assert_equal(fired.state.shot_history[0].outcome, "Swish!")
assert_equal(reset_history(fired.state).state.shot_history, [])
assert_equal(show_history(make_test_state()).state.shot_history, [])
assert_equal(show_stats(make_test_state()).state.shot_history, [])
# index, show_result, and show_graph only display data (show_graph also draws
# a picture), so they are checked by running the site by hand.


# ---------------------------------------------------------------------------
# STEP 6: Start the website
# ---------------------------------------------------------------------------
set_site_information(
    author="YOUR NAME",
    description="Predicts whether a basketball shot goes in using physics.",
    sources=["https://en.wikipedia.org/wiki/Projectile_motion"],
    planning=["planning.pdf"],
    links=["https://github.com/YOUR-USERNAME/YOUR-REPO"],
)
start_server(State(52.0, 7.6, 2.0, 4.6, False, [], ""))
