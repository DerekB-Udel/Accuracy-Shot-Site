"""Basketball Shot Predictor.

Pick a planet, set up your shot, and see if it goes in!

The program uses simple projectile physics (with NumPy) to work out where
the ball lands, and draws the ball's path on a graph.

How this file is organized:
    Step 1  Constants (numbers that never change)
    Step 2  Dataclasses (the things the program remembers)
    Step 3  Physics and helper functions
    Step 4  Pages (every screen and button on the website)
    Step 5  Tests (they run each time the program starts)
    Step 6  Start the website
"""
import math
import random
from dataclasses import dataclass

import drafter
import matplotlib
matplotlib.use("Agg")  # draw graphs without opening a window (works on GitHub and in Thonny)
import matplotlib.pyplot as plt  # noqa: E402  (has to come after matplotlib.use)
import numpy as np  # noqa: E402
from drafter import (Button, CheckBox, Header, MatPlotLibPlot, Page,  # noqa: E402
                     SelectBox, Table, TextBox, assert_equal, route,
                     set_site_information, start_server)

# ---------------------------------------------------------------------------
# STEP 1: Constants
# ---------------------------------------------------------------------------
GRAVITY = 9.80665          # Earth's gravity, m/s^2
HOOP_HEIGHT = 3.05         # a real rim is 3.05 meters up
HOOP_TOLERANCE = 0.20      # how close (meters) still counts as a swish
MAX_DEFENDER_HEIGHT = 1e100    # not a real limit, just stops numbers overflowing
MAX_DEFENDER_DISTANCE = 1e100  # same here: stand as far away as you like
MAX_DISTANCE = 1_000_000   # about as far as you want (1000 km)
MAX_SPEED = 1e100          # no real speed limit either
CUSTOM_CHOICE = "Custom"
DISTANCE_OPTIONS = {       # what the player sees -> distance in meters
    "Free throw (4.6 m)": 4.6,
    "Mid-range (6.0 m)": 6.0,
    "Three-pointer (7.2 m)": 7.2,
    "Half court (14.0 m)": 14.0,
}
PLANETS = {                # planet -> gravity in m/s^2
    "Earth": GRAVITY,
    "Moon": 1.62,
    "Mars": 3.72,
    "Jupiter": 24.79,
}
PLANET_BLURBS = {          # a fun line for each planet
    "Earth": "The classic. Normal basketball.",
    "Moon": "Floaty! Every shot flies super far.",
    "Mars": "Easy mode for big shots.",
    "Jupiter": "Heavy! You'll need real power.",
}
OUTCOME_LINES = {          # what we say about each result
    "Swish!": "Nothing but net! 🔥",
    "Short": "Not enough power. It fell short.",
    "Long": "Too much power. It flew right past.",
    "Blocked": "Rejected! The defender got a hand on it. ✋",
}
GRAPH_STYLES = {           # colors for the graph page
    "Neon night": {"background": "#0b1020", "text": "#e8ecff",
                   "path": "#00e5ff", "ghost": "#8892b0",
                   "hoop": "#ff9f1c", "floor": "#3a4a8a"},
    "Sunset": {"background": "#2b1030", "text": "#ffe6d5",
               "path": "#ffd166", "ghost": "#c77dff",
               "hoop": "#ff5d73", "floor": "#7b2d5f"},
    "Blueprint": {"background": "#0a3d91", "text": "#ffffff",
                  "path": "#ffffff", "ghost": "#9fc3ff",
                  "hoop": "#ffd60a", "floor": "#6aa0ff"},
}
DEFENDER_COLOR = "#ff4d6d"
GHOST_CHOICES = ["0", "1", "3", "5", "10"]  # how many old shots the graph can show
ANGLE_STEP = 0.1           # the coach tests an angle every 0.1 degrees
CHALLENGE_SHOTS = 5        # shots in one challenge
CHALLENGE_DISTANCES = [4.6, 6.0, 7.2, 9.0, 14.0]  # the hoop moves around
DEFENDER_CHANCE = 0.6      # how often a challenge round has a defender



LOOK = {
    "body": "background: radial-gradient(circle at top, #1b2550, #0b1020 70%) !important;"
            " color: #ffffff !important; font-family: 'Trebuchet MS', Verdana, sans-serif;",

    # Force all regular text to be white
    "p, span, label, div, li, td, th, strong, em": 
        "color: #ffffff !important;",

    # Headings
    "h1": "color: #ff9f1c !important; text-transform: uppercase;"
          " letter-spacing: 3px; text-shadow: 0 0 12px #ff9f1c;",

    "h2, h3, h4": "color: #00e5ff !important;"
                  " text-shadow: 0 0 8px #00e5ff;",

    # Buttons
    "button, input[type=submit]": "background: linear-gradient(#ffb347, #ff7b00) !important;"
                                  " color: #1a1200 !important; font-weight: bold;"
                                  " border: none; border-radius: 999px;"
                                  " padding: 8px 22px; margin: 4px;"
                                  " cursor: pointer;",

    # Input boxes
    "input[type=text], select": "background: #121a3a !important;"
                                " color: #ffffff !important;"
                                " border: 2px solid #3a4a8a;"
                                " border-radius: 8px; padding: 6px;",

    # Dropdown options
    "option": "background: #121a3a !important;"
              " color: #ffffff !important;",

    # Tables
    "table": "border-collapse: collapse; margin: 10px 0;",

    "th": "background: #1b2550 !important;"
          " color: #00e5ff !important; padding: 6px 12px;",

    "td": "background: #121a3a !important;"
          " color: #ffffff !important;"
          " border: 1px solid #3a4a8a; padding: 6px 12px;",

    # Checkboxes
    "input[type=checkbox]": "accent-color: #00e5ff;",
}



def apply_look() -> None:
    """Give the website its dark arcade style.

    Older versions of Drafter don't know every styling command, so if
    something isn't available we just skip it. The game still works.
    """
    add_css = getattr(drafter, "add_website_css", None)
    set_title = getattr(drafter, "set_website_title", None)
    hide_debug = getattr(drafter, "hide_debug_information", None)
    try:
        if add_css is not None:
            for selector, style in LOOK.items():
                add_css(selector, style)
        if set_title is not None:
            set_title("Basketball Shot Predictor")
        if hide_debug is not None:
            hide_debug()
    except Exception:  # styling is nice to have, never worth a crash
        print("Skipping the fancy look (this Drafter doesn't support it).")


# ---------------------------------------------------------------------------
# STEP 2: Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ShotResult:
    """One shot we already took, with everything needed to redraw it."""
    outcome: str           # "Swish!", "Short", "Long", or "Blocked"
    peak_height: float     # highest point of the ball, in meters
    distance_from: float   # how far it missed by, in meters (0.0 = scored)
    points: int            # points earned (0 for a miss)
    angle: float           # degrees
    speed: float           # meters per second
    height: float          # release height, meters
    distance: float        # distance to the hoop, meters
    gravity: float         # gravity of the planet it was shot on
    planet: str            # name of that planet


@dataclass
class State:
    """Everything the website remembers while you play."""
    current_angle: float            # degrees
    current_speed: float            # meters per second
    current_height: float           # release height, meters
    current_distance: float         # distance to the hoop, meters
    defender_on: bool               # is there a defender?
    defender_height: float          # how tall, meters
    defender_distance: float        # how far from you, meters
    shot_history: list[ShotResult]  # every shot so far
    message: str                    # a note shown once at the top of a page
    planet: str                     # which planet we're on
    hint: str                       # the coach's latest advice
    challenge_on: bool              # are we playing a challenge?
    challenge_shots_left: int       # shots left in this challenge
    challenge_score: int            # points in this challenge
    challenge_best: int             # best challenge score ever
    saved_setup: list               # your free-play setup, kept safe during a challenge
    ghost_count: int                # how many old shots to show on the graph
    graph_style: str                # which graph colors to use


def new_state() -> State:
    """Make a fresh State with the starting setup (a nice free throw)."""
    return State(52.0, 7.6, 2.0, 4.6, False, 2.1, 1.5, [], "",
                 "Earth", "", False, 0, 0, 0, [], 5, "Neon night")


# ---------------------------------------------------------------------------
# STEP 3: Physics and helper functions
# ---------------------------------------------------------------------------
def parse_number(text: str) -> float:
    """Turn typed text into a number. Bad text gives -1.0 instead of a crash.

    ADVANCED TECHNIQUE: try/except catches the error Python raises when
    float("abc") fails. isfinite also rejects "nan" and "inf".
    """
    try:
        number = float(text)
    except ValueError:
        return -1.0
    if not math.isfinite(number):
        return -1.0
    return number


def get_distance(choice: str, custom_text: str) -> float:
    """Get the distance in meters from the dropdown (or the custom box)."""
    if choice == CUSTOM_CHOICE:
        return parse_number(custom_text)
    return DISTANCE_OPTIONS[choice]


def distance_label(distance: float) -> str:
    """Find the dropdown label for a distance, or "Custom" if none match."""
    for label, meters in DISTANCE_OPTIONS.items():
        if meters == distance:
            return label
    return CUSTOM_CHOICE


def point_value(distance: float) -> int:
    """Points for scoring from this far away. Farther is worth more."""
    if distance < 5.5:
        return 1   # free throw range
    if distance < 7.0:
        return 2   # mid-range
    if distance < 12.0:
        return 3   # three-pointer range
    return 5       # bonus for a really long shot


def peak_height(angle: float, speed: float, height: float,
                gravity: float = GRAVITY) -> float:
    """How high the ball gets, in meters."""
    vertical_speed = speed * np.sin(np.radians(angle))
    # Physics: extra height = vy^2 / (2g)
    return float(height + vertical_speed ** 2 / (2 * gravity))


def ball_height_at(x: float, angle: float, speed: float, height: float,
                   gravity: float = GRAVITY) -> float:
    """How high the ball is when it's x meters away from the shooter."""
    horizontal_speed = speed * np.cos(np.radians(angle))
    vertical_speed = speed * np.sin(np.radians(angle))
    time = x / horizontal_speed
    return float(height + vertical_speed * time - 0.5 * gravity * time ** 2)


def time_at_hoop(angle: float, speed: float, height: float,
                 gravity: float = GRAVITY) -> float:
    """When the ball comes down through rim height (-1.0 if it never gets there).

    This solves 0.5*g*t^2 - vy*t + (HOOP_HEIGHT - height) = 0 with the
    quadratic formula. We take the bigger answer because the ball is
    falling at that moment.
    """
    vertical_speed = speed * np.sin(np.radians(angle))
    discriminant = vertical_speed ** 2 - 2 * gravity * (HOOP_HEIGHT - height)
    if discriminant < 0:  # the ball never climbs high enough
        return -1.0
    return float((vertical_speed + np.sqrt(discriminant)) / gravity)


def miss_amount(angle: float, speed: float, height: float,
                distance: float, gravity: float = GRAVITY) -> float:
    """How far past the hoop the ball lands (negative means it fell short)."""
    time = time_at_hoop(angle, speed, height, gravity)
    if time < 0:
        return -distance  # it never even reached the rim
    landing_x = speed * np.cos(np.radians(angle)) * time
    return float(landing_x - distance)


def defender_blocks(angle: float, speed: float, height: float, distance: float,
                    defender_distance: float, defender_height: float,
                    gravity: float = GRAVITY) -> bool:
    """True if the ball runs into the defender on the way to the hoop."""
    if defender_distance >= distance:  # defender is behind the hoop
        return False
    ball_height = ball_height_at(defender_distance, angle, speed, height,
                                 gravity)
    # Blocked only if the ball is still in the air and not above the defender
    return 0 <= ball_height <= defender_height


def judge_shot(angle: float, speed: float, height: float, distance: float,
               defender_on: bool, defender_height: float,
               defender_distance: float, gravity: float = GRAVITY) -> str:
    """Decide what happened: "Blocked", "Short", "Long", or "Swish!"."""
    # IF 1: did the defender get it?
    if defender_on and defender_blocks(angle, speed, height, distance,
                                       defender_distance, defender_height,
                                       gravity):
        return "Blocked"
    # IF 2: does the ball ever get as high as the rim?
    if time_at_hoop(angle, speed, height, gravity) < 0:
        return "Short"
    miss = miss_amount(angle, speed, height, distance, gravity)
    # IF 3: close enough to score, or short, or long?
    if abs(miss) <= HOOP_TOLERANCE:
        return "Swish!"
    if miss < 0:
        return "Short"
    return "Long"


def trajectory_points(angle: float, speed: float, height: float,
                      gravity: float = GRAVITY
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Lists of x and y positions along the ball's whole flight (for the graph)."""
    vx = speed * np.cos(np.radians(angle))
    vy = speed * np.sin(np.radians(angle))
    # When the ball hits the floor (y = 0)
    ground_time = (vy + np.sqrt(vy ** 2 + 2 * gravity * height)) / gravity
    times = np.linspace(0, ground_time, 100)  # 100 moments, evenly spaced
    return vx * times, height + vy * times - 0.5 * gravity * times ** 2


def validate_inputs(angle: float, speed: float, height: float,
                    distance: float, defender_height: float,
                    defender_distance: float) -> str:
    """Check every number. Gives a friendly error message, or "" if all good."""
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


def swish_angles(speed: float, height: float, distance: float, gravity: float,
                 defender_on: bool, defender_height: float,
                 defender_distance: float) -> list[float]:
    """Every launch angle (checked in 0.1 degree steps) that would swish."""
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
    """Squish a list of angles into [low, high] groups that touch each other."""
    ranges = []
    for angle in angles:
        # An angle right next to the last one joins the same group
        if len(ranges) > 0 and angle - ranges[-1][1] <= ANGLE_STEP * 1.5:
            ranges[-1][1] = angle
        else:
            ranges.append([angle, angle])
    return ranges


def make_hint(speed: float, height: float, distance: float, gravity: float,
              defender_on: bool, defender_height: float,
              defender_distance: float) -> str:
    """The coach's advice, as a sentence."""
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


def hint_for(state: State) -> str:
    """Ask the coach about whatever setup the player has right now."""
    return make_hint(state.current_speed, state.current_height,
                     state.current_distance, PLANETS[state.planet],
                     state.defender_on, state.defender_height,
                     state.defender_distance)


def record_shot(state: State) -> None:
    """Take the shot with the current setup and add it to the history."""
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


def take_message(state: State) -> str:
    """Hand over the note for the player, and clear it so it only shows once."""
    note = state.message
    state.message = ""
    return note


def apply_shot_inputs(state: State, angle_text: str, speed_text: str,
                      height_text: str) -> bool:
    """Save the typed angle, speed and height. False (plus a note) if they're bad."""
    angle = parse_number(angle_text)
    speed = parse_number(speed_text)
    height = parse_number(height_text)
    error = validate_inputs(angle, speed, height, state.current_distance,
                            state.defender_height, state.defender_distance)
    if error != "":
        state.message = error
        return False
    state.current_angle = angle
    state.current_speed = speed
    state.current_height = height
    state.hint = ""  # the old advice no longer fits
    return True


def apply_defender_inputs(state: State, defender: bool, height_text: str,
                          distance_text: str) -> bool:
    """Save the defender settings. False (plus a note) if they're bad."""
    defender_height = parse_number(height_text)
    defender_distance = parse_number(distance_text)
    error = validate_inputs(state.current_angle, state.current_speed,
                            state.current_height, state.current_distance,
                            defender_height, defender_distance)
    if error != "":
        state.message = error
        return False
    state.defender_on = defender
    state.defender_height = defender_height
    state.defender_distance = defender_distance
    state.hint = ""
    return True


def new_round(state: State) -> None:
    """Move the hoop and maybe add a defender for the next challenge shot."""
    state.current_distance = random.choice(CHALLENGE_DISTANCES)
    state.defender_on = random.random() < DEFENDER_CHANCE
    state.defender_height = round(random.uniform(1.8, 2.3), 2)
    # Always in front of the hoop, so the defender can really matter
    state.defender_distance = round(
        random.uniform(1.0, state.current_distance * 0.7), 1)
    state.message = ""
    state.hint = ""


# ---------------------------------------------------------------------------
# STEP 4: Pages
# ---------------------------------------------------------------------------
@route
def index(state: State) -> Page:
    """Home screen: your setup at a glance, plus a button for everything."""
    defender_line = "🛡️ Defender: off"
    if state.defender_on:
        defender_line = (f"🛡️ Defender: {state.defender_height} m tall, "
                         f"{state.defender_distance} m from you")
    return Page(state, [
        Header("🏀 Basketball Shot Predictor"),
        "Set up your shot, then hit Shoot!",
        take_message(state),
        state.hint,
        f"🪐 Planet: {state.planet}",
        f"📏 Hoop distance: {state.current_distance} m",
        f"🎯 Your shot: {state.current_angle}° at {state.current_speed} m/s, "
        f"released {state.current_height} m high",
        defender_line,
        Button("🏀 Shoot!", shoot),
        Button("🧠 Coach Hint", coach_hint),
        Button("🪐 Planet", planet_page),
        Button("📏 Distance", distance_page),
        Button("🎯 My Shot", shot_page),
        Button("🛡️ Defender", defender_page),
        Button("🎨 Graph Look", look_page),
        Button("🏆 Challenge Mode", start_challenge),
        Button("📜 Shot History", show_history),
        Button("📊 Stats", show_stats),
        Button("❓ How to Play", help_page),
    ])


@route
def help_page(state: State) -> Page:
    """A short guide to the game."""
    return Page(state, [
        Header("How to Play"),
        "1. Pick a planet, a distance, and your shot from the home screen.",
        "2. Press Shoot! to see if it goes in.",
        "3. Stuck? Ask the Coach. It tells you which angles would score.",
        "4. Try Challenge Mode: 5 shots, a moving hoop, and surprise defenders.",
        "Points for a made shot: 1 from the free throw line, 2 from "
        "mid-range, 3 from three-point range, and 5 for a really long one.",
        Button("Back", index),
    ])


@route
def planet_page(state: State) -> Page:
    """Choose which planet you're playing on."""
    lines = []
    # LOOP 6: one line of info per planet
    for name, gravity in PLANETS.items():
        lines.append(f"{name} (gravity {gravity}): {PLANET_BLURBS[name]}")
    return Page(state, [
        Header("🪐 Pick a Planet"),
        take_message(state),
        *lines,
        "Planet:",
        SelectBox("planet_choice", list(PLANETS), state.planet),
        Button("Save", save_planet),
        Button("Back", index),
    ])


@route
def save_planet(state: State, planet_choice: str) -> Page:
    """Remember the chosen planet and go home."""
    state.planet = planet_choice
    state.hint = ""
    state.message = f"Now playing on {planet_choice}!"
    return index(state)


@route
def distance_page(state: State) -> Page:
    """Choose how far away the hoop is."""
    return Page(state, [
        Header("📏 Pick a Distance"),
        take_message(state),
        "How far from the hoop?",
        SelectBox("distance_choice",
                  list(DISTANCE_OPTIONS) + [CUSTOM_CHOICE],
                  distance_label(state.current_distance)),
        "Custom distance in meters (only used if you pick 'Custom'):",
        TextBox("custom_text", str(state.current_distance)),
        Button("Save", save_distance),
        Button("Back", index),
    ])


@route
def save_distance(state: State, distance_choice: str,
                  custom_text: str) -> Page:
    """Remember the distance (if it makes sense) and go home."""
    distance = get_distance(distance_choice, custom_text)
    error = validate_inputs(state.current_angle, state.current_speed,
                            state.current_height, distance,
                            state.defender_height, state.defender_distance)
    if error != "":
        state.message = error
        return distance_page(state)
    state.current_distance = distance
    state.hint = ""
    state.message = f"Hoop is now {distance} m away."
    return index(state)


@route
def shot_page(state: State) -> Page:
    """Choose the angle, speed, and release height of your shot."""
    return Page(state, [
        Header("🎯 Your Shot"),
        take_message(state),
        "Launch angle (1-89 degrees):",
        TextBox("angle_text", str(state.current_angle)),
        "Ball speed (m/s, no limit):",
        TextBox("speed_text", str(state.current_speed)),
        "Release height (m):",
        TextBox("height_text", str(state.current_height)),
        Button("Save", save_shot),
        Button("Save and Shoot!", save_and_shoot),
        Button("Back", index),
    ])


@route
def save_shot(state: State, angle_text: str, speed_text: str,
              height_text: str) -> Page:
    """Remember the shot and go home."""
    if not apply_shot_inputs(state, angle_text, speed_text, height_text):
        return shot_page(state)
    state.message = "Shot saved."
    return index(state)


@route
def save_and_shoot(state: State, angle_text: str, speed_text: str,
                   height_text: str) -> Page:
    """Remember the shot and fire it right away."""
    if not apply_shot_inputs(state, angle_text, speed_text, height_text):
        return shot_page(state)
    return shoot(state)


@route
def defender_page(state: State) -> Page:
    """Turn the defender on or off and place them."""
    return Page(state, [
        Header("🛡️ Defender"),
        take_message(state),
        CheckBox("defender", state.defender_on),
        "Add a defender",
        "Defender height (m):",
        TextBox("defender_height_text", str(state.defender_height)),
        "How far the defender stands from you (m):",
        TextBox("defender_distance_text", str(state.defender_distance)),
        Button("Save", save_defender),
        Button("Back", index),
    ])


@route
def save_defender(state: State, defender: bool, defender_height_text: str,
                  defender_distance_text: str) -> Page:
    """Remember the defender settings and go home."""
    if not apply_defender_inputs(state, defender, defender_height_text,
                                 defender_distance_text):
        return defender_page(state)
    state.message = "Defender saved."
    return index(state)


@route
def look_page(state: State) -> Page:
    """Choose how the graph looks."""
    return Page(state, [
        Header("🎨 Graph Look"),
        take_message(state),
        "Colors:",
        SelectBox("style_choice", list(GRAPH_STYLES), state.graph_style),
        "Ghost trails (how many older shots to show faintly):",
        SelectBox("ghost_choice", GHOST_CHOICES, str(state.ghost_count)),
        Button("Save", save_look),
        Button("Back", index),
    ])


@route
def save_look(state: State, style_choice: str, ghost_choice: str) -> Page:
    """Remember the graph choices and go home."""
    state.graph_style = style_choice
    state.ghost_count = int(ghost_choice)
    state.message = "Graph look saved."
    return index(state)


@route
def shoot(state: State) -> Page:
    """Take a shot with the current setup."""
    state.hint = ""
    record_shot(state)
    return show_result(state)


@route
def coach_hint(state: State) -> Page:
    """Ask the coach which angles would score with the current setup."""
    state.hint = hint_for(state)
    return index(state)


@route
def show_result(state: State) -> Page:
    """Show what happened on the latest shot."""
    latest = state.shot_history[-1]
    content = [
        Header("Result: " + latest.outcome),
        OUTCOME_LINES[latest.outcome],
        f"Planet: {latest.planet} (gravity {latest.gravity} m/s²)",
        f"Peak height: {latest.peak_height} m",
        f"Missed the hoop by: {latest.distance_from} m",
        f"Points earned: {latest.points}",
    ]
    if state.challenge_on:
        content.append(f"Challenge score: {state.challenge_score}")
    content.append(Button("📈 See Graph", show_graph))
    if not state.challenge_on:
        content.append(Button("🏀 Shoot Again", shoot))
        content.append(Button("🎯 Tweak My Shot", shot_page))
        content.append(Button("🏠 Home", index))
    elif state.challenge_shots_left == 0:
        content.append(Button("🏁 Final Score", challenge_over))
    else:
        content.append(Button("➡️ Next Shot", next_challenge_shot))
    return Page(state, content)


@route
def show_graph(state: State) -> Page:
    """Draw the latest shot, with faint ghosts of the shots before it."""
    colors = GRAPH_STYLES[state.graph_style]
    latest = state.shot_history[-1]
    earlier = state.shot_history[-1 - state.ghost_count:-1]
    plt.close("all")  # clear old graphs so memory doesn't pile up
    figure, axes = plt.subplots()
    figure.patch.set_facecolor(colors["background"])
    axes.set_facecolor(colors["background"])
    # LOOP 4: draw each older shot as a faded dashed line
    for position, ghost in enumerate(earlier):
        ghost_x, ghost_y = trajectory_points(
            ghost.angle, ghost.speed, ghost.height, ghost.gravity)
        fade = 0.15 + 0.45 * (position + 1) / len(earlier)  # older = fainter
        label = "Earlier shots" if position == 0 else "_nolegend_"
        axes.plot(ghost_x, ghost_y, color=colors["ghost"], linestyle="--",
                  alpha=fade, label=label)
    x_values, y_values = trajectory_points(
        latest.angle, latest.speed, latest.height, latest.gravity)
    axes.plot(x_values, y_values, color=colors["path"], linewidth=3,
              label="Latest shot")
    axes.scatter([0], [latest.height], color=colors["path"], zorder=3)
    # The rim, drawn as a short thick line at hoop height
    axes.plot([latest.distance - 0.23, latest.distance + 0.23],
              [HOOP_HEIGHT, HOOP_HEIGHT], color=colors["hoop"],
              linewidth=5, label="Hoop")
    right_edge = max(float(x_values.max()), latest.distance)
    if state.defender_on:
        axes.plot([state.defender_distance, state.defender_distance],
                  [0, state.defender_height], color=DEFENDER_COLOR,
                  linewidth=5, label="Defender")
        right_edge = max(right_edge, state.defender_distance)
    axes.set_ylim(bottom=0)
    axes.set_xlim(0, right_edge * 1.05)  # leave a little room on the right
    axes.grid(True, color=colors["floor"], alpha=0.3)
    axes.tick_params(colors=colors["text"])
    for edge in axes.spines.values():
        edge.set_color(colors["floor"])
    axes.set_xlabel("Distance from shooter (m)", color=colors["text"])
    axes.set_ylabel("Height (m)", color=colors["text"])
    axes.set_title("Shot on " + latest.planet, color=colors["text"])
    legend = axes.legend(facecolor=colors["background"],
                         edgecolor=colors["floor"])
    for words in legend.get_texts():
        words.set_color(colors["text"])
    return Page(state, [
        Header("Shot Trajectory"),
        MatPlotLibPlot(),
        Button("Back to Result", show_result),
    ])


@route
def show_history(state: State) -> Page:
    """A table of every shot so far."""
    rows = [["Shot #", "Planet", "Outcome", "Peak height (m)",
             "Missed by (m)", "Points"]]
    # LOOP 1: one table row per shot
    for number, shot in enumerate(state.shot_history, start=1):
        rows.append([str(number), shot.planet, shot.outcome,
                     str(shot.peak_height), str(shot.distance_from),
                     str(shot.points)])
    return Page(state, [
        Header("📜 Shot History"),
        Table(rows),
        Button("Clear History", reset_history),
        Button("Back", index),
    ])


@route
def show_stats(state: State) -> Page:
    """Shooting percentage, points, and other numbers."""
    made = 0
    total_peak = 0.0
    total_points = 0
    # LOOP 2: count the made shots and add up peak heights and points
    for shot in state.shot_history:
        if shot.outcome == "Swish!":
            made += 1
        total_peak += shot.peak_height
        total_points += shot.points
    attempts = len(state.shot_history)
    percentage = 0.0
    average_peak = 0.0
    if attempts > 0:  # can't divide by zero shots
        percentage = round(made / attempts * 100, 1)
        average_peak = round(total_peak / attempts, 2)
    return Page(state, [
        Header("📊 Stats"),
        f"Shots taken: {attempts}",
        f"Shots made: {made}",
        f"Shooting percentage: {percentage}%",
        f"Total points: {total_points}",
        f"Average peak height: {average_peak} m",
        f"Best challenge score: {state.challenge_best}",
        Button("Back", index),
    ])


@route
def reset_history(state: State) -> Page:
    """Wipe the shot history."""
    state.shot_history = []
    state.message = "History cleared."
    return index(state)


# ---- Challenge mode ----
@route
def start_challenge(state: State) -> Page:
    """Start a 5-shot challenge on the planet you picked."""
    if not state.challenge_on:
        # Keep your free-play setup safe so we can give it back later
        state.saved_setup = [state.current_distance, state.defender_on,
                             state.defender_height, state.defender_distance]
    state.challenge_on = True
    state.challenge_shots_left = CHALLENGE_SHOTS
    state.challenge_score = 0
    new_round(state)
    return challenge_page(state)


@route
def challenge_page(state: State) -> Page:
    """The challenge screen: this round's hoop, plus your shot controls."""
    shot_number = CHALLENGE_SHOTS - state.challenge_shots_left + 1
    defender_line = "No defender this round."
    if state.defender_on:
        defender_line = (f"Defender: {state.defender_height} m tall, "
                         f"{state.defender_distance} m from you.")
    return Page(state, [
        Header("🏆 Challenge Mode"),
        f"Shot {shot_number} of {CHALLENGE_SHOTS}   |   "
        f"Score: {state.challenge_score}",
        f"Planet: {state.planet}",
        f"Hoop: {state.current_distance} m away "
        f"({point_value(state.current_distance)} points if you score)",
        defender_line,
        take_message(state),
        state.hint,
        "Launch angle (1-89 degrees):",
        TextBox("angle_text", str(state.current_angle)),
        "Ball speed (m/s):",
        TextBox("speed_text", str(state.current_speed)),
        "Release height (m):",
        TextBox("height_text", str(state.current_height)),
        Button("🏀 Fire!", challenge_shot),
        Button("🧠 Coach Hint", challenge_hint),
        Button("Quit Challenge", end_challenge),
    ])


@route
def challenge_shot(state: State, angle_text: str, speed_text: str,
                   height_text: str) -> Page:
    """Fire one challenge shot (a bad number doesn't cost you a shot)."""
    if not apply_shot_inputs(state, angle_text, speed_text, height_text):
        return challenge_page(state)
    record_shot(state)
    state.challenge_score += state.shot_history[-1].points
    state.challenge_shots_left -= 1
    return show_result(state)


@route
def challenge_hint(state: State, angle_text: str, speed_text: str,
                   height_text: str) -> Page:
    """Coach advice for this round."""
    if not apply_shot_inputs(state, angle_text, speed_text, height_text):
        return challenge_page(state)
    state.hint = hint_for(state)
    return challenge_page(state)


@route
def next_challenge_shot(state: State) -> Page:
    """Set up the next round."""
    new_round(state)
    return challenge_page(state)


@route
def challenge_over(state: State) -> Page:
    """The final score screen."""
    new_best = state.challenge_score > state.challenge_best
    if new_best:
        state.challenge_best = state.challenge_score
    summary = f"Best score: {state.challenge_best}"
    if new_best:
        summary = "🎉 New personal best!"
    return Page(state, [
        Header("🏁 Challenge Complete!"),
        f"Final score: {state.challenge_score}",
        summary,
        Button("Play Again", start_challenge),
        Button("Back to Home", end_challenge),
    ])


@route
def end_challenge(state: State) -> Page:
    """Leave the challenge and get your free-play setup back."""
    if state.challenge_on and len(state.saved_setup) == 4:
        state.current_distance = state.saved_setup[0]
        state.defender_on = state.saved_setup[1]
        state.defender_height = state.saved_setup[2]
        state.defender_distance = state.saved_setup[3]
    state.saved_setup = []
    state.challenge_on = False
    state.hint = ""
    return index(state)


# ---------------------------------------------------------------------------
# STEP 5: Tests (these run every time the program starts)
# ---------------------------------------------------------------------------
MOON = PLANETS["Moon"]
JUPITER = PLANETS["Jupiter"]

# -- The basics --
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
# blocks it, a 0.5 m one doesn't, and a defender behind the hoop doesn't
assert_equal(judge_shot(5, 8.0, 2.0, 4.6, True, 2.1, 1.5), "Blocked")
assert_equal(judge_shot(5, 8.0, 2.0, 4.6, True, 0.5, 1.5), "Short")
assert_equal(judge_shot(5, 8.0, 2.0, 4.6, True, 2.1, 10.0), "Short")
# A defender standing where the ball already landed doesn't block it
assert_equal(defender_blocks(5, 8.0, 2.0, 100.0, 50.0, 2.1), False)
# A hoop 500 m away is allowed (this shot just falls short)
assert_equal(judge_shot(52, 7.6, 2.0, 500.0, False, 2.1, 1.5), "Short")
# Speed has no cap, so a huge speed sends the ball way past a close hoop
assert_equal(judge_shot(52, 5000.0, 2.0, 4.6, False, 2.1, 1.5), "Long")
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 2.1, 1.5), "")
assert_equal(validate_inputs(52, 5000.0, 2.0, 500.0, 2.1, 300.0), "")
assert_equal(validate_inputs(0, 7.6, 2.0, 4.6, 2.1, 1.5) != "", True)
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 0.0, 1.5) != "", True)
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 2.1, 0.0) != "", True)
assert_equal(validate_inputs(52, 7.6, 2.0, 4.6, 5000.0, 9e9), "")  # no real cap

# -- Planets: the same shot acts very differently --
assert_equal(judge_shot(52, 7.6, 2.0, 4.6, False, 2.1, 1.5, MOON), "Long")
assert_equal(judge_shot(52, 7.6, 2.0, 4.6, False, 2.1, 1.5, JUPITER), "Short")
assert_equal(time_at_hoop(52, 7.6, 2.0, JUPITER), -1.0)  # never reaches the rim
assert_equal(round(peak_height(90, 1.62, 2.0, MOON), 2), 2.81)
moon_x, moon_y = trajectory_points(52, 7.6, 2.0, MOON)
assert_equal(round(float(moon_y[-1]), 6), 0.0)  # ends on the floor
assert_equal(float(moon_x.max()) > 30.0, True)  # and flies much farther

# -- Points --
assert_equal(point_value(4.6), 1)
assert_equal(point_value(6.0), 2)
assert_equal(point_value(7.2), 3)
assert_equal(point_value(9.0), 3)
assert_equal(point_value(14.0), 5)

# -- The coach --
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

# -- Shooting --
test = new_state()
fired = shoot(test)
assert_equal(len(fired.state.shot_history), 1)
assert_equal(fired.state.shot_history[0].outcome, "Swish!")
assert_equal(fired.state.shot_history[0].points, 1)
assert_equal(shoot(fired.state).state.shot_history[1].outcome, "Swish!")  # Shoot Again
assert_equal(reset_history(fired.state).state.shot_history, [])
assert_equal(show_history(new_state()).state.shot_history, [])
assert_equal(show_stats(new_state()).state.shot_history, [])

# -- The customization pages --
test = new_state()
assert_equal(save_planet(test, "Moon").state.planet, "Moon")
assert_equal(shoot(test).state.shot_history[0].outcome, "Long")  # the Moon is floaty
assert_equal(test.shot_history[0].gravity, MOON)

test = new_state()
save_distance(test, CUSTOM_CHOICE, "250")
assert_equal(test.current_distance, 250.0)
save_distance(test, "Mid-range (6.0 m)", "")
assert_equal(test.current_distance, 6.0)
save_distance(test, CUSTOM_CHOICE, "abc")  # bad input is refused
assert_equal(test.current_distance, 6.0)
assert_equal(test.message != "", True)

test = new_state()
save_shot(test, "40", "9", "1.5")
assert_equal(test.current_angle, 40.0)
assert_equal(test.current_speed, 9.0)
assert_equal(test.current_height, 1.5)
save_shot(test, "0", "9", "1.5")  # angle 0 is refused
assert_equal(test.current_angle, 40.0)
big = save_and_shoot(new_state(), "52", "5000", "2.0")  # no speed cap
assert_equal(big.state.shot_history[0].outcome, "Long")
refused = save_and_shoot(new_state(), "abc", "7.6", "2.0")
assert_equal(len(refused.state.shot_history), 0)

test = new_state()
save_defender(test, True, "3.0", "2.5")
assert_equal(test.defender_on, True)
assert_equal(test.defender_height, 3.0)
assert_equal(test.defender_distance, 2.5)
save_defender(test, False, "-1", "2.5")  # a bad height is refused
assert_equal(test.defender_on, True)

test = new_state()
save_look(test, "Sunset", "3")
assert_equal(test.graph_style, "Sunset")
assert_equal(test.ghost_count, 3)

# The coach button: gives advice but doesn't take a shot
advice = coach_hint(new_state())
assert_equal(advice.state.hint.startswith("Coach: try"), True)
assert_equal(len(advice.state.shot_history), 0)
jupiter = new_state()
save_planet(jupiter, "Jupiter")
assert_equal(coach_hint(jupiter).state.hint.startswith("Coach: no angle"), True)

# A note shows once, then goes away
note = new_state()
note.message = "hello"
index(note)
assert_equal(note.message, "")

# The pages that only show things should still build without crashing
shown = new_state()
assert_equal(planet_page(shown).state, shown)
assert_equal(distance_page(shown).state, shown)
assert_equal(shot_page(shown).state, shown)
assert_equal(defender_page(shown).state, shown)
assert_equal(look_page(shown).state, shown)
assert_equal(help_page(shown).state, shown)

# -- Challenge mode --
started = start_challenge(new_state())
assert_equal(started.state.challenge_on, True)
assert_equal(started.state.challenge_shots_left, CHALLENGE_SHOTS)
assert_equal(started.state.challenge_score, 0)
assert_equal(started.state.current_distance in CHALLENGE_DISTANCES, True)
round_state = new_state()
# LOOP 5: a defender must always stand in front of the hoop
for _ in range(50):
    new_round(round_state)
    assert_equal(0 < round_state.defender_distance
                 < round_state.current_distance, True)

# A made-up challenge: one swish (1 point), one miss, then the final score
challenge = new_state()
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
rejected_state = new_state()
rejected_state.challenge_on = True
rejected_state.challenge_shots_left = 3
rejected = challenge_shot(rejected_state, "0", "7.6", "2.0")
assert_equal(rejected.state.challenge_shots_left, 3)  # a bad number is free
hinted = challenge_hint(new_state(), "52", "7.6", "2.0")
assert_equal(hinted.state.hint.startswith("Coach: try"), True)

# Leaving a challenge (even after Play Again) gives back your free-play setup
keeper = new_state()
save_distance(keeper, "Mid-range (6.0 m)", "")
save_defender(keeper, True, "3.0", "2.5")
start_challenge(keeper)
start_challenge(keeper)  # like pressing Play Again
end_challenge(keeper)
assert_equal(keeper.challenge_on, False)
assert_equal(keeper.current_distance, 6.0)
assert_equal(keeper.defender_on, True)
assert_equal(keeper.defender_height, 3.0)
assert_equal(keeper.defender_distance, 2.5)
# index, show_result and show_graph show data (show_graph also draws a
# picture), so we check those by playing the site by hand.


# ---------------------------------------------------------------------------
# STEP 6: Start the website
# ---------------------------------------------------------------------------
apply_look()
set_site_information(
    author="Derek Bellin",
    description="Predicts whether a basketball shot goes in using physics.",
    sources=["https://en.wikipedia.org/wiki/Projectile_motion","LLM Documentation (2).pdf"],
    planning=["Copy of CS1 Final Project 1_ Design.pdf",
              "Copy of CS1 Final Project 0_ Pick a Problem.pdf"],
    links=["https://derekb-udel.github.io/Accuracy-Shot-Site/"],
)
start_server(new_state())
