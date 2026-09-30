from __future__ import annotations

import math
import random
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pygame
from dm_control import composer
from dm_control.locomotion.arenas import floors
from flybody.fruitfly import fruitfly
from flybody.tasks.flight_imitation import FlightImitationWBPG
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader
from flybody.tasks.task_utils import com2root


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750
ROOM_HALF_X = 2.0
ROOM_HALF_Y = 2.0
ROOM_HEIGHT = 2.2

CAMERA_START = np.array([0.0, -1.0, 1.35], dtype=float)
SWATTER_START = np.array([0.0, -1.0, 0.9], dtype=float)

CATCH_DISTANCE = 0.16
SWAT_DISTANCE = 0.95
MOUSE_SENSITIVITY = 0.0025
RENDER_WIDTH = 900
RENDER_HEIGHT = 600

SOURCE_SAMPLE_HZ = 5000.0
GAME_SAMPLE_HZ = 60.0
FLIGHT_DATA = (
    Path(__file__).resolve().parent
    / "data"
    / "flight-dataset_saccade-evasion_augmented.hdf5"
)

ROUTE_IDS = (
    "038", "201", "059", "057", "149", "177",
    "221", "210", "239", "265", "024", "034",
)

FLIGHT_STATE = "FLYING"
GROUNDED_STATE = "GROUNDED"
TAKEOFF_STATE = "TAKEOFF"
LANDING_STATE = "LANDING"


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm < 1e-9 else v / norm


def camera_forward(yaw: float, pitch: float) -> np.ndarray:
    cp = math.cos(pitch)
    return np.array(
        [cp * math.sin(yaw), cp * math.cos(yaw), math.sin(pitch)],
        dtype=float,
    )


def camera_quaternion(yaw: float, pitch: float) -> np.ndarray:
    forward = camera_forward(yaw, pitch)
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=float)
    up = np.cross(right, forward)
    rotation = np.column_stack((right, up, -forward))
    trace = float(np.trace(rotation))

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return np.array([
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        ])

    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        return np.array([
            (rotation[2, 1] - rotation[1, 2]) / s,
            0.25 * s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
        ])

    if rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        return np.array([
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[0, 1] + rotation[1, 0]) / s,
            0.25 * s,
            (rotation[1, 2] + rotation[2, 1]) / s,
        ])

    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
    return np.array([
        (rotation[1, 0] - rotation[0, 1]) / s,
        (rotation[0, 2] + rotation[2, 0]) / s,
        (rotation[1, 2] + rotation[2, 1]) / s,
        0.25 * s,
    ])


class Room(floors.Floor):
    def __init__(self) -> None:
        super().__init__(name="room")
        world = self.mjcf_model.worldbody

        wall_material = self.mjcf_model.asset.add(
            "material", name="wall_material", rgba=(0.72, 0.72, 0.72, 1.0)
        )
        floor_material = self.mjcf_model.asset.add(
            "material", name="floor_material", rgba=(0.22, 0.24, 0.27, 1.0)
        )
        ceiling_material = self.mjcf_model.asset.add(
            "material", name="ceiling_material", rgba=(0.60, 0.62, 0.66, 1.0)
        )

        for geom in self.ground_geoms:
            geom.material = floor_material
            geom.size = (ROOM_HALF_X, ROOM_HALF_Y, 0.05)

        wall_thickness = 0.06
        wall_height = ROOM_HEIGHT / 2

        for name, pos, size in (
            ("wall_x_min", (-ROOM_HALF_X, 0, wall_height),
             (wall_thickness, ROOM_HALF_Y + wall_thickness, wall_height)),
            ("wall_x_max", (ROOM_HALF_X, 0, wall_height),
             (wall_thickness, ROOM_HALF_Y + wall_thickness, wall_height)),
            ("wall_y_min", (0, -ROOM_HALF_Y, wall_height),
             (ROOM_HALF_X + wall_thickness, wall_thickness, wall_height)),
            ("wall_y_max", (0, ROOM_HALF_Y, wall_height),
             (ROOM_HALF_X + wall_thickness, wall_thickness, wall_height)),
            ("ceiling", (0, 0, ROOM_HEIGHT),
             (ROOM_HALF_X, ROOM_HALF_Y, wall_thickness)),
        ):
            world.add(
                "geom", name=name, type="box",
                pos=pos, size=size,
                material=ceiling_material if name == "ceiling" else wall_material,
            )

        world.add(
            "light", name="room_light",
            pos=(0.0, -0.5, 2.0),
            dir=(0.0, 0.15, -1.0),
            diffuse=(0.8, 0.8, 0.8),
            specular=(0.25, 0.25, 0.25),
        )

        world.add(
            "camera", name="game_camera",
            pos=tuple(CAMERA_START),
            quat=tuple(camera_quaternion(0.0, 0.0)),
            fovy=70,
        )


def build_environment():
    arena = Room()
    wbpg = WingBeatPatternGenerator()

    trajectory_loader = InferenceFlightTrajectoryLoader()

    task = FlightImitationWBPG(
        walker=fruitfly.FruitFly,
        arena=arena,
        wbpg=wbpg,
        traj_generator=trajectory_loader,
        terminal_com_dist=float("inf"),
        disable_legs=False,
        floor_contacts=False,
        initialize_qvel=False,
        force_actuators=False,
        joint_filter=0.0,
        time_limit=float("inf"),
        future_steps=0,
        trajectory_sites=False,
    )

    swatter = arena.mjcf_model.worldbody.add(
        "body", name="swatter", pos=tuple(SWATTER_START)
    )
    swatter.add("joint", name="swatter_free", type="free", damping=0.5)
    swatter.add(
        "geom", name="swatter_head", type="box",
        size=(0.075, 0.075, 0.015),
        rgba=(0.85, 0.85, 0.85, 1.0),
        contype=0, conaffinity=0,
    )
    swatter.add(
        "geom", name="swatter_handle", type="capsule",
        fromto=(0.0, 0.0, -0.015, 0.0, 0.0, -0.45),
        size=(0.025,),
        rgba=(0.15, 0.15, 0.15, 1.0),
        contype=0, conaffinity=0,
    )

    return composer.Environment(
        task=task,
        time_limit=float("inf"),
        strip_singleton_obs_buffer_dim=True,
    )


def _free_joint_qpos_address(physics) -> int:
    free_type = 0
    for joint_index, joint_type in enumerate(np.asarray(physics.model.jnt_type)):
        if int(joint_type) == free_type:
            return int(physics.model.jnt_qposadr[joint_index])
    raise RuntimeError("Could not find the fly free joint")


def set_fly_pose(physics, qpos: np.ndarray, position: np.ndarray) -> None:
    address = _free_joint_qpos_address(physics)
    free_qpos = physics.data.qpos[address:address + 7]
    free_qpos[:3] = position
    free_qpos[3:7] = qpos[3:7]
    physics.data.qvel[:] = 0.0


def set_swatter(physics, position: np.ndarray) -> None:
    physics.named.data.qpos["swatter_free"][:] = np.array(
        [position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0]
    )
    physics.named.data.qvel["swatter_free"][:] = 0.0


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    return quat / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def _interpolate_quaternion(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    b = b.copy()
    if float(np.dot(a, b)) < 0:
        b *= -1
    return _normalize_quaternion(a * (1.0 - alpha) + b * alpha)


class FlightReplay:
    """60 FPS kinematic replay of measured FlyBody flight trajectories."""

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing flight dataset: {path}\n"
                "Run download_motion_data.py first."
            )

        self.routes = []
        with h5py.File(path, "r") as dataset:
            timestep = float(dataset["timestep_seconds"][()])
            if not np.isclose(timestep, 1.0 / SOURCE_SAMPLE_HZ):
                raise ValueError(f"Unexpected flight dataset timestep: {timestep}")

            trajectories = dataset["trajectories"]
            for route_id in ROUTE_IDS:
                group = trajectories[route_id]
                qpos = np.asarray(group["com_qpos"], dtype=np.float64)
                if len(qpos) < 2:
                    continue

                root_qpos = com2root(qpos[:, :3], qpos[:, 3:7])
                relative_position = root_qpos[:, :3] - root_qpos[0, :3]
                self.routes.append({
                    "id": route_id,
                    "position": relative_position,
                    "quaternion": np.asarray(root_qpos[:, 3:7], dtype=np.float64),
                    "duration": (len(qpos) - 1) / SOURCE_SAMPLE_HZ,
                })

        if not self.routes:
            raise RuntimeError("No usable flight trajectories were found")

        self.position = np.zeros(3)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.route = None
        self.route_start = np.zeros(3)
        self.route_time = 0.0
        self.state = GROUNDED_STATE
        self.state_time = 0.0
        self.next_flight_delay = random.uniform(0.7, 2.0)

    def _choose_route(self) -> None:
        self.route = random.choice(self.routes)
        self.route_time = 0.0
        self.route_start = self.position.copy()

    def _sample_route(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        route = self.route
        source_position = route["position"]
        source_quaternion = route["quaternion"]

        sample = np.clip(seconds * SOURCE_SAMPLE_HZ, 0, len(source_position) - 1)
        lo = int(sample)
        hi = min(lo + 1, len(source_position) - 1)
        alpha = sample - lo

        position = (
            source_position[lo] * (1.0 - alpha)
            + source_position[hi] * alpha
        )
        quaternion = _interpolate_quaternion(
            source_quaternion[lo], source_quaternion[hi], alpha
        )
        return position, quaternion

    def update(self, dt: float) -> tuple[np.ndarray, np.ndarray, str]:
        self.state_time += dt

        if self.state == GROUNDED_STATE:
            if self.state_time >= self.next_flight_delay:
                self.state = TAKEOFF_STATE
                self.state_time = 0.0
                self._choose_route()

        elif self.state == TAKEOFF_STATE:
            progress = min(self.state_time / 0.45, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            self.position[2] = 0.16 + 0.95 * smooth
            self.quaternion = np.array([
                math.cos(math.radians(-47.5) / 2),
                0.0,
                math.sin(math.radians(-47.5) / 2),
                0.0,
            ])
            if progress >= 1.0:
                self.state = FLIGHT_STATE
                self.state_time = 0.0

        elif self.state == FLIGHT_STATE:
            self.route_time += dt
            local_position, quaternion = self._sample_route(self.route_time)
            self.position = self.route_start + local_position

            # Keep the measured motion inside the game room.
            margin = 0.35
            self.position[0] = np.clip(
                self.position[0], -ROOM_HALF_X + margin, ROOM_HALF_X - margin
            )
            self.position[1] = np.clip(
                self.position[1], -ROOM_HALF_Y + margin, ROOM_HALF_Y - margin
            )
            self.position[2] = np.clip(
                self.position[2] + 1.0, 0.45, ROOM_HEIGHT - 0.35
            )
            self.quaternion = quaternion

            if self.route_time >= self.route["duration"]:
                self.state = LANDING_STATE
                self.state_time = 0.0
                self.route_start = self.position.copy()

        elif self.state == LANDING_STATE:
            progress = min(self.state_time / 0.65, 1.0)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            self.position[2] = self.route_start[2] * (1.0 - smooth) + 0.16 * smooth
            if progress >= 1.0:
                self.state = GROUNDED_STATE
                self.state_time = 0.0
                self.next_flight_delay = random.uniform(0.8, 2.5)

        return self.position.copy(), self.quaternion.copy(), self.state


def fly_quaternion_from_dataset(quaternion: np.ndarray) -> np.ndarray:
    # Dataset stores MuJoCo quaternions as w,x,y,z.
    return _normalize_quaternion(quaternion)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Fly Me to the Flies")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    big_font = pygame.font.Font(None, 64)

    env = build_environment()
    camera_id = env.physics.model.name2id("game_camera", "camera")
    set_swatter(env.physics, SWATTER_START)

    replay = FlightReplay(FLIGHT_DATA)

    player_position = CAMERA_START.copy()
    yaw = 0.0
    pitch = 0.0
    caught = False
    running = True
    attacking = 0.0

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    # Initial fly pose.
    fly_position, fly_quaternion, fly_state = replay.update(0.0)
    set_fly_pose(env.physics, fly_quaternion_from_dataset(fly_quaternion), fly_position)
    env.physics.forward()

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                grabbed = not pygame.event.get_grab()
                pygame.event.set_grab(grabbed)
                pygame.mouse.set_visible(not grabbed)
            elif event.type == pygame.MOUSEMOTION and pygame.event.get_grab():
                yaw += event.rel[0] * MOUSE_SENSITIVITY
                pitch = float(np.clip(
                    pitch - event.rel[1] * MOUSE_SENSITIVITY,
                    -1.35,
                    1.35,
                ))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                attacking = 0.12

        if not caught:
            fly_position, fly_quaternion, fly_state = replay.update(dt)
            set_fly_pose(
                env.physics,
                fly_quaternion_from_dataset(fly_quaternion),
                fly_position,
            )

            env.physics.named.model.cam_pos[camera_id] = player_position
            env.physics.named.model.cam_quat[camera_id] = camera_quaternion(yaw, pitch)

            # No env.step(): this is now a kinematic replay, not a real-time
            # 0.1 ms/0.2 ms whole-body flight simulation.
            env.physics.forward()

            if attacking > 0.0:
                fly = fly_position
                aim = camera_forward(yaw, pitch)
                to_fly = fly - player_position
                along = float(np.dot(to_fly, aim))
                if 0.0 < along < SWAT_DISTANCE:
                    closest = player_position + aim * along
                    if np.linalg.norm(fly - closest) <= CATCH_DISTANCE:
                        caught = True
                attacking = max(0.0, attacking - dt)

        frame = env.physics.render(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            camera_id=camera_id,
        )
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        if surface.get_size() != screen.get_size():
            surface = pygame.transform.scale(surface, screen.get_size())
        screen.blit(surface, (0, 0))

        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        pygame.draw.line(
            screen, (245, 245, 245), (cx - 8, cy), (cx + 8, cy), 2
        )
        pygame.draw.line(
            screen, (245, 245, 245), (cx, cy - 8), (cx, cy + 8), 2
        )

        if caught:
            status = "CAUGHT!  ESC: quit"
        else:
            status = f"{fly_state}  |  LMB: swat  |  Mouse: look  |  ESC: quit"

        screen.blit(font.render(status, True, (245, 245, 245)), (18, 16))

        if caught:
            message = big_font.render(
                "FLY CAUGHT", True, (255, 240, 120)
            )
            screen.blit(
                message,
                message.get_rect(center=(WINDOW_WIDTH // 2, 90)),
            )

        pygame.display.flip()

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    pygame.quit()


if __name__ == "__main__":
    main()
