from __future__ import annotations

import math

import numpy as np
import pygame
from dm_control import composer
from dm_control.locomotion.arenas import floors
from flybody.fruitfly import fruitfly
from flybody.tasks.flight_imitation import FlightImitationWBPG
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750
ROOM_HALF_X = 2.0
ROOM_HALF_Y = 2.0
ROOM_HEIGHT = 2.2

SWATTER_START = np.array([0.0, -1.0, 0.9], dtype=float)
SWATTER_SPEED = 1.8
CATCH_DISTANCE = 0.11


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-9 else v / n


def camera_xyaxes(camera_pos: np.ndarray, target: np.ndarray) -> tuple[float, ...]:
    forward = normalize(target - camera_pos)
    world_up = np.array([0.0, 0.0, 1.0])
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))
    return tuple(np.concatenate((right, up)))


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
        wall_z = wall_height

        world.add(
            "geom", name="wall_x_min", type="box",
            pos=(-ROOM_HALF_X, 0, wall_z),
            size=(wall_thickness, ROOM_HALF_Y + wall_thickness, wall_height),
            material=wall_material,
        )
        world.add(
            "geom", name="wall_x_max", type="box",
            pos=(ROOM_HALF_X, 0, wall_z),
            size=(wall_thickness, ROOM_HALF_Y + wall_thickness, wall_height),
            material=wall_material,
        )
        world.add(
            "geom", name="wall_y_min", type="box",
            pos=(0, -ROOM_HALF_Y, wall_z),
            size=(ROOM_HALF_X + wall_thickness, wall_thickness, wall_height),
            material=wall_material,
        )
        world.add(
            "geom", name="wall_y_max", type="box",
            pos=(0, ROOM_HALF_Y, wall_z),
            size=(ROOM_HALF_X + wall_thickness, wall_thickness, wall_height),
            material=wall_material,
        )
        world.add(
            "geom", name="ceiling", type="box",
            pos=(0, 0, ROOM_HEIGHT),
            size=(ROOM_HALF_X, ROOM_HALF_Y, wall_thickness),
            material=ceiling_material,
        )

        world.add(
            "light", name="room_light",
            pos=(0.0, -0.5, 2.0),
            dir=(0.0, 0.15, -1.0),
            diffuse=(0.8, 0.8, 0.8),
            specular=(0.25, 0.25, 0.25),
        )

        camera_pos = np.array([3.7, -5.2, 3.0])
        target = np.array([0.0, 0.0, 0.85])
        world.add(
            "camera", name="game_camera",
            pos=tuple(camera_pos),
            xyaxes=camera_xyaxes(camera_pos, target),
            fovy=55,
        )


def make_flight_trajectory() -> tuple[np.ndarray, np.ndarray]:
    dt = 0.002
    duration = 60.0
    n = int(duration / dt)
    t = np.arange(n, dtype=float) * dt

    radius_x = 1.15
    radius_y = 1.15
    omega = 0.45
    z = 1.05 + 0.25 * np.sin(0.23 * t)

    x = radius_x * np.cos(omega * t)
    y = radius_y * np.sin(omega * t)
    vx = -radius_x * omega * np.sin(omega * t)
    vy = radius_y * omega * np.cos(omega * t)
    vz = 0.25 * 0.23 * np.cos(0.23 * t)

    angle = math.radians(-47.5)
    q = np.tile(
        np.array([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0]),
        (n, 1),
    )

    qpos = np.column_stack((x, y, z, q))
    qvel = np.column_stack((
        vx, vy, vz, np.zeros(n), np.zeros(n), np.zeros(n)
    ))
    return qpos, qvel


def build_environment():
    arena = Room()

    trajectory_loader = InferenceFlightTrajectoryLoader()
    qpos, qvel = make_flight_trajectory()
    trajectory_loader.set_next_trajectory(qpos, qvel)

    wbpg = WingBeatPatternGenerator()

    task = FlightImitationWBPG(
        walker=fruitfly.FruitFly,
        arena=arena,
        wbpg=wbpg,
        traj_generator=trajectory_loader,
        terminal_com_dist=float("inf"),
        disable_legs=False,
        floor_contacts=True,
        initialize_qvel=True,
        force_actuators=False,
        joint_filter=0.0,
        time_limit=60.0,
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
        contype=1, conaffinity=1,
    )
    swatter.add(
        "geom", name="swatter_handle", type="capsule",
        fromto=(0.0, 0.0, -0.015, 0.0, 0.0, -0.45),
        size=(0.025,),
        rgba=(0.15, 0.15, 0.15, 1.0),
        contype=1, conaffinity=1,
    )

    return composer.Environment(
        task=task, time_limit=60.0, strip_singleton_obs_buffer_dim=True
    )


def set_swatter(physics, position: np.ndarray) -> None:
    physics.named.data.qpos["swatter_free"][:] = np.array(
        [position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0]
    )
    physics.named.data.qvel["swatter_free"][:] = 0.0


def get_body_position(physics, body_name: str) -> np.ndarray:
    return np.array(physics.named.data.xpos[body_name], dtype=float)


def reset_environment(env, swatter_position: np.ndarray):
    env.reset()
    set_swatter(env.physics, swatter_position)
    return swatter_position.copy()


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Fly Me to the Flies - prototype")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)
    big_font = pygame.font.Font(None, 64)

    env = build_environment()
    camera_id = env.physics.model.name2id("game_camera", "camera")

    swatter_position = reset_environment(env, SWATTER_START)
    caught = False
    running = True

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    swatter_position = reset_environment(env, SWATTER_START)
                    caught = False
                elif event.key == pygame.K_r and caught:
                    swatter_position = reset_environment(env, SWATTER_START)
                    caught = False

        keys = pygame.key.get_pressed()

        if not caught:
            move = np.zeros(3, dtype=float)
            if keys[pygame.K_w]:
                move[1] += 1.0
            if keys[pygame.K_s]:
                move[1] -= 1.0
            if keys[pygame.K_d]:
                move[0] += 1.0
            if keys[pygame.K_a]:
                move[0] -= 1.0
            if keys[pygame.K_e]:
                move[2] += 1.0
            if keys[pygame.K_q]:
                move[2] -= 1.0

            if np.any(move):
                swatter_position += normalize(move) * SWATTER_SPEED * dt

            swatter_position[0] = np.clip(
                swatter_position[0], -ROOM_HALF_X + 0.15, ROOM_HALF_X - 0.15
            )
            swatter_position[1] = np.clip(
                swatter_position[1], -ROOM_HALF_Y + 0.15, ROOM_HALF_Y - 0.15
            )
            swatter_position[2] = np.clip(
                swatter_position[2], 0.12, ROOM_HEIGHT - 0.12
            )
            set_swatter(env.physics, swatter_position)

            action = np.zeros(
                env.action_spec().shape, dtype=env.action_spec().dtype
            )
            env.step(action)

            fly_position = get_body_position(env.physics, "walker/thorax")
            if np.linalg.norm(fly_position - swatter_position) <= CATCH_DISTANCE:
                caught = True

        frame = env.physics.render(
            width=WINDOW_WIDTH, height=WINDOW_HEIGHT, camera_id=camera_id
        )
        surface = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        screen.blit(surface, (0, 0))

        status = (
            "CAUGHT!  Press R to release/reset"
            if caught
            else "WASD: move   Q/E: down/up   SPACE: reset swatter"
        )
        screen.blit(font.render(status, True, (245, 245, 245)), (20, 18))

        if caught:
            message = big_font.render("FLY CAUGHT", True, (255, 240, 120))
            screen.blit(
                message,
                message.get_rect(center=(WINDOW_WIDTH // 2, 80)),
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
