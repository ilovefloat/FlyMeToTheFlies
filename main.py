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
PLAYER_SPEED = 2.8
ATTACK_DISTANCE = 0.95
CATCH_DISTANCE = 0.16
MOUSE_SENSITIVITY = 0.0025
RENDER_WIDTH = 900
RENDER_HEIGHT = 600


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-9 else v / n


def camera_quaternion(yaw: float, pitch: float) -> np.ndarray:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    return np.array([cp * cy, -sp * sy, sp * cy, cp * sy], dtype=float)


def camera_forward(yaw: float, pitch: float) -> np.ndarray:
    cp = math.cos(pitch)
    return np.array([cp * math.sin(yaw), cp * math.cos(yaw), math.sin(pitch)], dtype=float)


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

        world.add(
            "camera", name="game_camera",
            pos=(0.0, -1.0, 1.35),
            quat=tuple(camera_quaternion(0.0, 0.0)),
            fovy=70,
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
    font = pygame.font.Font(None, 28)
    big_font = pygame.font.Font(None, 64)

    env = build_environment()
    camera_id = env.physics.model.name2id("game_camera", "camera")

    player_position = np.array([0.0, -1.0, 1.35], dtype=float)
    yaw = 0.0
    pitch = 0.0
    caught = False
    running = True
    attacking = 0.0

    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and caught:
                    env.reset()
                    caught = False
                    attacking = 0.0
                elif event.key == pygame.K_F1:
                    grabbed = not pygame.event.get_grab()
                    pygame.event.set_grab(grabbed)
                    pygame.mouse.set_visible(not grabbed)
            elif event.type == pygame.MOUSEMOTION and pygame.event.get_grab():
                yaw += event.rel[0] * MOUSE_SENSITIVITY
                pitch -= event.rel[1] * MOUSE_SENSITIVITY
                pitch = float(np.clip(pitch, -1.35, 1.35))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                attacking = 0.12

        if not caught:
            keys = pygame.key.get_pressed()
            forward = camera_forward(yaw, 0.0)
            right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
            move = np.zeros(3, dtype=float)

            if keys[pygame.K_w]:
                move += forward
            if keys[pygame.K_s]:
                move -= forward
            if keys[pygame.K_d]:
                move += right
            if keys[pygame.K_a]:
                move -= right
            if keys[pygame.K_e]:
                move[2] += 1.0
            if keys[pygame.K_q]:
                move[2] -= 1.0

            if np.any(move):
                player_position += normalize(move) * PLAYER_SPEED * dt

            player_position[0] = np.clip(player_position[0], -ROOM_HALF_X + 0.15, ROOM_HALF_X - 0.15)
            player_position[1] = np.clip(player_position[1], -ROOM_HALF_Y + 0.15, ROOM_HALF_Y - 0.15)
            player_position[2] = np.clip(player_position[2], 0.18, ROOM_HEIGHT - 0.18)

            env.physics.named.model.cam_pos[camera_id] = player_position
            env.physics.named.model.cam_quat[camera_id] = camera_quaternion(yaw, pitch)
            env.physics.forward()

            action = np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)
            env.step(action)

            if attacking > 0.0:
                fly_position = get_body_position(env.physics, "walker/thorax")
                aim = camera_forward(yaw, pitch)
                to_fly = fly_position - player_position
                along = float(np.dot(to_fly, aim))
                if 0.0 < along < ATTACK_DISTANCE:
                    closest = player_position + aim * along
                    if np.linalg.norm(fly_position - closest) <= CATCH_DISTANCE:
                        caught = True
                attacking = max(0.0, attacking - dt)

        frame = env.physics.render(width=RENDER_WIDTH, height=RENDER_HEIGHT, camera_id=camera_id)
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, screen.get_size())
        screen.blit(surface, (0, 0))

        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        pygame.draw.line(screen, (245, 245, 245), (cx - 8, cy), (cx + 8, cy), 2)
        pygame.draw.line(screen, (245, 245, 245), (cx, cy - 8), (cx, cy + 8), 2)

        status = (
            "CAUGHT!  Press R to reset"
            if caught
            else "WASD: move  |  Mouse: look  |  LMB: swat  |  ESC: quit"
        )
        screen.blit(font.render(status, True, (245, 245, 245)), (18, 16))

        if caught:
            message = big_font.render("FLY CAUGHT", True, (255, 240, 120))
            screen.blit(message, message.get_rect(center=(WINDOW_WIDTH // 2, 90)))

        pygame.display.flip()

    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    pygame.quit()


if __name__ == "__main__":
    main()
