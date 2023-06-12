import os

import gym
from gym import spaces
from gym import Env
from typing import Optional
import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gym.wrappers.record_video import RecordVideo
from pygame.math import Vector2
from mbse.models.environment_models.bicyclecar_model import BicycleCarReward, BicycleCarModel
import jax.numpy as jnp
import math
import pygame
from pygame import gfxdraw


SCALE = 6.0  # Track scale
PLAYFIELD = 2000 / SCALE
WINDOW_W = 1000
WINDOW_H = 800
GRASS_DIM = PLAYFIELD / 20.0
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
TRACK_DETAIL_STEP = 21 / SCALE
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class TrajectoryGraph:
    """A stock trading visualization using matplotlib made to render
      OpenAI gym environments"""

    def __init__(self, title="RC Car Trajectory"):
        # Create a figure on screen and set the title
        # Create top subplot for net worth axis
        self.pos = []
        self.fig, self.pos_axis = plt.subplots()
        self.fig.suptitle(title)
        self.pos_axis.set_xticks([])
        self.pos_axis.set_yticks([])
        self.rectangle_size = 0.05
        self.canvas = self.fig.canvas
        self._bg = None
        # self.fr_number = self.pos_axis.annotate(
        #     "0",
        #     (0, 1),
        #     xycoords="axes fraction",
        #     xytext=(0, 0),
        #     textcoords="offset points",
        #     ha="left",
        #     va="top",
        #     animated=True,
        # )
        (self.ln,) = self.pos_axis.plot(0.0, 0.0, animated=True)
        (self.pn,) = self.pos_axis.plot(0.0, 0.0, 'ro', animated=True)
        (self.gn,) = self.pos_axis.plot(0.0, 0.0, 'o', animated=True, color='gold', label='Goal Destination')
        # self.car_pos = self.pos_axis.annotate(f'x_pos: {0:.2f}, y_pos: {0:.2f}'.format(0.0, 0.0),
        #                                       (0, 1),
        #                                       xycoords="axes fraction",
        #                                       xytext=(0, 0.95),
        #                                       ha="left",
        #                                       va="top",
        #                                       animated=True, )

        self.legend = self.fig.legend()

        self._artists = []
        # self.add_artist(self.fr_number)
        self.add_artist(self.ln)
        self.add_artist(self.pn)
        self.add_artist(self.gn)
        # self.add_artist(self.car_pos)
        self.add_artist(self.legend)
        self.cid = self.canvas.mpl_connect("draw_event", self.on_draw)
        # Show the graph without blocking the rest of the program
        plt.show(block=False)
        plt.pause(0.001)

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def add_rectangle(self, car_pos):
        rect = patches.Rectangle(xy=(car_pos[0] - self.rectangle_size, car_pos[1] - self.rectangle_size / 2.0),
                                 angle=car_pos[2] * 180.0 / np.pi,
                                 width=self.rectangle_size * 2.0,
                                 height=self.rectangle_size,
                                 alpha=0.5,
                                 rotation_point='center',
                                 edgecolor='red',
                                 facecolor='none',
                                 animated=True,
                                 )
        self.pos_axis.add_patch(rect)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()
        #data = np.frombuffer(cv.tostring_rgb(), dtype=np.uint8)
        #data = data.reshape(cv.get_width_height()[::-1] + (3,))
        return None

    def render(self, current_step, car_pos, goal_pos, window_size=40):
        self.pos.append(car_pos)
        pos = np.asarray(self.pos)
        self.pos_axis.clear()
        window_start = 0
        step_range = range(window_start, current_step)
        theta, v_x, v_y = car_pos[2], car_pos[3], car_pos[4]
        p_x_dot = v_x * np.cos(theta) - v_y * np.sin(theta)
        p_y_dot = v_x * np.sin(theta) + v_y * np.cos(theta)
        self.pn.set_data(car_pos[0], car_pos[1])
        self.pn.set_marker(marker=[3, 0, car_pos[2] * 180.0 / np.pi - 90.0])
        self.gn.set_data(goal_pos[0], goal_pos[1])
        # self.fr_number.set_text("frame: {j}".format(j=current_step))
        # self.car_pos.set_text("x_pos: {x}, y_pos: {y}".format(x=np.around(car_pos[0], 2), y=np.around(car_pos[1], 2)))
        self.ln.set_data(pos[step_range, 0], pos[step_range, 1])
        min_y = min(min(pos[:, 1]), goal_pos[1])
        y_lim_min = min_y / 1.25 if min_y >= 0 else min_y * 1.25
        max_y = max(max(pos[:, 1]), goal_pos[1])
        y_lim_max = max_y / 1.25 if max_y < 0 else max_y * 1.25
        self.pos_axis.set_ylim(
            y_lim_min,
            y_lim_max)
        min_x = min(min(pos[:, 0]), goal_pos[0])
        x_lim_min = min_x / 1.25 if min_x >= 0 else min_x * 1.25
        max_x = max(max(pos[:, 0]), goal_pos[0])
        x_lim_max = max_x / 1.25 if max_x < 0 else max_x * 1.25
        self.pos_axis.set_xlim(
            x_lim_min,
            x_lim_max)
        self.pos_axis.set_xticks([])
        self.pos_axis.set_yticks([])

        data = self.update()

        return data

    def close(self):
        plt.close(self.canvas.figure)


class BicycleEnv(Env):

    def __init__(self,
                 dynamics_model: BicycleCarModel = BicycleCarModel(),
                 reward_model: BicycleCarReward = BicycleCarReward(),
                 _np_random: Optional[np.random.Generator] = None,
                 render_mode: str = 'rgb_array',
                 ):
        super(BicycleEnv).__init__()
        self.render_mode = render_mode
        self.reward_model = reward_model
        self.dynamics_model = dynamics_model
        self.x_target = np.asarray(self.reward_model.x_target)
        self.room_boundary = self.dynamics_model.params.room_boundary
        self.velocity_limit = self.dynamics_model.params.velocity_limit
        high = np.asarray([self.room_boundary,
                           self.room_boundary,
                           np.pi,
                           self.velocity_limit,
                           self.velocity_limit,
                           self.velocity_limit]
                          )
        low = -high
        self.observation_space = spaces.Box(
            high=high,
            low=low,
        )
        self.dim_state = (6,)
        self.dim_action = (2,)
        self.max_steering = self.dynamics_model.params.max_steering
        high = np.ones(2)
        low = -high
        self.action_space = spaces.Box(
            high=high,
            low=low,
        )
        self.state = np.zeros(self.dim_state, )
        self._np_random = _np_random
        self.current_step = 0
        # self.visualization = None
        # self.window_size = 40
        # scale = 4
        # self.screen_width = 128 * scale
        # self.map_width = 512 * scale
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # assets_path = os.path.join(current_dir, "assets/")
        # self.car_image = pygame.image.load(assets_path + "car.png")
        # self.background_image = pygame.image.load(assets_path + "background.png")
        # self.goal_image = pygame.image.load(assets_path + "goal_flag.png")
        # self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        # self.bg_color = self.np_random.uniform(0, 210, size=3)
        # self.grass_color = np.copy(self.bg_color)
        # idx = self.np_random.integers(3)
        # self.grass_color[idx] += 20

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        self.state = np.zeros(self.dim_state)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        state = jnp.asarray(self.state).reshape(-1, 6)
        action = jnp.asarray(action)
        next_state = self.dynamics_model.predict(state, action)
        reward = self.reward_model.predict(self.state, action)
        self.current_step += 1
        if self.render_mode == "human":
            self.render()
        self.state = np.asarray(next_state).reshape(6, -1)
        return next_state.squeeze(), np.asarray(reward).squeeze().item(), False, False, {}

    # def render(self):
    #     if self.render_mode is None:
    #         gym.logger.warn(
    #             "You are calling render method without specifying any render mode. "
    #             "You can specify the render_mode at initialization, "
    #             f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
    #         )
    #         return
    #     if self.visualization is None:
    #         self.visualization = TrajectoryGraph()
    #
    #     return self.visualization.render(self.current_step, self.state, self.goal_pos,
    #                                      window_size=min(self.current_step, self.window_size))

    # def draw(self):
    #     # Draw background
    #     self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
    #     pos = self.state[0:2]
    #     trans = Vector2(pos[0], pos[1])
    #     trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])
    #     background_scaled = pygame.transform.scale(self.background_image, (self.map_width, self.map_width))
    #
    #     self.screen.blit(background_scaled, trans)
    #     # Draw car
    #     heading = self.state[2]
    #     car_scaled = pygame.transform.scale(self.car_image, (int(500 * self.dynamics_model.params.l),
    #                                                          int(250 * self.dynamics_model.params.l)))
    #     car_rotated = pygame.transform.rotate(car_scaled, 0 * heading * 180 / np.pi)
    #     self.screen.blit(car_rotated, (self.screen_width // 2, self.screen_width // 2))
    #
    #     goal_scaled = pygame.transform.scale(self.goal_image, (50, 50))
    #     self.screen.blit(goal_scaled, (4 * self.goal_pos[0],
    #                                    4 * self.goal_pos[1]))
    #     self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
    #     self._render_road(SCALE, trans, heading)
    #
    #     pygame.display.flip()
    #
    # def _render_road(self, zoom, translation, angle):
    #     bounds = PLAYFIELD
    #     field = [
    #         (bounds, bounds),
    #         (bounds, -bounds),
    #         (-bounds, -bounds),
    #         (-bounds, bounds),
    #     ]
    #
    #     # draw background
    #     self._draw_colored_polygon(
    #         self.surf, field, self.bg_color, zoom, translation, angle, clip=False
    #     )
    #
    #     # draw grass patches
    #     grass = []
    #     for x in range(-20, 20, 2):
    #         for y in range(-20, 20, 2):
    #             grass.append(
    #                 [
    #                     (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
    #                     (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
    #                     (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
    #                     (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
    #                 ]
    #             )
    #     for poly in grass:
    #         self._draw_colored_polygon(
    #             self.surf, poly, self.grass_color, zoom, translation, angle
    #         )
    #
    # def _draw_colored_polygon(
    #         self, surface, poly, color, zoom, translation, angle, clip=True
    # ):
    #     poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
    #     poly = [
    #         (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
    #     ]
    #     # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
    #     # Instead of calculating exactly if the polygon and screen overlap,
    #     # we simply check if the polygon is in a larger bounding box whose dimension
    #     # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
    #     # diagonal length of an environment object
    #     if not clip or any(
    #             (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
    #             and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
    #             for coord in poly
    #     ):
    #         gfxdraw.aapolygon(self.surf, poly, color)
    #         gfxdraw.filled_polygon(self.surf, poly, color)


if __name__ == "__main__":
    def simulate_car(k_p=1, k_d=0.6, horizon=500):
        from gym.wrappers.time_limit import TimeLimit
        env = BicycleEnv(reward_model=BicycleCarReward(goal_pos=np.ones(3) * 20))
        env = TimeLimit(env=env, max_episode_steps=horizon)
        # env = RecordVideo(env, video_folder='./', episode_trigger=lambda x: True)
        x, _ = env.reset()
        goal = env.goal_pos
        x_traj = np.zeros([horizon, 2])
        for h in range(horizon):
            pos_error = goal[0:2] - x[0:2]
            goal_direction = np.arctan2(pos_error[1], pos_error[0])
            goal_dist = np.sqrt(pos_error[0] ** 2 + pos_error[1] ** 2)
            velocity = np.sqrt(x[3] ** 2 + x[4] ** 2)
            s = np.clip(0.1 * goal_direction, a_min=-1, a_max=1)
            d = np.clip(k_p * goal_dist - k_d * velocity, a_min=-1, a_max=1)
            u = np.asarray([s, d])
            x, reward, terminate, truncate, _ = env.step(u)
            if terminate or truncate:
                x, _ = env.reset()
        return x_traj


    x_traj = simulate_car()
    check = True
