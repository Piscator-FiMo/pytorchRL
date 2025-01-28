from typing import Optional

import gymnasium as gym
import numpy as np
import pygame

from src.env.Labyrinth import Labyrinth

# Initialize the font module
pygame.font.init()
# Create a font object
font = pygame.font.Font(None, 24)


class GridWorldEnv(gym.Env):

    def __init__(self, labyrinth: Labyrinth):
        self.labyrinth = labyrinth

        # The size of the square grid
        self.render_mode = "invisible"
        self.window = None
        self.clock = None
        self.metadata = {"render_fps": 5, 'render_modes': ["invisible", "human", "rgb_array"]}
        self.steps = 0
        self.max_distance = np.abs(np.linalg.norm(
            np.array([self.labyrinth.columns - 1, self.labyrinth.rows - 1]) - np.array([0, 0]), ord=1
        ))

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Discrete(self.labyrinth.columns * self.labyrinth.rows)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0], dtype=np.int32),  # right
            1: np.array([0, -1], dtype=np.int32),  # up
            2: np.array([-1, 0], dtype=np.int32),  # left
            3: np.array([0, 1], dtype=np.int32),  # down
        }

    def _get_obs(self):
        return int(self.labyrinth.rows * self._agent_location[0] + self._agent_location[1])

    def _get_info(self):
        return {
            "distance": np.abs(np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )),
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.render_mode = options["render_mode"]
        self.steps = 0

        self.labyrinth.regenerate_start()
        self._agent_location = np.array(self.labyrinth.start.position, dtype=np.int32)
        self._target_location = np.array(self.labyrinth.end.position, dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        prev_obs = self._get_obs()
        prev_info = self._get_info()

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        new_location = self._agent_location + direction

        # if agent hits a wall, don't move
        if self.labyrinth.is_wall_at(new_location[0], new_location[1]):
            new_location = self._agent_location

        self._agent_location = new_location

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        self.steps += 1
        truncated = not terminated and self.steps >= self.labyrinth.columns * self.labyrinth.rows * 10
        reward = self.calculate_reward(terminated, truncated, self.steps, {"observation": prev_obs, "info": prev_info}, {
                                       "observation": self._get_obs(), "info": self._get_info()})
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def calculate_reward(self, terminated: bool, truncated: bool, steps_taken: int, previous: dict, current: dict):
        reward = 0

        if terminated:
            return 1

        if truncated:
            return -1

        if np.array_equal(previous["info"]["agent"], current["info"]["agent"]):
            reward -= 1

        reward -= 0.05

        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        cell_size = 50
        width = self.labyrinth.columns * cell_size
        height = self.labyrinth.rows * cell_size
        size = (width, height + font.get_height())

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(size)
        # The size of a single grid square in pixels
        canvas.fill((0, 0, 0))

        # Finally, add some gridlines
        for rows in self.labyrinth.tiles:
            for tile in rows:
                color = None
                if tile.value == '.':
                    color = (255, 255, 255)
                elif tile.value == 'S':
                    color = (0, 0, 255)
                elif tile.value == 'E':
                    color = (0, 255, 0)
                if color:
                    pygame.draw.rect(canvas, color, (cell_size * tile.x + 1, cell_size *
                                     tile.y - 1, cell_size - 2, cell_size - 2))

        # Now we draw the agent
        pygame.draw.circle(canvas, (255, 0, 0), (self._agent_location + 0.5) * cell_size, cell_size / 3)

        # Render the step count text
        step_count_text = font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        canvas.blit(step_count_text, (10, cell_size * self.labyrinth.rows))
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def draw_border_rect(self, canvas, pix_square_size, width, height, position):
        pygame.draw.rect(
            canvas,
            (50, 50, 50),
            pygame.Rect(
                pix_square_size * position,
                (width, height),
            ),
        )


if __name__ == "__main__":
    lab = Labyrinth(7, 7)
    env = GridWorldEnv(lab)
    env.reset(options={"render_mode": "human"})
    env.metadata["render_fps"] = 60
    env._render_frame()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT:
                    action = 0
                elif event.key == pygame.K_UP:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
                else:
                    continue

                observation, reward, terminated, truncated, info = env.step(action)
                print(f"Steps: {env.steps}, Reward: {reward}, Distance: {info['distance']}, observation: {observation}")
                if terminated or truncated:
                    env.reset(options={"render_mode": "human"})
        env._render_frame()
