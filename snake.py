import bfs
import sys
import math
import numpy as np
import pygame
import random
"""Implementation of a classic '00s Snake game."""

class SnakeGame:
    def __init__(self, display_width=600, display_height=440, snake_block=20, snake_speed=5, ai_mode=False,
                 theme='default'):
        # Graphics settings.
        self.display_width = display_width
        self.display_height = display_height
        self.snake_block = snake_block
        self.display = None
        self.theme = theme
        pygame.font.init()
        if self.theme == 'default':
            self.background_color = pygame.Color("white")
            self.snake_color = pygame.Color("blue")
            self.snake_head_color = pygame.Color("red")
            self.food_color = pygame.Color("green")
            self.message_color = pygame.Color("black")
            self.line_color = pygame.Color("black")
            self.font_size = 30
            self.font_style = pygame.font.SysFont('arial', self.font_size)
        # A top bar to display a current score.
        self.top_bar_height = math.ceil(self.font_size / snake_block) * snake_block
        # Grid settings.
        self.n_rows = (display_height - self.top_bar_height) / snake_block
        self.n_cols = display_width / snake_block
        if self.n_rows % 1 != 0. or self.n_cols % 1 != 0.:
            sys.exit('Game board dimensions must by a multiple of a Snake block size.')
        self.n_rows = int(self.n_rows)
        self.n_cols = int(self.n_cols)
        # Grid.
        self.grid = np.full([self.n_rows, self.n_cols], 1, dtype=int)
        # Snake itself.
        self.snake = [np.empty(2, dtype=int)]
        self.snake_head = np.empty(2, dtype=int)
        self.snake_length = 1
        self.move = np.empty(2, dtype=int)
        self.food = np.empty(2, dtype=int)
        self.snake_speed = snake_speed
        # Possible moves for a Snake.
        self.directions = np.array(list(zip([0, -1, 0, 1], [1, 0, -1, 0])), dtype=int)
        # Obstacles, food and moves coding.
        self.obstacles_coding = np.zeros(4, dtype=int)
        self.food_coding = np.zeros(4, dtype=int)
        self.move_coding = np.zeros(4, dtype=int)
        # This will store all three together. I use here four objects instead of just one for clarity reasons.
        self.step_coding = []
        # For controlling a game loop.
        # 'Game over' does not mean end of the fun. One can always try again.
        self.game_over = False
        self.closing_game = False
        self.clock = pygame.time.Clock()
        self.score = 0

        # Last but not least.
        self.ai_mode = ai_mode
        if self.ai_mode:
            self.steps_coding = []
            self.scores = []
            if self.ai_mode == 'if_statement':
                self.ordered_directions = np.empty(4, dtype=int)

        return

    def set_game_board(self):
        # Initialize a game board.
        pygame.display.init()
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Atak dzikich wenszy.')

        return

    def display_centered_text(self, text, offset=0):
        # Display centered text.
        text_rendered = self.font_style.render(text, True, self.message_color, self.background_color)
        text_rect = text_rendered.get_rect(center=(self.display_width / 2, self.display_height / 2 + offset))
        self.display.blit(text_rendered, text_rect)

        return

    def display_score(self):
        # Display score.
        message_rendered = self.font_style.render("Score: {}".format(self.score), True, self.message_color,
                                                  self.background_color)
        self.display.blit(message_rendered, [0, 0])

        return

    def get_control_events(self):
        # Capture keyboard events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.move = np.array([0, 1])
                elif event.key == pygame.K_LEFT:
                    self.move = np.array([0, -1])
                elif event.key == pygame.K_UP:
                    self.move = np.array([-1, 0])
                elif event.key == pygame.K_DOWN:
                    self.move = np.array([1, 0])
 
    def reset_snake(self):
        # After the game is over reset snake before the next turn.
        self.snake = [np.empty(2, dtype=int)]
        # Position Snake in the middle of a board.
        r_middle = self.n_rows / 2 - 1 if self.n_rows % 2 == 0 else self.n_rows // 2
        c_middle = self.n_cols / 2 - 1 if self.n_cols % 2 == 0 else self.n_cols // 2
        self.snake_head[:] = [r_middle, c_middle]
        self.move.fill(0)
        self.snake_length = 1
        self.score = 0

        return

    def place_food(self):
        # Place food randomly on a game board.
        collides_snake = True
        while collides_snake:
            self.food[0] = random.randrange(0, self.n_rows)
            self.food[1] = random.randrange(0, self.n_cols)
            collides_snake = False
            for segment in self.snake:
                if np.all(segment == self.food):
                    collides_snake = True

        return

    def directions_order(self):
        # Order possible move directions based on a distance to food.
        distances = np.empty(4)
        for direction in range(4):
            # Could be optimized better. There is already such a loop.
            new_position = self.snake_head + self.directions[direction, :]
            distances[direction] = np.linalg.norm(self.food - new_position)
        self.ordered_directions = np.argsort(distances)
        self.food_coding.fill(0)
        self.food_coding[np.argmin(distances)] = 1

        return

    def check_obstacles(self):
        # Check obstacles around the Snake's head.
        self.obstacles_coding.fill(0)
        for direction in range(4):
            new_position = self.snake_head + self.directions[direction, :]
            if new_position[0] < 0 or new_position[0] >= self.n_rows:
                self.obstacles_coding[direction] = 1
            if new_position[1] < 0 or new_position[1] >= self.n_cols:
                self.obstacles_coding[direction] = 1
            for segment in self.snake:
                if np.all(segment == new_position):
                    self.obstacles_coding[direction] = 1

        if self.obstacles_coding.sum() == 4:
            print('We are xxxxxx!')
            self.game_over = True

        return

    def navigate_snake_bool_logic(self):
        # Once obstacles and directions order have been established navigate the Snake.
        self.move_coding.fill(0)
        for direction in self.ordered_directions:
            if self.obstacles_coding[direction] == 0:
                self.move_coding[direction] = 1

                return

    # def navigate_snake_regression(self):
    #     self.moves.fill(0)
    #     self.place_food_vector()
    #     y = np.empty(4, dtype=int)
    #
    #     for i in range(4):
    #         y[i] = (1 - 2 * self.obstacles[i]) * math.pi - math.acos(
    #             np.dot(self.food_vector, self.all_directions_vec[i]))
    #
    #     self.moves[np.argmax(y)] = 1
    #
    #     return

    def update_grid(self):
        # Reset.
        self.grid.fill(1)
        # Update grid by marking places where the Snake is with zeros.
        for segment in self.snake:
            if segment.sum() != -2:
                self.grid[tuple(segment)] = 0

        return

    def set_move_value(self):
        # Set move value(row index, column index change) based on a current move_coding.
        try:
            move_direction = np.argwhere(self.move_coding == 1).item()
            self.move = self.directions[move_direction, :].copy()
        except ValueError:
            print('The final score is: {}.'.format(self.score))

        return

    # def get_random_control_events(self):
    #     # For simplification we discard a backward move as only valid for a snake of a length 1.
    #     self.move_coding.fill(0)
    #     possible_moves = [0, 1, 2, 3]
    #     random_move = random.choice(possible_moves)
    #     self.move_coding[random_move] = 1
    #
    #     return

    def update_snake_position(self):
        # Update snake coordinates based on a current move.
        if self.snake_length > 1:
            self.snake[1:] = self.snake[:-1]
        self.snake_head = self.snake_head + self.move
        self.snake[0] = self.snake_head

        return

    def check_if_crashed(self):
        # Check it Snake crashed.
        crash = False
        if self.snake_head[0] < 0 or self.snake_head[0] >= self.n_rows:
            crash = True
        if self.snake_head[1] < 0 or self.snake_head[1] >= self.n_cols:
            crash = True

        if self.snake_length > 1:
            for segment in range(1, self.snake_length):
                if np.all(self.snake[segment] == self.snake_head):
                    crash = True
                    break
        if crash:
            if self.ai_mode:
                self.game_over = True
            else:
                self.closing_game = True

        return

    def snake_eats_food(self):
        # If Snake catches food, make it one segment longer.
        if np.all(self.snake_head == self.food):
            self.snake_length += 1
            self.snake.append(np.full(2, -1, dtype=int))
            self.place_food()
            self.score += 10

    def draw_snake_food(self):
        # Draw Snake and food on a game board.
        self.display.fill(self.background_color)
        pygame.draw.line(self.display, self.line_color, [0, self.top_bar_height - 1], [self.display_width,
                                                                                       self.top_bar_height - 1])
        for count, segment in enumerate(self.snake):
            # Counterintuitive. Yet we are on a grid([row_id, col_id]).
            y_snake = self.top_bar_height + segment[0] * self.snake_block
            x_snake = segment[1] * self.snake_block
            if count == 0:
                pygame.draw.rect(self.display, self.snake_head_color,
                                 [x_snake, y_snake, self.snake_block,
                                  self.snake_block])
            else:
                pygame.draw.rect(self.display, self.snake_color,
                                 [x_snake, y_snake, self.snake_block,
                                  self.snake_block])

        x_food = self.food[1] * self.snake_block
        y_food = self.top_bar_height + self.food[0] * self.snake_block
        pygame.draw.rect(self.display, self.food_color, [x_food, y_food, self.snake_block, self.snake_block])
        self.display_score()
        pygame.display.update()

        return

    def closing_game_dialog(self):
        # After the Snake crashes offer to start again.
        while self.closing_game:
            self.display.fill(self.background_color)
            self.display_centered_text('You have lost!')
            self.display_centered_text('Press Q-quit or C-play again.', offset=self.font_size)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.game_over = True
                        self.closing_game = False
                    if event.key == pygame.K_c:
                        pygame.display.quit()
                        self.closing_game = False
                        self.game_loop()

        return

    # def gather_data(self):
    #     self.food_coding()
    #     self.coding.append(list(self.obstacles) + list(self.moves) + list(self.food))

    def game_loop(self):
        self.set_game_board()
        self.reset_snake()
        self.place_food()
        while not self.game_over:
            self.get_control_events()
            self.update_snake_position()
            self.check_if_crashed()
            self.draw_snake_food()
            self.snake_eats_food()
            self.clock.tick(self.snake_speed)
            self.closing_game_dialog()
        pygame.quit()

        return

    def if_game_loop(self, iterations):
        for i in range(iterations):
            self.game_over = False
            self.set_game_board()
            self.reset_snake()
            self.place_food()
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                self.check_obstacles()
                self.directions_order()
                self.navigate_snake_bool_logic()
                # self.navigate_snake_regression()
                self.set_move_value()
                # self.gather_data()
                self.update_snake_position()
                self.check_if_crashed()
                self.draw_snake_food()
                self.snake_eats_food()
                self.clock.tick(self.snake_speed)

            self.scores.append(self.score)
        pygame.quit()

        return

    def bfs_game_loop(self, iterations):
        for i in range(iterations):
            self.game_over = False
            self.set_game_board()
            self.reset_snake()
            self.place_food()
            self.update_snake_position()
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                self.update_grid()
                if self.snake_length == 1:
                    snake_ = self.snake
                else:
                    snake_ = self.snake[:-1]
                print('Calculating optimal route.')
                bfs_on_grid = bfs.BreadthFirstSearch(self.grid, self.snake_head.copy(), self.food.copy(), snake_.copy())
                bfs_on_grid.search()
                track_exists = bfs_on_grid.reconstruct_track()
                if not track_exists:
                    self.game_over = True
                    continue
                n_moves = len(bfs_on_grid.track)
                print("Number of moves: {}".format(n_moves))
                for ith_move in range(n_moves):
                    self.move = bfs_on_grid.track[ith_move] - self.snake[0]
                    self.update_snake_position()
                    self.check_if_crashed()
                    self.draw_snake_food()
                    self.snake_eats_food()
                    self.clock.tick(self.snake_speed)
            self.scores.append(self.score)
        pygame.quit()

        return
#
    # def drbm_game_loop(self, iterations, drbm):
    #     for i in range(iterations):
    #         self.set_game_board()
    #         self.reset_snake()
    #         self.place_food()
    #         self.score = 0
    #         self.game_over = False
    #         while not self.game_over:
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     self.game_over = True
    #             self.check_obstacles()
    #             # self.place_food_vector()
    #             self.food_coding()
    #             # self.navigate_snake_regression()
    #             self.moves.fill(0)
    #             prediction = drbm.predict(np.hstack([self.obstacles, self.food]))
    #             self.moves[np.argmax(prediction)] = 1
    #             self.set_move_value()
    #             self.gather_data()
    #             self.update_snake_position()
    #             self.check_if_snake_inside()
    #             self.draw_snake_food()
    #             self.snake_eats_food()
    #             self.clock.tick(self.snake_speed)
    #
    #         self.scores.append(self.score)
    #     pygame.quit()
    #
    #     return self.scores, self.step_coding
    #
    # def random_game_loop(self, iterations):
    #     for i in range(iterations):
    #         self.set_game_board()
    #         self.reset_snake()
    #         self.get_x_y_food()
    #         self.score = 0
    #         self.game_over = False
    #         while not self.game_over:
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     self.game_over = True
    #             self.get_random_control_events()
    #             self.set_move_value()
    #             self.update_snake_position()
    #             self.check_if_snake_inside()
    #             self.draw_snake_food()
    #             self.snake_eats_food()
    #             self.clock.tick(self.snake_speed)
    #
    #         self.scores[i] = self.score
    #     pygame.quit()
    #
    #     return self.scores, self.step_coding
