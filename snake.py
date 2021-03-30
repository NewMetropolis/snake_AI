import math
import numpy as np
import pygame
import random
import sys


class SnakeGame:

    def __init__(self, display_width=600, display_height=440, snake_block=20, snake_speed=5, ai_mode=False,
                 theme='default'):

        # Graphics settings.
        self.display_width = display_width
        self.display_height = display_height
        self.snake_block = snake_block
        self.theme = theme
        self.display = None
        # Initialize fonts.
        pygame.font.init()
        if self.theme == 'default':
            self.background_color = pygame.Color("white")
            self.snake_color = pygame.Color("blue")
            self.food_color = pygame.Color("green")
            self.message_color = pygame.Color("black")
            self.line_color = pygame.Color("black")
            self.font_size = 30
            self.font_style = pygame.font.SysFont('arial', self.font_size)
        # A top bar to display a current score.
        self.top_bar_height = math.ceil(self.font_size / self.snake_block) * self.snake_block
        # Grid settings.
        self.n_rows = (self.display_height - self.top_bar_height) / self.snake_block
        self.n_cols = self.display_width / self.snake_block
        if self.n_rows % 1 != 0. or self.n_cols % 1 != 0.:
            sys.exit('Game board dimensions must by a multiple of a Snake block size.')
        self.n_rows = int(self.n_rows)
        self.n_cols = int(self.n_cols)
        # Grid.
        self.grid = np.full([self.n_rows, self.n_cols], 1, dtype=int)
        # The Snake itself.
        self.snake = [np.empty(2, dtype=int)]
        self.snake_head = np.empty(2, dtype=int)
        self.snake_length = 1
        self.move = np.empty(2, dtype=int)
        self.food = np.empty(2, dtype=int)
        self.snake_speed = snake_speed
        # Possible moves for a Snake.
        self.directions = np.array(list(zip([1, 0, -1, 0], [0, -1, 0, 1])), dtype=int)
        # Obstacles, food and moves coding.
        self.obstacles_coding = np.zeros(4, dtype=int)
        self.food_coding = np.zeros(4, dtype=int)
        self.move_coding = np.zeros(4, dtype=int)
        # This will store all three together. I use here four objects instead of one for a clarity reasons.
        self.step_coding = []
        # For controlling a game loop.
        self.game_over = False
        self.closing_game = False
        self.clock = pygame.time.Clock()
        self.score = 0

        # Last but not least.
        self.ai_mode = ai_mode
        if self.ai_mode:
            self.steps_coding = []
            self.scores = []

    def reset_snake(self):
        self.snake = [[]]
        # Position Snake at the middle of a board.
        c_middle = self.n_cols / 2 - 1 if self.n_cols % 2 == 0 else self.n_cols // 2
        r_middle = self.n_rows / 2 - 1 if self.n_rows % 2 == 0 else self.n_rows // 2
        self.snake_head[:] = [c_middle, r_middle]
        self.move.fill(0)
        self.snake_length = 1
        self.score = 0

    def get_food(self):
        collides_snake = True
        while collides_snake:
            self.food[0] = random.randrange(0, self.n_cols)
            self.food[1] = random.randrange(0, self.n_rows)

            collides_snake = False
            for segment in self.snake:
                if np.all(segment == self.food):
                    collides_snake = True
        return

    def directions_order(self):
        distances = np.empty(4)
        for direction in range(4):
            distances = np.linalg.norm(self.food - self.directions[direction, :])
        order = np.argsort(distances)

        self.food_coding.fill(0)
        self.food_coding[np.argmin(distances)] = 1

        return order

    def check_obstacles(self):
        self.obstacles_coding.fill(0)
        for direction in range(4):
            new_position = self.snake_head + self.directions[direction, :]
            if new_position[0] < 0 or new_position[0] >= self.n_cols:
                self.obstacles_coding[direction] = 1
            if new_position[1] < 0 or new_position[1] >= self.n_rows:
                self.obstacles_coding[direction] = 1
            for segment in self.snake:
                if np.all(segment == new_position):
                    self.obstacles_coding[direction] = 1

        if self.obstacles_coding.sum() == 4:
            print('Shit: {}'.format(self.score))

    def navigate_snake_bool_logic(self):
        self.move_coding.fill(0)
        for direction in self.directions_order():
            if self.obstacles_coding[direction] == 0:
                self.move_coding[direction] = 1

                return

    # def navigate_snake_regression(self):
    #     self.moves.fill(0)
    #     self.get_food_vector()
    #     y = np.empty(4, dtype=int)
    # 
    #     for i in range(4):
    #         y[i] = (1 - 2 * self.obstacles[i]) * math.pi - math.acos(
    #             np.dot(self.food_vector, self.all_directions_vec[i]))
    # 
    #     self.moves[np.argmax(y)] = 1
    # 
    #     return

    def move_snake(self):
        move_direction = np.argwhere(self.move_coding == 1)
        self.move = self.directions[move_direction, :]

        return

    def set_display(self):
        pygame.display.init()
        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Atak dzikich wenszy.')

    def display_centered_text(self, text, offset=0):
        text_rendered = self.font_style.render(text, True, self.message_color, self.background_color)
        text_rect = text_rendered.get_rect(center=(self.display_width / 2, self.display_height / 2 + offset))
        self.display.blit(text_rendered, text_rect)

    def display_score(self):
        message_rendered = self.font_style.render("Score: {}".format(self.score), True, self.message_color,
                                                  self.background_color)
        self.display.blit(message_rendered, [0, 0])

    def get_control_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.move = np.array([1, 0])
                elif event.key == pygame.K_LEFT:
                    self.move = np.array([-1, 0])
                elif event.key == pygame.K_UP:
                    self.move = np.array([0, -1])
                elif event.key == pygame.K_DOWN:
                    self.move = np.array([0, 1])

    # def get_random_control_events(self):
    #     # For simplification we discard a backward move as only valid for a snake of a length 1.
    #     self.move_coding.fill(0)
    #     possible_moves = [0, 1, 2, 3]
    #     random_move = random.choice(possible_moves)
    #     self.move_coding[random_move] = 1
    #
    #     return

    def update_snake_position(self):
        if self.snake_length > 1:
            self.snake[1:] = self.snake[:-1]
        self.snake_head = self.snake_head + self.move
        self.snake[0] = self.snake_head

    def check_if_crashed(self):
        crash = False
        if self.snake_head[0] < 0 or self.snake_head[0] >= self.n_cols:
            crash = True
        if self.snake_head[1] < 0 or self.snake_head[1] >= self.n_rows:
            crash = True

        if self.snake_length > 1:
            for segment in self.snake:
                if np.all(segment == self.snake_head):
                    crash = True
                    break
        if crash:
            if self.ai_mode:
                self.game_over = True

            else:
                self.closing_game = True

        return

    def snake_eats_food(self):
        if np.all(self.snake_head == self.food):
            self.snake_length += 1
            self.snake.append(np.zeros(2, dtype=int))
            self.get_food()
            self.score += 10

    def closing_game_dialog(self):
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

    def draw_snake_food(self):
        self.display.fill(self.background_color)
        pygame.draw.line(self.display, self.line_color, [0, self.top_bar_height - 1], [self.display_width,
                                                                                       self.top_bar_height - 1])
        for count, segment in enumerate(self.snake):
            x_snake = segment[0] * self.snake_block
            y_snake = self.top_bar_height + segment[1] * self.snake_block
            pygame.draw.rect(self.display, self.snake_color,
                             [x_snake, y_snake, self.snake_block,
                              self.snake_block])

        x_food = self.food[0] * self.snake_block
        y_food = self.top_bar_height + self.food[1] * self.snake_block
        pygame.draw.rect(self.display, self.food_color, [x_food, y_food, self.snake_block, self.snake_block])
        self.display_score()
        pygame.display.update()

    # def gather_data(self):
    #     self.food_coding()
    #     self.coding.append(list(self.obstacles) + list(self.moves) + list(self.food))

    def game_loop(self):
        self.set_display()
        self.reset_snake()
        self.get_food()
        while not self.game_over:
            self.get_control_events()
            self.update_snake_position()
            self.check_if_crashed()
            self.draw_snake_food()
            self.snake_eats_food()
            self.clock.tick(self.snake_speed)
            self.closing_game_dialog()
        pygame.quit()

    def if_game_loop(self, iterations):
        for i in range(iterations):
            self.set_display()
            self.reset_snake()
            self.get_food()
            self.game_over = False
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                self.check_obstacles()
                self.navigate_snake_bool_logic()
                # self.navigate_snake_regression()
                self.move_snake()
                # self.gather_data()
                self.update_snake_position()
                self.check_if_crashed()
                self.draw_snake_food()
                self.snake_eats_food()
                self.clock.tick(self.snake_speed)

            self.scores.append(self.score)
        pygame.quit()

        return self.scores, self.step_coding
    #
    # def drbm_game_loop(self, iterations, drbm):
    #     for i in range(iterations):
    #         self.set_display()
    #         self.reset_snake()
    #         self.get_food()
    #         self.score = 0
    #         self.game_over = False
    #         while not self.game_over:
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     self.game_over = True
    #             self.check_obstacles()
    #             # self.get_food_vector()
    #             self.food_coding()
    #             # self.navigate_snake_regression()
    #             self.moves.fill(0)
    #             prediction = drbm.predict(np.hstack([self.obstacles, self.food]))
    #             self.moves[np.argmax(prediction)] = 1
    #             self.move_snake()
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
    #         self.set_display()
    #         self.reset_snake()
    #         self.get_x_y_food()
    #         self.score = 0
    #         self.game_over = False
    #         while not self.game_over:
    #             for event in pygame.event.get():
    #                 if event.type == pygame.QUIT:
    #                     self.game_over = True
    #             self.get_random_control_events()
    #             self.move_snake()
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
