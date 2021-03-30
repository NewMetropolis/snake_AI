from colour import Color
import math
import numpy as np
import pygame
import random
import sys


class SnakeGame:

    def __init__(self, display_width=600, display_height=440, snake_block=20, snake_speed=5,
                 ai_mode=False, theme='default'):

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
        # The Snake itself. In variables below 'ri' stands for 'row index' and 'ci' for 'column index'.
        self.ci_snake = []
        self.ri_snake = []
        self.ci_head = None
        self.ri_head = None
        self.snake_length = 1
        self.ci_move = None
        self.ri_move = None
        self.ci_food = None
        self.ri_food = None
        self.snake_speed = snake_speed
        # Possible moves for a Snake.
        self.ci_changes = [1, 0, -1, 0]
        self.ri_changes = [0, -1, 0, 1]
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
        self.ci_snake = []
        self.ri_snake = []
        # Position Snake at the middle of a board.
        c_middle = self.n_cols / 2 - 1 if self.n_cols % 2 == 0 else self.n_cols // 2
        r_middle = self.n_rows / 2 - 1 if self.n_rows % 2 == 0 else self.n_rows // 2
        self.ci_head = c_middle
        self.ri_head = r_middle
        self.ci_move = 0
        self.ri_move = 0
        self.snake_length = 1

    def get_ci_ri_food(self):
        collides_snake = True
        while collides_snake:
            self.ci_food = random.randrange(0, self.n_cols)
            self.ri_food = random.randrange(0, self.n_rows)

            collides_snake = False
            for segment in range(self.snake_length):
                if self.ci_snake[segment] == self.ci_food and self.ri_snake[segment] == self.ri_food:
                    collides_snake = True
        return

    def directions_order(self):
        distances = np.empty(4)
        for direction in range(4):
            distances = np.linalg.norm([self.ci_food[direction] - self.ci_snake[direction], self.ri_food[direction] -
                                        self.ri_snake[direction]])
        order = np.argsort(distances)

        self.food_coding.fill(0)
        self.food_coding[np.argmin(distances)] = 1

        return order

    def check_obstacles(self):
        self.obstacles_coding.fill(0)
        for direction in range(4):
            new_ci = self.ci_head + self.ci_changes[direction]
            new_ri = self.ri_head + self.ri_changes[direction]
            if new_ci < 0 or new_ci >= self.n_cols:
                self.obstacles_coding[direction] = 1
            if new_ri < 0 or new_ri >= self.n_rows:
                self.obstacles_coding[direction] = 1
            for segment in range(self.snake_length):
                if self.ci_snake[segment] == new_ci and self.ri_snake[segment] == new_ri:
                    self.obstacles_coding[direction] = 1

        if self.obstacles_coding.sum() == 4:
            print('Shit: {}'.format(self.score))

    def navigate_snake_bool_logic(self):
        self.move_coding.fill(0)
        for direction in self.directions_order():
            if self.obstacles_coding[direction] == 0:
                self.move_coding = 1

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
        self.ci_move = self.ci_changes[move_direction]
        self.ri_move = self.ri_changes[move_direction]

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
            if event.key == pygame.K_RIGHT:
                self.ci_move = 1
                self.ri_move = 0
            elif event.key == pygame.K_LEFT:
                self.ci_move = -1
                self.ri_move = 0
            elif event.key == pygame.K_UP:
                self.ci_move = 0
                self.ri_move = -1
            elif event.key == pygame.K_DOWN:
                self.ci_move = 0
                self.ri_move = 1

    def get_random_control_events(self):
        # For simplification we discard a backward move as only valid for a snake of a length 1.
        self.move_coding.fill(0)
        possible_moves = [0, 1, 2, 3]
        random_move = random.choice(possible_moves)
        self.move_coding[random_move] = 1

        return

    def update_snake_position(self):
        if self.snake_length > 1:
            self.ci_snake[1:] = self.ci_snake[:-1]
            self.ri_snake[1:] = self.ri_snake[:-1]
        self.ci_head += self.ci_move
        self.ri_head += self.ri_move
        self.ci_snake = self.ci_head
        self.ri_snake = self.ri_head

    def check_if_snake_inside(self):
        crash = False
        if self.ci_head < 0 or self.ci_head >= self.n_cols:
            crash = True
        if self.ri_head < 0 or self.ri_head >= self.n_rows:
            crash = True

        if crash:
            if self.ai_mode:
                self.game_over = True

                return

            else:
                self.closing_game = True

                return

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
            pygame.draw.rect(self.display, self.snake_color,
                             [segment[0], segment[1], self.snake_block,
                              self.snake_block])
            if count > 0:
                if segment[0] == self.snake[0][0] and segment[1] == self.snake[0][1]:
                    if self.ai_mode:
                        self.game_over = True

                        return

                    else:
                        self.closing_game = True

                        return

        pygame.draw.rect(self.display, self.food_color, [self.x_food, self.y_food, self.snake_block, self.snake_block])
        self.display_score()
        pygame.display.update()

    def snake_eats_food(self):
        if self.snake[0][0] == self.x_food and self.snake[0][1] == self.y_food:
            self.length += 1
            self.snake.append([0, 0])
            self.get_x_y_food()
            self.score += 10

    # def gather_data(self):
    #     self.food_coding()
    #     self.coding.append(list(self.obstacles) + list(self.moves) + list(self.food))

    def game_loop(self):
        self.set_display()
        self.reset_snake()
        self.get_x_y_food()
        self.score = 0
        while not self.game_over:
            self.get_control_events()
            self.update_snake_position()
            self.check_if_snake_inside()
            self.draw_snake_food()
            self.snake_eats_food()
            self.clock.tick(self.snake_speed)
            self.closing_game_dialog()
        pygame.quit()

    def if_game_loop(self, iterations):
        for i in range(iterations):
            self.set_display()
            self.reset_snake()
            self.get_x_y_food()
            self.score = 0
            self.game_over = False
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                self.check_obstacles()
                self.navigate_snake_bool_logic()
                # self.navigate_snake_regression()
                self.move_snake()
                self.gather_data()
                self.update_snake_position()
                self.check_if_snake_inside()
                self.draw_snake_food()
                self.snake_eats_food()
                self.clock.tick(self.snake_speed)

            self.scores.append(self.score)
        pygame.quit()

        return self.scores, self.step_coding

    def drbm_game_loop(self, iterations, drbm):
        for i in range(iterations):
            self.set_display()
            self.reset_snake()
            self.get_x_y_food()
            self.score = 0
            self.game_over = False
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                self.check_obstacles()
                self.get_food_vector()
                self.food_coding()
                # self.navigate_snake_regression()
                self.moves.fill(0)
                prediction = drbm.predict(np.hstack([self.obstacles, self.food]))
                self.moves[np.argmax(prediction)] = 1
                self.move_snake()
                self.gather_data()
                self.update_snake_position()
                self.check_if_snake_inside()
                self.draw_snake_food()
                self.snake_eats_food()
                self.clock.tick(self.snake_speed)

            self.scores.append(self.score)
        pygame.quit()

        return self.scores, self.step_coding

    def random_game_loop(self, iterations):
        for i in range(iterations):
            self.set_display()
            self.reset_snake()
            self.get_x_y_food()
            self.score = 0
            self.game_over = False
            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game_over = True
                self.get_random_control_events()
                self.move_snake()
                self.update_snake_position()
                self.check_if_snake_inside()
                self.draw_snake_food()
                self.snake_eats_food()
                self.clock.tick(self.snake_speed)

            self.scores[i] = self.score
        pygame.quit()

        return self.scores, self.step_coding
