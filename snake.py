import math
import numpy as np
import pygame
import random
import sys


class SnakeGame:

    def __init__(self, display_width=600, display_height=430, snake_block=20, snake_speed=5, font_size=30,
                 ai_mode=False, coloring_scheme='default'):

        self.display_width = display_width
        self.display_height = display_height
        # This has to be changed.
        self.top_bar_height = font_size
        self.snake_block = snake_block
        # Grid dimensions.
        self.n_rows = (self.display_height - self.top_bar_height) / self.snake_block
        self.n_cols = self.display_width / self.snake_block
        if self.n_rows % 1 != 0. or self.n_cols % 1 != 0.:
            sys.exit('Game board dimensions must by a multiple of a Snake block size.')
        self.n_rows = int(self.n_rows)
        self.n_cols = int(self.n_cols)
        # Grid.
        self.grid = np.full([self.n_rows, self.n_cols], 1, dtype=int)
        self.snake_speed = snake_speed
        self.coloring_scheme = coloring_scheme
        pygame.font.init()
        self.font_size = font_size
        self.font_style = pygame.font.SysFont('arial', self.font_size)
        white = (255, 255, 255)
        black = (0, 0, 0)
        # red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        if self.coloring_scheme == 'default':
            self.background_color = white
            self.snake_color = blue
            self.food_color = green
            self.message_color = black
            self.line_color = black

        self.display = None
        self.clock = pygame.time.Clock()
        self.game_over = False
        self.closing_game = False
        self.snake = [[0, 0]]
        # Middle row and column index.
        r_middle = self.n_rows / 2 - 1 if self.n_rows % 2 == 0 else self.n_rows // 2
        c_middle = self.n_cols / 2 - 1 if self.n_cols % 2 == 0 else self.n_cols // 2
        self.head = [r_middle, c_middle]
        self.x_head_change = self.y_head_change = self.x_food = self.y_food = 0
        self.length = 1
        self.score = 0
        self.coding = []
        self.obstacles = np.zeros(4, dtype=int)
        self.food = np.zeros(4, dtype=int)
        self.moves = np.zeros(4, dtype=int)

        i_v = np.array([1., 0.])
        j_v = np.array([0., -1.])
        k_v = np.array([-1., 0.])
        l_v = np.array([0., 1.])
        self.all_directions_vec = [i_v, j_v, k_v, l_v]
        self.direction_vec_index_mapping = {
            str(i_v): 0,
            str(j_v): 1,
            str(k_v): 2,
            str(l_v): 3
        }

        self.food_vector = np.empty(2)
        self.ai_mode = ai_mode
        if self.ai_mode:
            self.step_coding = []
            self.scores = []

    # Function determines possible directions order for a snake, once given a vector pointing to food.
    # An order is set such that theoretically shortest paths come first.
    #
    # The order is established as follows:
    # 1) Calculate angles between the food vector and all four possible directions.
    # 2) Sort a list of angles in ascending order.
    # 3) Make a list of directions accordingly.
    #
    # Ties are handled by returning a list of lists for ordered directions. Directions with the same rank are stored in
    # the same inner list.
    # In fact, most often directions will be grouped in two pairs. That is because the snake is moving on
    # a grid. For every food location where the smallest angle is larger then 0.00 two directions closest to the vector
    # assure equally long routes.

    def directions_order(self):
        angles = np.zeros(4)
        food_vector = self.food_vector
        for count, direction in enumerate(self.all_directions_vec):
            angle = math.acos(np.dot(food_vector, direction))
            angles[count] = angle
        order = np.argsort(angles)
        angles = angles[order]
        directions_sorted = [self.direction_vec_index_mapping[str(self.all_directions_vec[index])] for index in order]
        ordered_directions = []
        if angles[0] == 0.:
            ordered_directions.append([directions_sorted[0]])
            ordered_directions.append([directions_sorted[1], directions_sorted[2]])
            ordered_directions.append([directions_sorted[3]])
        else:
            ordered_directions.append([directions_sorted[0], directions_sorted[1]])
            ordered_directions.append([directions_sorted[2], directions_sorted[3]])

        return ordered_directions

    def get_food_vector(self):
        food_direction_x = self.x_food - self.head[0]
        food_direction_y = self.y_food - self.head[1]
        food_vector = np.array([food_direction_x, food_direction_y])
        food_vector /= np.linalg.norm(food_vector)
        self.food_vector = np.round(food_vector, 2) + 0.

        return

    def food_coding(self):
        self.food.fill(0)
        angles = np.zeros(4)
        for count, direction in enumerate(self.all_directions_vec):
            angle = math.acos(np.dot(self.food_vector, direction))
            angles[count] = angle

        self.food[np.argmin(angles)] = 1

        return

    def check_obstacles(self):
        self.obstacles.fill(0)
        if self.head[0] + self.snake_block == self.display_width:
            self.obstacles[0] = 1
        if self.head[1] == self.top_bar_height:
            self.obstacles[1] = 1
        if self.head[0] == 0:
            self.obstacles[2] = 1
        if self.head[1] + self.snake_block == self.display_height:
            self.obstacles[3] = 1

        for count, segment in enumerate(self.snake):
            if count > 0:
                diff_x = self.head[0] - segment[0]
                diff_y = self.head[1] - segment[1]
                block = self.snake_block

                if diff_x == -block and diff_y == 0:
                    self.obstacles[0] = 1
                if diff_x == 0 and diff_y == block:
                    self.obstacles[1] = 1
                if diff_x == block and diff_y == 0:
                    self.obstacles[2] = 1
                if diff_x == 0 and diff_y == -block:
                    self.obstacles[3] = 1

        if self.obstacles.sum() == 4:
            print('Shit: {}'.format(self.score))

    def navigate_snake_bool_logic(self):
        self.moves.fill(0)
        self.get_food_vector()
        ordered_directions = self.directions_order()

        for element in ordered_directions:
            # if len(element) > 1:
            #     random.shuffle(element)
            for direction in element:
                index = direction
                if self.obstacles[index] == 0:
                    self.moves[index] = 1

                    return

    def navigate_snake_regression(self):
        self.moves.fill(0)
        self.get_food_vector()
        y = np.empty(4, dtype=int)

        for i in range(4):
            y[i] = (1 - 2 * self.obstacles[i]) * math.pi - math.acos(
                np.dot(self.food_vector, self.all_directions_vec[i]))

        self.moves[np.argmax(y)] = 1

        return

    # def navigate_snake_regression(self):
    #     self.moves.fill(0)
    #     food_vector = np.round(self.get_food_vector(), 2) + 0.
    #     ordered_directions = self.directions_order(food_vector)
    #
    #     for element in ordered_directions:
    #         if len(element) > 1:
    #             random.shuffle(element)
    #         for direction in element:
    #             index = direction
    #             if self.obstacles[index] == 0:
    #                 self.moves[index] = 1
    #
    #                 return

    def move_snake(self):
        if self.moves[0]:
            self.x_head_change = self.snake_block
            self.y_head_change = 0
        elif self.moves[1]:
            self.x_head_change = 0
            self.y_head_change = -self.snake_block
        elif self.moves[2]:
            self.x_head_change = -self.snake_block
            self.y_head_change = 0
        elif self.moves[3]:
            self.x_head_change = 0
            self.y_head_change = self.snake_block

    def reset_snake(self):
        self.snake = [[0, 0]]
        self.head = [self.display_width / 2, (self.display_height + self.top_bar_height) / 2]
        self.x_head_change = 0
        self.y_head_change = 0
        self.length = 1

    def get_x_y_food(self):

        collides_snake = True

        while collides_snake:

            self.x_food = round(random.randrange(0, self.display_width - self.snake_block) / self.snake_block) * \
                          self.snake_block
            self.y_food = round(random.randrange(0, self.display_height - self.top_bar_height - self.snake_block) /
                                self.snake_block) * self.snake_block + self.top_bar_height

            collides_snake = False
            for element in self.snake:
                if self.x_food == element[0] and self.y_food == element[1]:
                    collides_snake = True

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
                if event.key == pygame.K_LEFT:
                    self.x_head_change = -self.snake_block
                    self.y_head_change = 0
                elif event.key == pygame.K_RIGHT:
                    self.x_head_change = self.snake_block
                    self.y_head_change = 0
                elif event.key == pygame.K_UP:
                    self.x_head_change = 0
                    self.y_head_change = -self.snake_block
                elif event.key == pygame.K_DOWN:
                    self.x_head_change = 0
                    self.y_head_change = self.snake_block

    def get_random_control_events(self):
        # For simplification we discard a backward move as only valid for a snake of a length 1.
        self.moves.fill(0)
        possible_moves = [0, 1, 2, 3]
        random_move = random.choice(possible_moves)
        self.moves[random_move] = 1

        return

    def update_snake_position(self):
        if self.length > 1:
            self.snake[1:] = self.snake[:-1]
        self.head = [self.head[0] + self.x_head_change, self.head[1] + self.y_head_change]
        # self.head[1] += self.y_head_change
        self.snake[0] = self.head

    def check_if_snake_inside(self):
        if self.snake[0][0] >= self.display_width or self.snake[0][0] < 0 or self.snake[0][1] >= self.display_height \
                or self.snake[0][1] < self.top_bar_height:

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

    def gather_data(self):
        self.food_coding()
        self.coding.append(list(self.obstacles) + list(self.moves) + list(self.food))

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
