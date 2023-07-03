import pygame as pg
from pygame import image as img

screen_width = 400
screen_height = 400
map_cell_width = 20
map_cell_height = 20
game_cell_width = 80
game_cell_height = 80

img1 = img.load('obj1.png')
img2 = img.load('obj2.png')
obj1_map = pg.transform.scale(img1, (map_cell_width, map_cell_height))
obj2_map = pg.transform.scale(img2, (map_cell_width, map_cell_height))
obj1_game = pg.transform.scale(img1, (game_cell_width, game_cell_height))
obj2_game = pg.transform.scale(img2, (game_cell_width, game_cell_height))

COLORS = {0: 'black', 1: 'gray',
          2: 'green', 3: 'blue'}

#0: "floor", 1: "obj1", 2:"obj2", "3": "wall", "4":"gray"

class WoozleGame():
    def __init__(self):
        self.screen = pg.display.set_mode((screen_width, screen_height))
        self.reset()
        self.action_to_direction = {0: [0,1], 1:[0,-1], 2:[-1,0], 3:[1,0]}

    def reset(self):
        self.make_env()
        self.curr_mode = "map"

    def make_env(self):
        self.env = [[0]*20 for _ in range(20)]
        self.obj1_pos = (15,10)
        self.obj2_pos = (5,11)
        
        self.env[self.obj1_pos[0]][self.obj1_pos[1]] = 1
        self.env[self.obj2_pos[0]][self.obj2_pos[1]] = 2

    def switch_mode(self, mode=None):
        modes = ["game", "map"]
        if mode==None:
            self.curr_mode = modes[modes.index(self.curr_mode)-1]
            self.render()
        elif mode in ["game", "map"]:
            if mode != self.curr_mode:
                mode = self.curr_mode
                self.render()

    def render(self):
        if self.curr_mode == "game":
            self.render_game()
        else:
            self.render_map()
   
    def render_map(self):
        for r in range(len(self.env)):
            for c in range(len(self.env[r])):
                self.draw_square(r, c, self.env[r][c], map_cell_width, map_cell_height)

    def render_game(self):
        #gray except for squares all around self?
        (row, col) = self.obj1_pos
        for r in range(max(row-1, 0), min(row+1, len(self.env))):
            for c in range(max(col-1, 0), min(col+1, len(self.env[0]))):
                self.draw_square(r, c, self.env[r][c], game_cell_width, game_cell_height)

    def draw_square(self, row, col, color, w, h):
        if color==1:
            if self.curr_mode == "game":
                self.screen.blit(obj1_game, (row*w, col*h))  
            else:
                self.screen.blit(obj1_map, (row*w, col*h))  
        elif color==2:
            if self.curr_mode == "game":
                self.screen.blit(obj2_game, (row*w, col*h))  
            else:
                self.screen.blit(obj2_map, (row*w, col*h))  
        else:
            pg.draw.rect(self.screen, COLORS[color],  pg.Rect(row*w,col*h,w,h))

    def get_next_pos(self, dir):
        return [min(max(0, self.obj1_pos[0] + dir[0]), len(self.env)), min(max(0, self.obj1_pos[1] + dir[1]), len(self.env[0]))]

    def update_env(self, action):
        dir = self.action_to_direction[action]
        next_pos = self.get_next_pos(dir)
        self.env[self.obj1_pos[0]][self.obj1_pos[1]] = 0
        self.env[next_pos[0]][next_pos[1]] = 1
        self.obj1_pos = next_pos
