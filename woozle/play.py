from copy import deepcopy
import pygame as pg
from game import WoozleGame
import warnings


def play():
    pg.init()
    game = WoozleGame()
    game.render()
    running = True
    step = 0
    while running:
        action = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            # checking if keydown event happened or not
            if event.type == pg.KEYDOWN:
                if event.key in [pg.K_e, pg.K_UP, pg.K_DOWN, pg.K_RIGHT]:
                    print(f'Step {step}')
                if event.key == pg.K_z:
                    game.switch_mode()
                elif event.key == pg.K_UP:
                    action = 0
                elif event.key == pg.K_DOWN:
                    action = 1
                elif event.key == pg.K_LEFT:
                    action = 2
                elif event.key == pg.K_RIGHT:
                    action = 3
                else:
                    pass
                if action is not None:
                    break
        if action is not None:
            step += 1
            game.update_env(action)
        game.render()
        pg.display.update()

if __name__ == '__main__':
    play()
