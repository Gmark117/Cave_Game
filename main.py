import os

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

from Game import Game

if __name__ == '__main__':
    # Create and run the game
    game = Game()
    game.run()
    