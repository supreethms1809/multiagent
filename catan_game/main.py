import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from catangame import CatanGame
from boardView import BoardView
from player import Player
from dice import Dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=tuple, default=(10, 10), help="Size of the board")
    parser.add_argument("--num_players", type=int, default=4, help="Number of players")
    parser.add_argument("--plot", action="store_true", help="Plot the board")
    parser.add_argument("--animate", action="store_true", help="Animate the game")
    parser.add_argument("--play", action="store_true", help="Play the game")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps")
    args = parser.parse_args()
    config = args
    catan_game = CatanGame(config)

    if config.plot:
        catan_game.plot_board()
    if config.animate:
        catan_game.animate_board()

if __name__ == "__main__":
    main()