import pygame
from Drone import Drone

class DroneManager():
    def __init__(self, game, n_drones):
        self.game = game
        self.cave = game.cave.bin_map
        self.drone_list = self.build_drones(n_drones)

    def build_drones(self, n):
        # To be completed
        pass

