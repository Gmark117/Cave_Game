import os
import time
import configparser
import pygame
import Assets

class ControlCenter():
    def __init__(self, game, num_drones):
        self.game = game
        self.tic = 0
        self.explored_percent = 100 # TO BE CALCULATED OUTSIDE AND PASSED AS ARGUMENT

        # Get number of deployed drones and rovers
        self.num_drones = num_drones
        self.num_rovers = 1 + (4 % num_drones)

        # Calculate surface origin
        self.origin_x = Assets.FULLSCREEN_W - Assets.LEGEND_WIDTH
        self.origin_y = 0
        self.origin   = (self.origin_x,self.origin_y)

        # Calculate surface mid points
        self.mid_x = self.origin_x + (Assets.LEGEND_WIDTH / 2)
        self.mid_y = Assets.FULLSCREEN_H / 2

        # Define surface
        self.control_surf = pygame.Surface((Assets.LEGEND_WIDTH, Assets.FULLSCREEN_H), pygame.SRCALPHA)
        self.control_surf.fill((*Assets.Colors.BLACK.value, 255))

        # Create dictionaries
        self.drone_dict()
        self.rover_dict()

    # Create drone dictionary
    def drone_dict(self):
        self.drones = {
            'Blinky': {
                'id': 0,
                'color': Assets.DroneColors.RED.value,
                'battery': 10,
                'status': 'Ready'
            },
            'Pinky': {
                'id': 1,
                'color': Assets.DroneColors.PINK.value,
                'battery': 50,
                'status': 'Homing'
            },
            'Inky': {
                'id': 2,
                'color': Assets.DroneColors.L_BLUE.value,
                'battery': 100,
                'status': 'Charging'
            },
            'Clyde': {
                'id': 3,
                'color': Assets.DroneColors.ORANGE.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Sue': {
                'id': 4,
                'color': Assets.DroneColors.PURPLE.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Tim': {
                'id': 5,
                'color': Assets.DroneColors.BROWN.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Funky': {
                'id': 6,
                'color': Assets.DroneColors.GREEN.value,
                'battery': 100,
                'status': 'Ready'
            },
            'Kinky': {
                'id': 7,
                'color': Assets.DroneColors.GOLD.value,
                'battery': 100,
                'status': 'Ready'
            }
        }

    # Create rover dictionary
    def rover_dict(self):
        self.rovers = {
            'Huey' : {
                'id': 0,
                'color': Assets.RoverColors.RED.value,
                'battery': 2400,
                'status': 'Ready'
            },
            'Dewey' : {
                'id': 1,
                'color': Assets.RoverColors.BLUE.value,
                'battery': 1400,
                'status': 'Updating'
            },
            'Louie' : {
                'id': 2,
                'color': Assets.RoverColors.GREEN.value,
                'battery': 240,
                'status': 'Ready'
            }
        }
    
    def start_timer(self):
        self.tic = time.perf_counter()
    
    def format_timer(self):
        #str(round(time.perf_counter()-self.tic))
        elapsed = round(time.perf_counter() - self.tic)
        minutes, seconds = divmod(elapsed, 60)

        str_minutes = '0' + str(minutes) if (minutes<10) else str(minutes)
        str_seconds = '0' + str(seconds) if (seconds<10) else str(seconds)

        return str_minutes + ':' + str_seconds

#  ____   ____      _  __        __ ___  _   _   ____ 
# |  _ \ |  _ \    / \ \ \      / /|_ _|| \ | | / ___|
# | | | || |_) |  / _ \ \ \ /\ / /  | | |  \| || |  _
# | |_| ||  _ <  / ___ \ \ V  V /   | | | |\  || |_| |
# |____/ |_| \_\/_/   \_\ \_/\_/   |___||_| \_| \____|

    # Set color depending on the percentage
    def percent_color(self, val, max_val=100):
        if val < max_val*20/100:
            return Assets.Colors['RED'].value
        elif val < max_val*80/100:
            return Assets.Colors['YELLOW'].value
        else:
            return Assets.Colors['GREEN'].value

    # Write title and voices
    def draw_text(self, texts, size, x, y, font, handle):
        style = pygame.font.Font(font, size)
        total_width = 0
        surfaces = []
        
        # Create individual surfaces for each substring and calculate total width
        for substring, color, alpha in texts:
            text_surface = style.render(substring, True, color)
            text_surface.set_alpha(alpha)
            surfaces.append(text_surface)
            total_width += text_surface.get_width()
        
        # Set initial x position based on alignment
        if handle == 'Center':
            start_x = x - total_width // 2
        elif handle == 'Midtop':
            start_x = x - total_width // 2
        elif handle == 'Midright':
            start_x = x - total_width
        elif handle == 'Midleft':
            start_x = x
        else:
            start_x = x  # Default to left-aligned if handle is unknown
        
        # Blit each surface with appropriate horizontal offset
        for surface in surfaces:
            text_rect = surface.get_rect()
            text_rect.midleft = (start_x, y)
            self.game.window.blit(surface, text_rect)
            start_x += surface.get_width()  # Move x for next substring


    def draw_statistics(self):
        # Draw time
        self.draw_text([('MET:           ', Assets.Colors['GREY'].value, 255),
                        (self.format_timer(), Assets.Colors['WHITE'].value, 255)],
                       25,
                       self.origin_x,
                       120,
                       Assets.Fonts['BIG'].value,
                       Assets.RectHandle['MIDLEFT'].value)
        # Draw explored map percentage
        self.draw_text([('Explored:     ', Assets.Colors['GREY'].value, 255),
                        (f'{self.explored_percent}%', self.percent_color(self.explored_percent), 255)],
                       25,
                       self.origin_x,
                       150,
                       Assets.Fonts['BIG'].value,
                       Assets.RectHandle['MIDLEFT'].value)
    
    def draw_status(self, label, rover=False, deployed=True):
        # Get data
        if not rover:
            number = self.drones[label]['id']
            color = self.drones[label]['color']
            battery = self.drones[label]['battery']
            status = self.drones[label]['status']

            name_height = 230
            data_height = 260
            max_battery = 100
        else:
            number = self.rovers[label]['id']
            color = self.rovers[label]['color']
            battery = self.rovers[label]['battery']
            status = self.rovers[label]['status']

            name_height = 760
            data_height = 790
            max_battery = 2400

        # Blit label
        self.draw_text([(label, color, 255)], 25,
                       self.origin_x,
                       name_height + 60*number,
                       Assets.Fonts['BIG'].value,
                       Assets.RectHandle['MIDLEFT'].value)
        
        if deployed:
            # Define Status color
            match status:
                case 'Ready'|'Done':
                    status_color = Assets.Colors['GREEN'].value
                case 'Updating'|'Advancing'|'Sharing'|'Charging':
                    status_color = Assets.Colors['YELLOW'].value
                case 'Deployed'|'Homing':
                    status_color = Assets.Colors['WHITE'].value
                case _:
                    status_color = Assets.Colors['RED'].value
            
            # Define Battery color
            battery_color = self.percent_color(battery, max_battery)

            # Blit data
            self.draw_text([(f'{battery}%', battery_color, 128),
                            ('  |  ', Assets.Colors['WHITE'].value, 128),
                            (status, status_color, 128)],
                           25,
                           self.origin_x,
                           data_height + 60*number,
                           Assets.Fonts['BIG'].value,
                           Assets.RectHandle['MIDLEFT'].value)
        else:
            # Blit 'N/A'
            self.draw_text([('N/A', Assets.Colors['GREY'].value, 128)],
                           25,
                           self.origin_x,
                           data_height + 60*number,
                           Assets.Fonts['BIG'].value,
                           Assets.RectHandle['MIDLEFT'].value)

    # Blit the control center on the map
    def draw_control_center(self):
        self.game.window.blit(self.control_surf, self.origin)
        # Draw title
        self.draw_text([('Control Center', Assets.Colors['RED'].value, 255)],
                       35,
                       self.origin_x,
                       70,
                       Assets.Fonts['BIG'].value,
                       Assets.RectHandle['CENTER'].value)
        # Draw statistics
        self.draw_statistics()
        # Draw subtitle
        self.draw_text([('Drones', Assets.Colors['EUCALYPTUS'].value, 255)],
                       30,
                       self.mid_x,
                       195,
                       Assets.Fonts['BIG'].value,
                       Assets.RectHandle['CENTER'].value)

        for drone in self.drones:
            if self.drones[drone]['id'] < self.num_drones:
                self.draw_status(drone)
            else:
                self.draw_status(drone, deployed=False)
        # Draw subtitle
        self.draw_text([('Rovers', Assets.Colors['EUCALYPTUS'].value, 255)],
                       30,
                       self.mid_x,
                       725,
                       Assets.Fonts['BIG'].value,
                       Assets.RectHandle['CENTER'].value)

        for rover in self.rovers:
            if self.rovers[rover]['id'] < self.num_rovers:
                self.draw_status(rover, rover=True)
            else:
                self.draw_status(rover, rover=True, deployed=False)
        