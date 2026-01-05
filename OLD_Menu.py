import pygame
import pygame.mixer as mix
import Assets

class Menu():
    def __init__(self, game):
        # Initialise the mixer module for audio 
        mix.init()  
        # Store the game instance to access its variables and methods
        self.game = game
        
        # Define positions for centering elements on the screen
        self.mid_w = Assets.DISPLAY_W/2
        self.mid_h = Assets.DISPLAY_H/2
        
        # Initialise a flag to control the menu loop
        self.run_display = True
        
        # Load background images for the menu
        self.background      = pygame.image.load(Assets.Images['CAVE'].value)
        self.dark_background = pygame.image.load(Assets.Images['DARK_CAVE'].value)

        # Load the ambient audio file for background music
        mix.music.load(Assets.Audio['AMBIENT'].value)  
        
        # Load the button click sound effect and set its volume
        self.button = mix.Sound(Assets.Audio['BUTTON'].value)
        self.button.set_volume(0.5)
        
    # Write title and menu voices
    def draw_text(self, text, size, x, y, font, color, handle):
        # Create a text surface using the specified font and size
        style = pygame.font.Font(font, size)
        text_surface = style.render(text, True, color)
        
        # Get dimensions of the writable rectangle for positioning
        text_rect = text_surface.get_rect()

        # Choose the point of the surface to which the coordinates refer for alignment
        match handle:
            case 'Center':
                text_rect.center = (x,y)
            case 'Midtop':
                text_rect.midtop = (x,y)
            case 'Midright':
                text_rect.midright = (x,y)
            case 'Midleft':
                text_rect.midleft = (x,y)
        
        # Render the text on the display surface
        self.game.display.blit(text_surface,text_rect)
    
    # Move the cursor between menu items based on user input
    def move_cursor(self, states, state, cursor_pos, x_coords = [], y_coords = []):
        # Loop through the list of states to find the current state
        for i in range(len(states)):
            # If no input is detected to change the state, exit the loop
            if (not self.game.UP_KEY) and (not self.game.DOWN_KEY):
                break

            # If the input was to go up, update the cursor position and the state
            if state == states[i] and self.game.UP_KEY:
                cursor_pos = [x_coords[(i-1) % len(states)], y_coords[(i-1) % len(states)]]
                state      = states[(i-1) % len(states)]
                break
            
            # If the input was to go down, update the cursor position and the state
            if state == states[i] and self.game.DOWN_KEY:
                cursor_pos = [x_coords[(i+1) % len(states)], y_coords[(i+1) % len(states)]]
                state      = states[(i+1) % len(states)]
                break
        
        return cursor_pos, state

    # Play the button sound if the toggle allows it
    def play_button(self, switch):
        if switch == 'on':
            self.button.play()

    # Change the display back to the main menu and return display flag
    def to_main_menu(self):
        self.game.curr_menu = self.game.main_menu
        display_flag = False
        
        return display_flag
    
    # Display the loading screen with a dynamic dot based on processes completed
    def loading_screen(self, proc_counter):
        text1 = 'Digging...'
        text2 = ['Waiting for', 'stalactites to grow...']
        match proc_counter:
            case 0:
                self.blit_loading([text1[:-3]])
            case 1:
                self.blit_loading([text1[:-2]])
            case 2:
                self.blit_loading([text1[:-1]])
            case 3:
                self.blit_loading([text1])
            case 5:
                self.blit_loading([text2[0], text2[1][:-3]])
            case 6:
                self.blit_loading([text2[0], text2[1][:-2]])
            case 7:
                self.blit_loading([text2[0], text2[1][:-1]])
            case 8:
                self.blit_loading(text2)

    # Display a static loading screen accounting for multiline texts
    def blit_loading(self, text=['Loading...']):
        # Draw the dark background for the loading screen
        self.game.display.blit(self.dark_background,(0,0))

        lines = len(text)  # Number of lines in the text to display
        line_offset = 100
        decenter_offset = 50

        # Calculate the starting y-coordinate based on the number of lines
        if lines%2==0:
            upper_lines = int((lines/2))
            first_line_y = self.mid_h - decenter_offset - line_offset*(upper_lines - 1)
        else:
            upper_lines = int(((lines - 1)/2))
            first_line_y = self.mid_h - line_offset*upper_lines 

        # Draw each line of the loading text on the display
        for line in range(lines):
                self.draw_text(text[line], 100,
                       self.mid_w,
                       first_line_y + line_offset*line, 
                       Assets.Fonts['BIG'].value,
                       Assets.Colors['WHITE'].value,
                       Assets.RectHandle['CENTER'].value)

        # Update the display 
        self.game.blit_screen()
