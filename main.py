"""Command-line entry point for Cave Game.

This file is intentionally tiny: it creates the high-level ``Game`` object,
lets the game own all menu/mission behavior, and keeps process cleanup
(``pygame.quit`` and exit codes) in one predictable place.
"""

import os
import logging

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import pygame

from game import Game

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the game and keep process termination at the entry point."""
    try:
        Game().run()
    except RuntimeError:
        logger.exception("Cave Game stopped due to an unrecoverable error")
        return 1
    finally:
        pygame.quit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
