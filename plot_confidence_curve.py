from pathlib import Path

import pygame


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def main() -> None:
    pygame.init()

    width, height = 1100, 760
    surface = pygame.Surface((width, height))
    surface.fill((255, 255, 255))

    margin_left = 110
    margin_right = 70
    margin_top = 70
    margin_bottom = 110
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    plot_rect = pygame.Rect(margin_left, margin_top, plot_w, plot_h)

    black = (25, 25, 25)
    light_gray = (220, 220, 220)
    gray = (120, 120, 120)
    blue = (40, 100, 240)
    red = (220, 80, 60)

    # Grid lines
    for i in range(6):
        x = int(lerp(plot_rect.left, plot_rect.right, i / 5))
        y = int(lerp(plot_rect.top, plot_rect.bottom, i / 5))
        pygame.draw.line(surface, light_gray, (x, plot_rect.top), (x, plot_rect.bottom), 1)
        pygame.draw.line(surface, light_gray, (plot_rect.left, y), (plot_rect.right, y), 1)

    # Axes
    pygame.draw.rect(surface, black, plot_rect, 2)
    font = pygame.font.Font(None, 32)
    title_font = pygame.font.Font(None, 42)

    title = title_font.render("Occupancy Confidence Curve", True, black)
    surface.blit(title, title.get_rect(center=(width // 2, 28)))

    x_label = font.render("confidence", True, black)
    y_label = font.render("normalized brightness", True, black)
    surface.blit(x_label, x_label.get_rect(center=(width // 2, height - 40)))
    y_rotated = pygame.transform.rotate(y_label, 90)
    surface.blit(y_rotated, y_rotated.get_rect(center=(42, height // 2)))

    # Tick labels
    for i in range(6):
        c = i / 5
        x = int(lerp(plot_rect.left, plot_rect.right, c))
        y = int(lerp(plot_rect.bottom, plot_rect.top, c))
        tick = font.render(f"{c:.1f}", True, gray)
        surface.blit(tick, tick.get_rect(center=(x, plot_rect.bottom + 22)))
        tick_y = font.render(f"{c:.1f}", True, gray)
        surface.blit(tick_y, tick_y.get_rect(center=(plot_rect.left - 28, y)))

    # Current curve: gamma confidence response used in SlamRenderer
    points = []
    linear_points = []
    for i in range(401):
        c = i / 400.0
        gamma = c ** 6.0
        brightness = (30.0 + 225.0 * gamma) / 255.0
        linear_brightness = c
        x = int(lerp(plot_rect.left, plot_rect.right, c))
        y = int(lerp(plot_rect.bottom, plot_rect.top, brightness))
        y_linear = int(lerp(plot_rect.bottom, plot_rect.top, linear_brightness))
        points.append((x, y))
        linear_points.append((x, y_linear))

    if len(linear_points) > 1:
        pygame.draw.lines(surface, red, False, linear_points, 3)
    if len(points) > 1:
        pygame.draw.lines(surface, blue, False, points, 4)

    legend_font = pygame.font.Font(None, 28)
    legend_items = [
        (blue, "current brightness curve: 30 + 225 * c^6.0"),
        (red, "reference linear response: c"),
    ]
    legend_x = plot_rect.left + 20
    legend_y = plot_rect.top + 20
    for color, text in legend_items:
        pygame.draw.rect(surface, color, (legend_x, legend_y + 5, 18, 18))
        label = legend_font.render(text, True, black)
        surface.blit(label, (legend_x + 28, legend_y))
        legend_y += 34

    out_path = Path("generated-images/confidence-curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(out_path))


if __name__ == "__main__":
    main()