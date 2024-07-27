from dataclasses import dataclass

import pygame
from pygame.math import Vector2 as Vec2

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 60


@dataclass
class Vec2Int:
    x: int
    y: int

    def to_vec2(self) -> Vec2:
        return Vec2(self.x, self.y)


def render(screen):
    clear(screen)
    draw_poincare_disk(screen)
    pygame.display.update()


def draw_poincare_disk(screen):
    draw_boundary_circle(screen)


def draw_boundary_circle(screen):
    pygame.draw.circle(
        screen,
        (0, 0, 0),
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
        0.4 * SCREEN_WIDTH,
        1,
    )


def unit_circle_coordinates_to_real(coord: Vec2, width: int, height: int) -> Vec2Int:
    frame_x = (coord.x + 1) / 2
    frame_y = 1 - (coord.y + 1) / 2

    real_x = int(width * frame_x)
    real_y = int(height * frame_y)
    return Vec2Int(real_x, real_y)


def circle_to_bounding_rect(center: Vec2, radius: float):
    top_left_corner = center - Vec2(radius, radius)
    width, height = 2 * radius, 2 * radius
    return (top_left_corner.x, top_left_corner.y, width, height)

def draw_hyperbolic_geodesic(screen, alpha, beta):
    #TODO: Handle the diametric case
    


def clear(screen):
    screen.fill((255, 255, 255))


def mainLoop():
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    exit = False

    while not exit:
        clock.tick(FPS)
        render(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Poincare Disk")
    mainLoop()
    pygame.quit()
