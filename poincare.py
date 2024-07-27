from dataclasses import dataclass
import math

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
    draw_hyperbolic_geodesic(screen, math.pi / 3, 4 * math.pi / 3)


def draw_boundary_circle(screen):
    pygame.draw.circle(
        screen,
        (0, 0, 0),
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
        0.5 * SCREEN_WIDTH,
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


def diametrically_opposite_angles(alpha: float, beta: float) -> bool:
    theta = (beta - alpha) / math.pi
    return float_is_int(theta)


def float_is_int(num: float) -> bool:
    return abs(num - round(num)) < 0.001


def draw_hyperbolic_geodesic(screen, alpha: float, beta: float):
    if diametrically_opposite_angles(alpha, beta):
        draw_diameter(screen, alpha)
    else:
        draw_perpendicular_arc(screen, alpha, beta)


def draw_diameter(screen, alpha: float):
    start_point = Vec2(math.cos(alpha), math.sin(alpha))
    beta = alpha + math.pi
    end_point = Vec2(math.cos(beta), math.sin(beta))

    start_real = unit_circle_coordinates_to_real(
        start_point, SCREEN_WIDTH, SCREEN_HEIGHT
    )
    end_real = unit_circle_coordinates_to_real(end_point, SCREEN_WIDTH, SCREEN_HEIGHT)

    pygame.draw.line(screen, (0, 0, 0), start_real.to_vec2(), end_real.to_vec2())


def draw_perpendicular_arc(screen, alpha: float, beta: float):
    theta = beta - alpha
    r = math.tan(theta / 2)
    unrotated_center = Vec2(math.sqrt(1 + r**2), 0)

    center = unrotated_center.rotate_rad(theta / 2 + alpha)
    alpha_prime = math.pi / 2 + theta + alpha
    beta_prime = 3 * math.pi / 2 + alpha

    center_real_coordinates = unit_circle_coordinates_to_real(
        center, SCREEN_WIDTH, SCREEN_HEIGHT
    )

    bbox = circle_to_bounding_rect(
        center_real_coordinates.to_vec2(), r * SCREEN_WIDTH / 2
    )  # TODO: fix this. Also, GODFUCKINGDAMMIT

    pygame.draw.arc(screen, (100, 100, 100), bbox, alpha_prime, beta_prime)


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
