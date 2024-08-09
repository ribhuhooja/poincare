from dataclasses import dataclass
from typing import List, Tuple
import math
import cmath

import pygame
from pygame.math import Vector2 as Vec2

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
FPS = 60

POINT_RADIUS = 5


@dataclass
class Vec2Int:
    x: int
    y: int

    def to_vec2(self) -> Vec2:
        return Vec2(self.x, self.y)


@dataclass
class MobiusTransform:
    a: float
    b: float
    c: float
    d: float

    # NOTE: ad - bc should equal 1
    # Represents the matrix
    # |a b|
    # |c d|

    def __iter__(self):
        return iter((self.a, self.b, self.c, self.d))

    def __call__(self, z: complex) -> complex:
        a, b, c, d = self
        return (a * z + b) / (c * z + d)


def cexpi(theta: float):
    return cmath.exp(1j * theta)


@dataclass
class IdealPolygon:
    points: List[float]

    @staticmethod
    def regular(num_sides: int, starting_angle: float) -> "IdealPolygon":
        angle_difference = 2 * math.pi / num_sides
        angles = [starting_angle + i * angle_difference for i in range(num_sides)]
        return IdealPolygon(angles)


class PoincareDisk:
    def __init__(self):
        self.points: List[complex] = []
        self.ideal_polygons: List[IdealPolygon] = []
        self.geodesics: List[Tuple[float, float]] = []

    def add_point(self, point: complex):
        self.points.append(point)

    def add_ideal_polygon(self, polygon: IdealPolygon):
        self.ideal_polygons.append(polygon)

    def add_geodesic(self, alpha: float, beta: float):
        self.geodesics.append((alpha, beta))


def render(screen, disk: PoincareDisk):
    clear(screen)
    draw_poincare_disk(screen, disk)
    pygame.display.update()


def draw_poincare_disk(screen, disk: PoincareDisk):
    draw_boundary_circle(screen)

    for point in disk.points:
        draw_point(screen, point, POINT_RADIUS)

    for polygon in disk.ideal_polygons:
        draw_ideal_polygon(screen, polygon)

    for geodesic in disk.geodesics:
        draw_hyperbolic_geodesic(screen, geodesic[0], geodesic[1])


def draw_boundary_circle(screen):
    pygame.draw.circle(
        screen,
        (0, 0, 0),
        (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
        0.5 * SCREEN_WIDTH,
        1,
    )


def cnum_to_pixels(coord: complex, width: int, height: int) -> Vec2Int:
    frame_x = (coord.real + 1) / 2
    frame_y = 1 - (coord.imag + 1) / 2

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


def draw_ideal_polygon(screen, polygon: IdealPolygon):
    angles = polygon.points
    length = len(angles)
    if length <= 1:
        return

    for i in range(length - 1):
        draw_hyperbolic_geodesic(screen, angles[i], angles[i + 1])

    draw_hyperbolic_geodesic(screen, angles[-1], angles[0])


def draw_diameter(screen, alpha: float):
    start_point = complex(math.cos(alpha), math.sin(alpha))
    beta = alpha + math.pi
    end_point = complex(math.cos(beta), math.sin(beta))

    start_real = cnum_to_pixels(start_point, SCREEN_WIDTH, SCREEN_HEIGHT)
    end_real = cnum_to_pixels(end_point, SCREEN_WIDTH, SCREEN_HEIGHT)

    pygame.draw.line(screen, (0, 0, 0), start_real.to_vec2(), end_real.to_vec2())


def draw_perpendicular_arc(screen, alpha: float, beta: float):

    bigger, smaller = max(alpha, beta), min(alpha, beta)
    while bigger - smaller > math.pi:
        smaller += 2 * math.pi
        bigger, smaller = max(bigger, smaller), min(bigger, smaller)

    beta, alpha = bigger, smaller

    theta = beta - alpha
    r = math.tan(theta / 2)
    unrotated_center = complex(math.sqrt(1 + r**2), 0)

    center = unrotated_center * cexpi(theta / 2 + alpha)
    alpha_prime = math.pi / 2 + theta + alpha
    beta_prime = 3 * math.pi / 2 + alpha

    center_real_coordinates = cnum_to_pixels(center, SCREEN_WIDTH, SCREEN_HEIGHT)

    bbox = circle_to_bounding_rect(
        center_real_coordinates.to_vec2(), r * SCREEN_WIDTH / 2
    )  # TODO: fix this. Also, GODFUCKINGDAMMIT

    pygame.draw.arc(screen, (100, 100, 100), bbox, alpha_prime, beta_prime)


def draw_point(screen, cnum: complex, radius):
    coords = cnum_to_pixels(cnum, SCREEN_HEIGHT, SCREEN_HEIGHT)
    pygame.draw.circle(screen, (0, 0, 0), coords.to_vec2(), radius)


def clear(screen):
    screen.fill((255, 255, 255))


def mainLoop():
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    exit = False
    disk = PoincareDisk()

    disk.add_point(0.4 + 0.2j)
    disk.add_ideal_polygon(IdealPolygon.regular(3, math.pi / 12))
    disk.add_ideal_polygon(IdealPolygon([math.pi / 3, math.pi, 5 * math.pi / 3]))
    disk.add_geodesic(5 * math.pi / 3, math.pi / 3)

    while not exit:
        clock.tick(FPS)
        render(screen, disk)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pass
                # alpha += 0.1


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("Poincare Disk")
    mainLoop()
    pygame.quit()
