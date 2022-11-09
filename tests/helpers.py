import numpy as np
from scipy.stats import norm
from typing import Tuple


def linspace(a,b=None,n=10,extremes=True):
    if b is None:
        a,b = 0,a
    r = (np.arange(n) / (n-1)) if extremes else (np.arange(n) + 0.5) / n
    return a + r * (b-a)


def uniform_cube(n_points:int, extremes=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generate a uniform 3D grid of points in a cube"""
    d_points = int(np.round(n_points ** (1/3)))
    side = linspace(0,1,d_points,extremes)
    x,y,z = np.meshgrid(side, side, side)
    x,y,z = x.flatten(), y.flatten(), z.flatten()
    return x, y, z


def uniform_square(n_points:int, extremes=False) -> Tuple[np.ndarray, np.ndarray]:
    """generate a uniform 2D grid of points in a square"""
    d_points = int(np.round(n_points ** (1/2)))
    side = linspace(0,1,d_points,extremes)
    x,y = np.meshgrid(side, side)
    x,y = x.flatten(), y.flatten()
    return x, y


def sphere_from_icdf(x:np.ndarray, y:np.ndarray, z:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """convert to 3D coordinates on a sphere via the inverse normal cumulative distribution"""
    xn,yn,zn = norm.ppf(x), norm.ppf(y), norm.ppf(z) #inverse normal transform
    r = np.sqrt(xn**2 + yn**2 + zn**2)
    xn,yn,zn = xn/r, yn/r, zn/r

    return xn,yn,zn


def sphere_from_area(z:np.ndarray, φ:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """convert to 3D coordinates on a sphere via an equal area transformation"""
    z = 2*z - 1
    φ = 2*np.pi*φ
    r = np.sqrt(1-z**2)
    x = r*np.cos(φ)
    y = r*np.sin(φ)
    return x,y,z


def random_normal_uniform(n_points:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """directly generate random points for x,y,z, which should be approximately uniformly distributed"""
    x,y,z = np.random.normal(size=(3,n_points))
    r = np.sqrt(x**2 + y**2 + z**2)
    x,y,z = x/r, y/r, z/r
    return x,y,z


def fibonacci_sphere(n_points:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    φ = np.pi * (3 - np.sqrt(5)) #golden angle
    y = linspace(1, -1, n_points, True)
    r = np.sqrt(1 - y*y)
    θ = φ * np.arange(n_points)
    x = np.cos(θ) * r
    z = np.sin(θ) * r
    return x,y,z


def xyz2latlon(x:np.ndarray, y:np.ndarray, z:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.arcsin(z)
    lon = np.arctan2(y,x)
    lat,lon = np.rad2deg(lat), np.rad2deg(lon)
    return lat, lon


def latlon2xyz(lat:np.ndarray, lon:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat,lon = np.deg2rad(lat), np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x,y,z


def generate_n_point_latlon_grid(n_points:int) -> Tuple[np.ndarray, np.ndarray]:
    """generate points at uniform lat/lon intervals"""
    d_points = int(np.round((n_points/2) ** (1/2)))
    lats = linspace(-90, 90, d_points, False)
    lons = linspace(-180, 180, 2*d_points, False)
    lat,lon = np.meshgrid(lats, lons)
    lat,lon = lat.flatten(), lon.flatten()
    return lat, lon


def generate_latlon_square(delta:float, d_points:int) -> Tuple[np.ndarray, np.ndarray]:
    """generate points in a d_points x d_points square grid with the specified delta (degrees) spacing between points"""
    lats = np.arange(d_points) * delta
    lons = np.arange(d_points) * delta
    lat,lon = np.meshgrid(lats, lons)
    lat,lon = lat.flatten(), lon.flatten()
    return lat, lon


def generate_latlon_rect(lat_delta, lon_delta, lat_points, lon_points) -> Tuple[np.ndarray, np.ndarray]:
    """generate points in a lat_points x lon_points rectangular grid with the specified lat/lon spacing between points"""
    lats = np.arange(lat_points) * lat_delta
    lons = np.arange(lon_points) * lon_delta
    lat,lon = np.meshgrid(lats, lons)
    lat,lon = lat.flatten(), lon.flatten()
    return lat, lon
