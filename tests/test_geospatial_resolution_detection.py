import numpy as np
import pandas as pd
from scipy.stats import norm
from cartwright.analysis.space_resolution import preprocess_latlon, detect_latlon_resolution
from cartwright.schemas import AngleUnit, Uniformity
from typing import Tuple

import pytest
import pdb


@pytest.mark.parametrize('file,unit,scale', [
    ('tests/test_data/0.25_degree.csv', AngleUnit.degrees, 0.25), #slow
    ('tests/test_data/0.5_degree.csv', AngleUnit.degrees, 0.5),
    ('tests/test_data/1_degree.csv', AngleUnit.degrees, 1.0),
    ('tests/test_data/zos_AVISO_L4_199210-201012.csv', AngleUnit.degrees, 1.0),
    ('tests/test_data/2_degree.csv', AngleUnit.degrees, 2.0),
    ('tests/test_data/2.5_degree.csv', AngleUnit.degrees, 2.5),
    ('tests/test_data/NOAAGlobalTemp_v5.0.0_gridded_s188001_e202209_c20221008T133300.csv', AngleUnit.degrees, 5.0),
])
def test_known_grid(file:str, unit:AngleUnit, scale:float):
    df = pd.read_csv(file)
    lat,lon = df.iloc[:,0].values, df.iloc[:,1].values
    res = detect_latlon_resolution(lat, lon)
    assert res is not None, f'failed to detect resolution for {unit} with scale {scale} in {file}'
    assert res.square is not None, f'detected resolution is not square for {unit} with scale {scale} in {file}'
    assert res.square.unit == unit, f'detected resolution unit is not {unit} for {unit} with scale {scale} in {file}'
    assert res.square.resolution - scale < 1e-6, f'detected resolution scale is not {scale} for {unit} with scale {scale} in {file}'



#TODO: for some reason, can't generate a grid that has regularly uniformity. can only generate perfect or not uniform
@pytest.mark.parametrize("unit,scale", [
    (unit, scale) for unit in AngleUnit for scale in [0.25, 0.5, 1.0, 1.5]
])
def test_synthetic_square_grid(unit:AngleUnit, scale:float):
    lat,lon = generate_latlon_square(unit.value*scale, 20)

    res = detect_latlon_resolution(lat, lon)
    assert res is not None, f'failed to detect resolution for {unit} with scale {scale}'
    assert res.square is not None, f'detected resolution is not square for {unit} with scale {scale}'
    assert res.square.unit == unit, f'detected resolution unit is not {unit} for {unit} with scale {scale}'
    assert res.square.resolution - scale < 1e-6, f'detected resolution scale is not {scale} for {unit} with scale {scale}'



@pytest.mark.parametrize('unit,lat_scale,lon_scale', [
    (unit, lat_scale, lon_scale) for unit in AngleUnit for lat_scale in [0.25, 0.5, 1.0, 1.5] for lon_scale in [0.25, 0.5, 1.0, 1.5] if lat_scale != lon_scale
])
def test_synthetic_rect_grid(unit:AngleUnit, lat_scale:float, lon_scale:float):
    lat,lon = generate_latlon_rect(unit.value*lat_scale, unit.value*lon_scale, 20, 20)

    res = detect_latlon_resolution(lat, lon)
    assert res is not None, f'failed to detect resolution for {unit} with lat_scale {lat_scale} and lon_scale {lon_scale}'
    assert res.lat is not None and res.lon is not None, f'detected resolution is not rect for {unit} with lat_scale {lat_scale} and lon_scale {lon_scale}'
    assert res.lat.unit == unit, f'detected resolution latitude unit is not {unit} for {unit} with lat_scale {lat_scale} and lon_scale {lon_scale}'
    assert res.lat.resolution - lat_scale < 1e-6, f'detected resolution latitude scale is not {lat_scale} for {unit} with lat_scale {lat_scale} and lon_scale {lon_scale}'
    assert res.lon.unit == unit, f'detected resolution longitude unit is not {unit} for {unit} with lat_scale {lat_scale} and lon_scale {lon_scale}'
    assert res.lon.resolution - lon_scale < 1e-6, f'detected resolution longitude scale is not {lon_scale} for {unit} with lat_scale {lat_scale} and lon_scale {lon_scale}'



def test_1_degree_globe():
    n_points = 64800
    d_points = int(np.round((n_points/2) ** (1/2)))
    lats = (np.arange(d_points) + 0.5) / d_points * 180 - 90
    lons = (np.arange(2*d_points) + 0.5) / (2 * d_points) * 360 - 180
    lat,lon = np.meshgrid(lats, lons)
    lat,lon = lat.flatten(), lon.flatten()
    res = detect_latlon_resolution(lat, lon)

    # should be perfectly uniform at 1 degree
    assert res is not None, f'failed to detect resolution for in perfectly uniform grid test'
    assert res.square is not None, f'detected resolution is not square for in perfectly uniform grid test'
    assert res.square.unit == AngleUnit.degrees, f'detected resolution unit is not degrees for in perfectly uniform grid test'
    assert res.square.resolution - 1.0 < 1e-6, f'detected resolution scale is not 1.0 for in perfectly uniform grid test'




############################### HELPER FUNCTIONS ########################################

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





