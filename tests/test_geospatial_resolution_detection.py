import numpy as np
import pandas as pd
from cartwright.analysis.space_resolution import preprocess_latlon, detect_latlon_resolution
from cartwright.schemas import AngleUnit, Uniformity
from .helpers import latlon2xyz, generate_latlon_square, generate_latlon_rect

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



def main():
    from matplotlib import pyplot as plt
    def set_axes_equal(ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        ax.set_box_aspect([1,1,1])
        ax.set_proj_type('ortho')
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])



    def africa_grid_example():
        df = pd.read_csv('tests/test_data/Africa-0.1.csv')
        lat,lon = df['latitude'].values, df['longitude'].values
        return lat,lon

    def netcdf_examples():
        from glob import glob
        for f in glob('tests/test_data/netcdf/CSVs/*.csv'):
            print(f)
            df = pd.read_csv(f)
            #should be 2 columns with unknown names. try lat,lon first
            lat,lon = df.iloc[:,0].values, df.iloc[:,1].values

            #remove duplicates and empty rows
            lat,lon = preprocess_latlon(lat, lon)

            #check if lat/lon are in the right range
            check_lat0 = lambda x: (x >= -90) & (x <= 90)
            check_lat1 = lambda x: (x >= 0) & (x <= 180)
            check_lon0 = lambda x: (x >= -180) & (x <= 180)
            check_lon1 = lambda x: (x >= 0) & (x <= 360)

            if (check_lat0(lat).all() or check_lat1(lat).all()) and (check_lon0(lon).all() or check_lon1(lon).all()):
                yield lat,lon
                continue

            lat,lon = lon,lat
            if (check_lat0(lat).all() or check_lat1(lat).all()) and (check_lon0(lon).all() or check_lon1(lon).all()):
                yield lat,lon
                continue
            
            pdb.set_trace()

    #Some experiments with plotting points on a sphere
    n_points = 500
    # x,y,z = sphere_from_area(*uniform_square(n_points))
    # x,y,z = fibonacci_sphere(n_points)
    # lat, lon = xyz2latlon(x,y,z)
    # lat, lon = africa_grid_example()
    # x,y,z = latlon2xyz(lat,lon)
    for lat,lon in netcdf_examples():
        #lat,lon are already preprocessed
        if len(lat) < 5:
            print('skipping, too few points')
            continue

        #DEBUG plot latlon points in 2D
        s = 10.0 if len(lat) < 5000 else 0.1
        plt.scatter(lon, lat, s=s)
        plt.show()

        # #DEBUG plot points in 3D
        x,y,z = latlon2xyz(lat,lon)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z, s=s)
        set_axes_equal(ax)
        plt.show()



        res = detect_latlon_resolution(lat, lon)
        print(res)
    

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







if __name__ == '__main__':
    main()