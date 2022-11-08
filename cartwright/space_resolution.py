from typing import Optional, Dict, Tuple
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
from scipy.spatial import Delaunay
from cartwright.schemas import Uniformity#, SpaceResolution, LatLonResolution, SphericalResolution, CategoricalResolution
from enum import Enum, EnumMeta
import pandas as pd

from matplotlib import pyplot as plt
import pdb


#TODO: move these to schema or perhaps a dedicated file for space_schemas
#TODO: some sort of coverage metric?

from dataclasses import dataclass

class Unit(float, Enum): ...

class AngleUnit(Unit):
    degrees = 1
    minutes = degrees/60
    seconds = degrees/3600
    # radians = 180/np.pi #probably don't include since we usually only want to return degrees


@dataclass
class Resolution:
    uniformity: Uniformity
    resolution: float
    unit: Unit
    error: float


@dataclass
class CategoricalResolution:
    uniformity: Uniformity
    # category: GeoCategory
    # coverage: float # percent of category elements covered

@dataclass
class GeoSpatialResolution:
    lat: Optional[Resolution]=None
    lon: Optional[Resolution]=None
    latlon: Optional[Resolution]=None
    spherical: Optional[Resolution]=None
    categorical: Optional[CategoricalResolution]=None #TODO: this could maybe be a list?
    #TODO: other possible resolutions


def get_uniformity(vals: np.ndarray, avg: float):
    uniformity_score = np.abs(vals - avg)
    if np.all(uniformity_score < 1e-9 * (avg_mag:=np.abs(avg))):
        return Uniformity.PERFECT
    elif uniformity_score.max() < 0.01 * avg_mag:
        return Uniformity.UNIFORM
    else:
        return Uniformity.NOT_UNIFORM

def preprocess_latlon(lat:np.ndarray, lon:np.ndarray, rad=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    filter out non-unique points, empty rows, and optionally convert to radians
    
    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]

    @return: lat, lon
    """

    latlon = np.stack([lat, lon], axis=0)
    latlon = np.unique(latlon, axis=1)
    latlon = latlon[:,~np.isnan(latlon).any(axis=0)]
    if rad: 
        latlon = np.deg2rad(latlon)
    lat, lon = latlon

    return lat, lon

def match_unit(cls:EnumMeta, avg:float) -> Tuple[float, Unit]:
    #find the closest matching unit
    names = [*cls.__members__.keys()]
    durations = np.array([getattr(cls, name).value for name in names], dtype=float)
    unit_errors = np.abs(durations - avg)/durations
    closest = np.argmin(unit_errors)
    unit = getattr(cls, names[closest])
    return avg/durations[closest], unit



def detect_resolution(lat:np.ndarray, lon:np.ndarray) -> GeoSpatialResolution:
    """
    Detect the resolution of the lat/lon coordinates.

    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]

    @return: SpaceResolution(<TODO>) object where 
        ...
    """

    #detect the lat/lon resolution
    latlon_resolution = detect_latlon_resolution(lat, lon)
    # spherical_resolution = detect_spherical_resolution(lat, lon)
    # print(spherical_resolution)
    # pdb.set_trace()
    # categorical_resolution = detect_categorical_resolution(lat, lon)

    return GeoSpatialResolution(
        **latlon_resolution,
        # **spherical_resolution,
        # **categorical_resolution
    )



def detect_latlon_resolution(lat:np.ndarray, lon:np.ndarray) -> Dict[str, Resolution]:
    """
    Detect if the lat/lon coordinates are drawn from a uniform grid.

    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]

    @return: {'latlon': Resolution()} | {'lat': Resolution(), 'lon': Resolution()} | {}
        ...
    """

    #filter duplicates, and convert to radians
    lat, lon = preprocess_latlon(lat, lon, rad=True)

    #fail if not enough points
    if lat.size <= 2:
        return {}

    #compute the Delaunay triangulation on the lat/lon points
    tri = Delaunay(np.stack([lat, lon], axis=1))

    #collect together all edges of the triangles (in lat/lon space)
    edges = np.concatenate([
        #edge 1
        [lon[tri.simplices[:,0]] - lon[tri.simplices[:,1]],
         lat[tri.simplices[:,0]] - lat[tri.simplices[:,1]]],
        #edge 2
        [lon[tri.simplices[:,1]] - lon[tri.simplices[:,2]],
         lat[tri.simplices[:,1]] - lat[tri.simplices[:,2]]],
        #edge 3
        [lon[tri.simplices[:,2]] - lon[tri.simplices[:,0]],
         lat[tri.simplices[:,2]] - lat[tri.simplices[:,0]],]
    ], axis=1)

    #find edges that are either horizontal or vertical
    horizontal = edges[:, np.abs(edges[1]) < 1e-6]
    vertical = edges[:, np.abs(edges[0]) < 1e-6]

    #if less than 33% of the edges are horizontal or vertical, then no grid was detected
    #should be around 2/3 if it was a full grid, 1/3 if only horizontal or vertical
    if len(horizontal.T) + len(vertical.T) < len(edges.T) * 0.33333: 
        return {}

    #collect the lengths of the horizontal and vertical edges
    dlon = np.abs(horizontal[0])
    dlat = np.abs(vertical[1])
    dlon_avg = np.median(dlon)
    dlat_avg = np.median(dlat)

    
    # square grid
    if np.abs(dlon_avg - dlat_avg) < 1e-6:
        deltas = np.concatenate([dlon, dlat])
        avg = np.median(deltas)
        uniformity = get_uniformity(deltas, avg)

        scale, unit = match_unit(AngleUnit, np.rad2deg(avg))
        error = np.rad2deg(np.abs(1 - deltas/avg).mean()) / unit

        return {'latlon': Resolution(uniformity, scale, unit, error)}
    

    # rectangular grid
    dlon_uniformity = get_uniformity(dlon, dlon_avg)
    dlon_scale, dlon_unit = match_unit(AngleUnit, np.rad2deg(dlon_avg))
    dlon_error = np.rad2deg(np.abs(1 - dlon/dlon_avg).mean()) / dlon_unit

    dlat_uniformity = get_uniformity(dlat, dlat_avg)
    dlat_scale, dlat_unit = match_unit(AngleUnit, np.rad2deg(dlat_avg))
    dlat_error = np.rad2deg(np.abs(1 - dlat/dlat_avg).mean()) / dlat_unit

    return {
        'lat':Resolution(dlat_uniformity, dlat_scale, dlat_unit, dlat_error),
        'lon':Resolution(dlon_uniformity, dlon_scale, dlon_unit, dlon_error)
    }


def detect_spherical_resolution(lat:np.ndarray, lon:np.ndarray) -> Dict[str, Resolution]:
    """
    Detect if points are uniformly distributed on a sphere.

    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]

    @return: {'spherical':Resolution()} | {}
    """
    
    #filter out non-unique points, and convert to radians
    lat, lon = preprocess_latlon(lat, lon, rad=True)

    #fail if not enough points
    if lat.size <= 2:
        return {}

    #convert to 3D cartesian coordinates
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    #filter any points at the north pole
    mask = np.abs(z) < 1.0-1e-9
    x,y,z = x[mask], y[mask], z[mask]

    #compute an orthographic projection of the points on the sphere
    X = x/(1-z)
    Y = y/(1-z)

    #compute the Delaunay triangulation of the projected points (this is equivalent to a Delaunay triangulation directly on the sphere)
    tri = Delaunay(np.stack([X,Y], axis=1))

    #create a sparse adjacency matrix from the triangulation
    adj = csr_matrix((np.ones(len(tri.simplices)), (tri.simplices[:,0], tri.simplices[:,1])), shape=(len(X), len(X)))
    adj += csr_matrix((np.ones(len(tri.simplices)), (tri.simplices[:,1], tri.simplices[:,2])), shape=(len(X), len(X)))
    adj += csr_matrix((np.ones(len(tri.simplices)), (tri.simplices[:,2], tri.simplices[:,0])), shape=(len(X), len(X)))

    pdb.set_trace()


    #collect together all edges of the triangles (in 3D space)
    i0, i1, i2 = tri.simplices[:,0], tri.simplices[:,1], tri.simplices[:,2]
    p0 = np.stack([x[i0], y[i0], z[i0]], axis=1)
    p1 = np.stack([x[i1], y[i1], z[i1]], axis=1)
    p2 = np.stack([x[i2], y[i2], z[i2]], axis=1)

    #TODO: come up with a good way to characterize the uniformity of the points, based on the triangulation


    #compute the areas of the triangles
    areas = np.linalg.norm(np.cross(p1-p0, p2-p0), axis=1)
    plt.hist(areas, bins=100)
    plt.show()
    pdb.set_trace()

    #compute the great circle distance between each pair of points
    d01 = np.arccos((p0 * p1).sum(axis=1))
    d12 = np.arccos((p1 * p2).sum(axis=1))
    d20 = np.arccos((p2 * p0).sum(axis=1))

    deltas = np.concatenate([d01, d12, d20])
    avg = np.median(deltas)
    uniformity = get_uniformity(deltas, avg)
    error = np.abs(1 - deltas/avg).mean()

    #DEBUG plot points and edges in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    for i in range(len(tri.simplices)):
        ax.plot([x[i0[i]], x[i1[i]]], [y[i0[i]], y[i1[i]]], [z[i0[i]], z[i1[i]]], 'k-')
        ax.plot([x[i1[i]], x[i2[i]]], [y[i1[i]], y[i2[i]]], [z[i1[i]], z[i2[i]]], 'k-')
        ax.plot([x[i2[i]], x[i0[i]]], [y[i2[i]], y[i0[i]]], [z[i2[i]], z[i0[i]]], 'k-')
    set_axes_equal(ax)
    plt.show()

    return {'spherical': Resolution(uniformity, np.rad2deg(avg), GeoUnit.DEGREES, np.rad2deg(error))}


def detect_categorical_resolution(lat:np.ndarray, lon:np.ndarray) -> Optional[CategoricalResolution]:
    #edge cases/issues
    # - how stable are the categories? countries come and go, change names, etc. would be a pain to have to make this aware of the date, but how else do you handle?
    #    - possibly just assume the date is now?
    #    - possibly have the user specify the date/handling process?
    #
    pdb.set_trace()


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


def main():

    #Some experiments with plotting points on a sphere
    n_points = 500

    def linspace(a,b,n,extremes=False):
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



    # convert xn,yn,zn to lat/lon
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

    def generate_latlon_grid(n_points:int) -> Tuple[np.ndarray, np.ndarray]:
        """generate points at uniform lat/lon intervals"""# and then convert to 3D
        d_points = int(np.round((n_points/2) ** (1/2)))
        lats = (np.arange(d_points) + 0.5) / d_points * 180 - 90
        lons = (np.arange(2*d_points) + 0.5) / (2 * d_points) * 360 - 180
        lat,lon = np.meshgrid(lats, lons)
        lat,lon = lat.flatten(), lon.flatten()
        return lat, lon

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



        res = detect_resolution(lat, lon)
        print(res)
    
    

def test1():
    n_points = 64800
    d_points = int(np.round((n_points/2) ** (1/2)))
    lats = (np.arange(d_points) + 0.5) / d_points * 180 - 90
    lons = (np.arange(2*d_points) + 0.5) / (2 * d_points) * 360 - 180
    lat,lon = np.meshgrid(lats, lons)
    lat,lon = lat.flatten(), lon.flatten()
    res = detect_latlon_resolution(lat, lon)

    #TODO: test, should be perfectly uniform at 1 degree







us_latlons = {
    'Wisconsin, USA': [44.500000, -89.500000],
    'West Virginia, USA': [39.000000, -80.500000],
    'Vermont, USA': [44.000000, -72.699997],
    'Texas, the USA': [31.000000, -100.000000],
    'South Dakota, the US': [44.500000, -100.000000],
    'Rhode Island, the US': [41.742325, -71.742332],
    'Oregon, the US': [44.000000, -120.500000],
    'New York, USA': [43.000000, -75.000000],
    'New Hampshire, USA': [44.000000, -71.500000],
    'Nebraska, USA': [41.500000, -100.000000],
    'Kansas, the US': [38.500000, -98.000000],
    'Mississippi, USA': [33.000000, -90.000000],
    'Illinois, USA': [40.000000, -89.000000],
    'Delaware, the US': [39.000000, -75.500000],
    'Connecticut, USA': [41.599998, -72.699997],
    'Arkansas, the US': [34.799999, -92.199997],
    'Indiana, USA': [40.273502, -86.126976],
    'Missouri, USA': [38.573936, -92.603760],
    'Florida, USA': [27.994402, -81.760254],
    'Nevada, USA': [39.876019, -117.224121],
    'Maine, the USA': [45.367584, -68.972168],
    'Michigan, USA': [44.182205, -84.506836],
    'Georgia, the USA': [33.247875, -83.441162],
    'Hawaii, USA': [19.741755, -155.844437],
    'Alaska, USA': [66.160507, -153.369141],
    'Tennessee, USA': [35.860119, -86.660156],
    'Virginia, USA': [37.926868, -78.024902],
    'New Jersey, USA': [39.833851, -74.871826],
    'Kentucky, USA': [37.839333, -84.270020],
    'North Dakota, USA': [47.650589, -100.437012],
    'Minnesota, USA': [46.392410, -94.636230],
    'Oklahoma, the USA': [36.084621, -96.921387],
    'Montana, USA': [46.965260, -109.533691],
    'Washington, the USA': [47.751076, -120.740135],
    'Utah, USA': [39.419220, -111.950684],
    'Colorado, USA': [39.113014, -105.358887],
    'Ohio, USA': [40.367474, -82.996216],
    'Alabama, USA': [32.318230, -86.902298],
    'Iowa, the USA': [42.032974, -93.581543],
    'New Mexico, USA': [34.307144, -106.018066],
    'South Carolina, USA': [33.836082, -81.163727],
    'Pennsylvania, USA': [41.203323, -77.194527],
    'Arizona, USA': [34.048927, -111.093735],
    'Maryland, USA': [39.045753, -76.641273],
    'Massachusetts, USA': [42.407211, -71.382439],
    'California, the USA': [36.778259, -119.417931],
    'Idaho, USA': [44.068203, -114.742043],
    'Wyoming, USA': [43.075970, -107.290283],
    'North Carolina, USA': [35.782169, -80.793457],
    'Louisiana, USA': [30.391830, -92.329102],
}







    
    # lat,lon = np.deg2rad(lat), np.deg2rad(lon)
    # xn,yn,zn = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)



    # if False:
    #     #plot the points in 3D
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(xn, yn, zn)
    #     ax.set_box_aspect([1,1,1])
    #     ax.set_proj_type('ortho')
    #     set_axes_equal(ax)

    #     #plot the lat/lon points in 2D
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='polar')
    #     # ax.scatter(lon, lat)

    #     plt.show()

    # #2 side by side plots in 3D
    # # fig, ax = plt.subplots(1,2, subplot_kw={'projection':'3d'})
    # # ax[0].scatter(x,y,z, c='r', marker='o')
    # # ax[0].plot3D(x,y,z, c='grey', alpha=0.5)
    # # ax[1].scatter(xn,yn,zn, c='r', marker='o')
    # # ax[1].plot3D(xn,yn,zn, c='grey', alpha=0.5)

    # # for a in ax:
    # #     a.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    # #     a.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    # #     set_axes_equal(a) # IMPORTANT - this is also required


    # # plt.show()






    # #compute an orthographic projection of the points on the sphere
    # X = xn/(1-zn)
    # Y = yn/(1-zn)

    # #compute the Delaunay triangulation of the points
    # tri = Delaunay(np.array([X,Y]).T)

    # # #plot the points and the triangulation
    # # fig, ax = plt.subplots(1,2, subplot_kw={'aspect':'equal'})
    # # ax[0].scatter(X,Y, c='r', marker='o')
    # # ax[0].plot(X,Y, c='grey', alpha=0.5)
    # # ax[1].scatter(X,Y, c='r', marker='o')
    # # # ax[1].plot(X,Y, c='grey', alpha=0.5)
    # # for simplex in tri.simplices:
    # #     ax[1].plot(X[simplex], Y[simplex], c='k')

    # # plt.show()

    # if False:
    #     #draw corresponding polygons from the triangulation on the sphere
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     # ax.scatter(xn,yn,zn, c='r', marker='o')
    #     # ax.plot3D(xn,yn,zn, c='grey', alpha=0.5)
    #     for simplex in tri.simplices:
    #         ax.plot3D(xn[simplex], yn[simplex], zn[simplex], c='k')


    #     ax.set_box_aspect([1,1,1])
    #     ax.set_proj_type('ortho')
    #     set_axes_equal(ax)

    #     plt.show()





    # # compute the grid size of points in lat/lon space
    # # convert xn,yn,zn to lat/lon
    # lat = np.arcsin(zn)
    # lon = np.arctan2(yn, xn)
    # lat,lon = np.rad2deg(lat), np.rad2deg(lon)

    # if False:
    #     #plot lat lon points and triangulation
    #     plt.scatter(lon,lat, c='r', marker='o')
    #     for simplex in tri.simplices:
    #         plt.plot(lon[simplex], lat[simplex], c='k')

    #     plt.show()


    # #collect together all edges of the triangles (in lat/lon space)
    # edges = np.concatenate([
    #     #edge 1
    #     [
    #         lon[tri.simplices[:,0]] - lon[tri.simplices[:,1]],
    #         lat[tri.simplices[:,0]] - lat[tri.simplices[:,1]]
    #     ],
    #     #edge 2
    #     [
    #         lon[tri.simplices[:,1]] - lon[tri.simplices[:,2]],
    #         lat[tri.simplices[:,1]] - lat[tri.simplices[:,2]]
    #     ],
    #     #edge 3
    #     [
    #         lon[tri.simplices[:,2]] - lon[tri.simplices[:,0]],
    #         lat[tri.simplices[:,2]] - lat[tri.simplices[:,0]],
    #     ]
    # ], axis=1)


    # horizontal = edges[:, np.abs(edges[1]) < 1e-6]
    # vertical = edges[:, np.abs(edges[0]) < 1e-6]

    # dlon = np.abs(horizontal[0])
    # dlat = np.abs(vertical[1])


    # #plot histograms of horizontal and vertical spacings
    # fig, ax = plt.subplots(1,2)
    # ax[0].hist(dlon, bins=20)
    # ax[1].hist(dlat, bins=20)
    # plt.show()


    # pdb.set_trace()



if __name__ == '__main__':
    main()