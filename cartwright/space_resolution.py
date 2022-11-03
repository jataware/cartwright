from typing import Optional, Dict, Tuple
import numpy as np
from scipy.stats import norm
from scipy.spatial import Delaunay
from cartwright.schemas import Uniformity#, SpaceResolution, LatLonResolution, SphericalResolution, CategoricalResolution
from enum import Enum, auto

from matplotlib import pyplot as plt
import pdb


#TODO: move these to schema or perhaps a dedicated file for space_schemas
#TODO: some sort of coverage metric?

from dataclasses import dataclass

class GeoUnit(Enum):
    DEGREES = auto()
    KILOMETERS = auto()

@dataclass
class Resolution:
    uniformity: Uniformity
    resolution: float
    unit: GeoUnit
    error: float


@dataclass
class CategoricalResolution:
    uniformity: Uniformity
    # category: GeoCategory
    # coverage: float # percent of category elements covered

@dataclass
class GeoSpatialResolution:
    lat: Optional[Resolution]
    lon: Optional[Resolution]
    latlon: Optional[Resolution]
    spherical: Optional[Resolution]
    categorical: Optional[CategoricalResolution] #TODO: this could maybe be a list?
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
    filter out non-unique points, and optionally convert to radians
    
    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]

    @return: lat, lon
    """

    latlon = np.stack([lat, lon], axis=1)
    latlon = np.unique(latlon, axis=0)
    if rad: 
        latlon = np.deg2rad(latlon)
    lat, lon = np.rollaxis(latlon, 1)

    return lat, lon


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
    spherical_resolution = detect_spherical_resolution(lat, lon)
    print(spherical_resolution, '\n\n')
    pdb.set_trace()

    categorical_resolution = detect_categorical_resolution(lat, lon)

    return GeoSpatialResolution(**latlon_resolution, **spherical_resolution, **categorical_resolution)



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
        error = np.abs(1 - deltas/avg).mean()

        return {'latlon': Resolution(uniformity, np.rad2deg(avg), GeoUnit.DEGREES, np.rad2deg(error))}
    

    # rectangular grid
    dlon_uniformity = get_uniformity(dlon, dlon_avg)
    dlon_error = np.abs(1 - dlon/dlon_avg).mean()
    dlat_uniformity = get_uniformity(dlat, dlat_avg)
    dlat_error = np.abs(1 - dlat/dlat_avg).mean()

    return {
        'lat':Resolution(dlat_uniformity, np.rad2deg(dlat_avg), GeoUnit.DEGREES, np.rad2deg(dlat_error)),
        'lon':Resolution(dlon_uniformity, np.rad2deg(dlon_avg), GeoUnit.DEGREES, np.rad2deg(dlon_error))
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

    #collect together all edges of the triangles (in 3D space)
    i0, i1, i2 = tri.simplices[:,0], tri.simplices[:,1], tri.simplices[:,2]
    p0 = np.stack([x[i0], y[i0], z[i0]], axis=1)
    p1 = np.stack([x[i1], y[i1], z[i1]], axis=1)
    p2 = np.stack([x[i2], y[i2], z[i2]], axis=1)

    #compute the great circle distance between each pair of points
    d01 = np.arccos((p0 * p1).sum(axis=1))
    d12 = np.arccos((p1 * p2).sum(axis=1))
    d20 = np.arccos((p2 * p0).sum(axis=1))

    deltas = np.concatenate([d01, d12, d20])
    avg = np.median(deltas)
    uniformity = get_uniformity(deltas, avg)
    error = np.abs(1 - deltas/avg).mean()

    return {'spherical': Resolution(uniformity, np.rad2deg(avg), GeoUnit.DEGREES, np.rad2deg(error))}


def detect_categorical_resolution(lat:np.ndarray, lon:np.ndarray) -> Optional[CategoricalResolution]:
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
    n_points = 100

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

    def generate_latlon_grid(n_points:int) -> Tuple[np.ndarray, np.ndarray]:
        """generate points at uniform lat/lon intervals"""# and then convert to 3D
        d_points = int(np.round((n_points/2) ** (1/2)))
        lats = (np.arange(d_points) + 0.5) / d_points * 180 - 90
        lons = (np.arange(2*d_points) + 0.5) / (2 * d_points) * 360 - 180
        lat,lon = np.meshgrid(lats, lons)
        lat,lon = lat.flatten(), lon.flatten()
        return lat, lon


    # x,y,z = sphere_from_area(*uniform_square(n_points, True))
    x,y,z = fibonacci_sphere(n_points)

    #DEBUG plot points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    set_axes_equal(ax)
    plt.show()
    pdb.set_trace()


    detect_resolution(lat, lon)
    
    

def test1():
    n_points = 64800
    d_points = int(np.round((n_points/2) ** (1/2)))
    lats = (np.arange(d_points) + 0.5) / d_points * 180 - 90
    lons = (np.arange(2*d_points) + 0.5) / (2 * d_points) * 360 - 180
    lat,lon = np.meshgrid(lats, lons)
    lat,lon = lat.flatten(), lon.flatten()
    res = detect_latlon_resolution(lat, lon)

    #TODO: test, should be perfectly uniform at 1 degree











    
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