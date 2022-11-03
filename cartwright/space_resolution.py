from typing import Optional, Dict
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

# @dataclass
# class SphericalResolution:
#     uniformity: Uniformity
#     #TODO: some sort of coverage metric?
#     density: float
#     error: float
#     #TODO:maybe a distribution of the spacings?

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

    #filter out non-unique points, and convert to radians
    latlon = np.stack([lat, lon], axis=1)
    latlon = np.unique(latlon, axis=0)
    lat,lon = np.rollaxis(np.deg2rad(latlon), 1)

    #compute the Delaunay triangulation on the lat/lon points
    tri = Delaunay(latlon)

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
    if len(horizontal.T) + len(vertical.T) < len(edges.T) / 3: #should be around 2/3 if it was a full grid, 1/3 if only horizontal or vertical
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

        return {'latlon': Resolution(uniformity, avg, GeoUnit.DEGREES, error)}
    

    # rectangular grid
    dlon_uniformity = get_uniformity(dlon, dlon_avg)
    dlon_errors = np.abs(1 - dlon/dlon_avg).mean()
    dlat_uniformity = get_uniformity(dlat, dlat_avg)
    dlat_errors = np.abs(1 - dlat/dlat_avg).mean()

    return {
        'lat':Resolution(dlat_uniformity, dlat_avg, GeoUnit.DEGREES, dlat_errors),
        'lon':Resolution(dlon_uniformity, dlon_avg, GeoUnit.DEGREES, dlon_errors)
    }


def detect_spherical_resolution(lat:np.ndarray, lon:np.ndarray) -> Dict[str, Resolution]:
    """
    Detect if points are uniformly distributed on a sphere.

    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]

    @return: {'spherical':Resolution} | {}
    """
    
    #filter out non-unique points, and convert to radians
    latlon = np.stack([lat, lon], axis=1)
    latlon = np.unique(latlon, axis=0)
    lat,lon = np.rollaxis(np.deg2rad(latlon), 1)

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
    tri = Delaunay(np.array([X,Y]).T)
    # tri = Delaunay(np.stack([x,y,z], axis=1))

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

    lengths = np.sqrt(np.sum(edges**2, axis=0))

    #TODO: construct a results object based on these lengths
    pdb.set_trace()

def detect_categorical_resolution(lat:np.ndarray, lon:np.ndarray) -> Optional[CategoricalResolution]:
    pdb.set_trace()


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
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
    #test conversion to uniform
    # p = 10000
    # x = (np.arange(p) + 0.5) / p
    # xn = norm.ppf(x)
    # plt.hist(xn, bins=100)
    # plt.show()
    # pdb.set_trace()

    #Some experiments with plotting points on a sphere
    n_points = 180
    d_points = int(np.round(n_points ** (1/3)))

    side = (np.arange(d_points) + 0.5) / d_points
    x,y,z = np.meshgrid(side, side, side)
    x,y,z = x.flatten(), y.flatten(), z.flatten()

    #convert to 3D coordinates on a sphere via normal distribution
    xn,yn,zn = norm.ppf(x), norm.ppf(y), norm.ppf(z) #inverse normal transform
    r = np.sqrt(xn**2 + yn**2 + zn**2)
    xn,yn,zn = xn/r, yn/r, zn/r


    #DEBUG just generate random (normal) points for xn,yn,zn instead of uniformly distributed
    xn,yn,zn = np.random.normal(size=(3,n_points))
    r = np.sqrt(xn**2 + yn**2 + zn**2)
    xn,yn,zn = xn/r, yn/r, zn/r


    #DEBUG generate points at uniform lat/lon intervals and then convert to 3D
    d_points = int(np.round((n_points/2) ** (1/2)))
    lats = (np.arange(d_points) + 0.5) / d_points * 180 - 90
    lons = (np.arange(2*d_points) + 0.5) / (2 * d_points) * 360 - 180
    lat,lon = np.meshgrid(lats, lons)
    lat,lon = lat.flatten(), lon.flatten()
    detect_resolution(lat, lon)
    exit(1)
    
    
    
    lat,lon = np.deg2rad(lat), np.deg2rad(lon)
    xn,yn,zn = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)



    if False:
        #plot the points in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xn, yn, zn)
        ax.set_box_aspect([1,1,1])
        ax.set_proj_type('ortho')
        set_axes_equal(ax)

        #plot the lat/lon points in 2D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='polar')
        # ax.scatter(lon, lat)

        plt.show()

    #2 side by side plots in 3D
    # fig, ax = plt.subplots(1,2, subplot_kw={'projection':'3d'})
    # ax[0].scatter(x,y,z, c='r', marker='o')
    # ax[0].plot3D(x,y,z, c='grey', alpha=0.5)
    # ax[1].scatter(xn,yn,zn, c='r', marker='o')
    # ax[1].plot3D(xn,yn,zn, c='grey', alpha=0.5)

    # for a in ax:
    #     a.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    #     a.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    #     set_axes_equal(a) # IMPORTANT - this is also required


    # plt.show()






    #compute an orthographic projection of the points on the sphere
    X = xn/(1-zn)
    Y = yn/(1-zn)

    #compute the Delaunay triangulation of the points
    tri = Delaunay(np.array([X,Y]).T)

    # #plot the points and the triangulation
    # fig, ax = plt.subplots(1,2, subplot_kw={'aspect':'equal'})
    # ax[0].scatter(X,Y, c='r', marker='o')
    # ax[0].plot(X,Y, c='grey', alpha=0.5)
    # ax[1].scatter(X,Y, c='r', marker='o')
    # # ax[1].plot(X,Y, c='grey', alpha=0.5)
    # for simplex in tri.simplices:
    #     ax[1].plot(X[simplex], Y[simplex], c='k')

    # plt.show()

    if False:
        #draw corresponding polygons from the triangulation on the sphere
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(xn,yn,zn, c='r', marker='o')
        # ax.plot3D(xn,yn,zn, c='grey', alpha=0.5)
        for simplex in tri.simplices:
            ax.plot3D(xn[simplex], yn[simplex], zn[simplex], c='k')


        ax.set_box_aspect([1,1,1])
        ax.set_proj_type('ortho')
        set_axes_equal(ax)

        plt.show()





    # compute the grid size of points in lat/lon space
    # convert xn,yn,zn to lat/lon
    lat = np.arcsin(zn)
    lon = np.arctan2(yn, xn)
    lat,lon = np.rad2deg(lat), np.rad2deg(lon)

    if False:
        #plot lat lon points and triangulation
        plt.scatter(lon,lat, c='r', marker='o')
        for simplex in tri.simplices:
            plt.plot(lon[simplex], lat[simplex], c='k')

        plt.show()


    #collect together all edges of the triangles (in lat/lon space)
    edges = np.concatenate([
        #edge 1
        [
            lon[tri.simplices[:,0]] - lon[tri.simplices[:,1]],
            lat[tri.simplices[:,0]] - lat[tri.simplices[:,1]]
        ],
        #edge 2
        [
            lon[tri.simplices[:,1]] - lon[tri.simplices[:,2]],
            lat[tri.simplices[:,1]] - lat[tri.simplices[:,2]]
        ],
        #edge 3
        [
            lon[tri.simplices[:,2]] - lon[tri.simplices[:,0]],
            lat[tri.simplices[:,2]] - lat[tri.simplices[:,0]],
        ]
    ], axis=1)


    horizontal = edges[:, np.abs(edges[1]) < 1e-6]
    vertical = edges[:, np.abs(edges[0]) < 1e-6]

    dlon = np.abs(horizontal[0])
    dlat = np.abs(vertical[1])


    #plot histograms of horizontal and vertical spacings
    fig, ax = plt.subplots(1,2)
    ax[0].hist(dlon, bins=20)
    ax[1].hist(dlat, bins=20)
    plt.show()


    pdb.set_trace()



if __name__ == '__main__':
    main()