from typing import Optional
import numpy as np
from scipy.stats import norm
from scipy.spatial import Delaunay
from cartwright.schemas import SpaceResolution, LatLonResolution, SphericalResolution, CategoricalResolution


from matplotlib import pyplot as plt
import pdb

def detect_resolution(lat:np.ndarray, lon:np.ndarray) -> SpaceResolution:
    """
    Detect the resolution of the lat/lon coordinates.

    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]
    """

    #detect the lat/lon resolution
    # latlon_resolution = detect_latlon_resolution(lat, lon)
    spherical_resolution = detect_spherical_resolution(lat, lon)
    categorical_resolution = detect_categorical_resolution(lat, lon)

    #TODO: other checks, i.e. spherical, categorical, etc.

    return SpaceResolution(
        latlon_resolution=latlon_resolution,
        spherical_resolution=spherical_resolution,
        categorical_resolution=categorical_resolution
    )

def detect_latlon_resolution(lat:np.ndarray, lon:np.ndarray) -> Optional[LatLonResolution]:
    """
    Detect if the lat/lon coordinates are drawn from a uniform grid.
    NOTE: assumes all lat/lon pairs are unique.

    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]
    """

    #filter out non-unique points, and convert to radians
    latlon = np.stack([lat, lon], axis=1)
    latlon = np.unique(latlon, axis=0)
    lat,lon = np.rollaxis(np.deg2rad(latlon), 1)

    #convert to 3D cartesian coordinates
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    # #filter any points at the north pole
    # mask = np.abs(z) < 1.0-1e-9
    # x,y,z = x[mask], y[mask], z[mask]

    # #compute an orthographic projection of the points on the sphere
    # X = x/(1-z)
    # Y = y/(1-z)

    # #TODO: can we just run the Delaunay triangulation directly on the lat/lon points?

    # #compute the Delaunay triangulation of the points
    # tri = Delaunay(np.array([X,Y]).T)
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

    #DEBUG plot the horizontal and vertical edges in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, marker='.')
    for tri in tri.simplices:
        if np.abs(lat[tri[0]] - lat[tri[1]]) < 1e-6:
            ax.plot(x[tri[0:2]], y[tri[0:2]], z[tri[0:2]], color='red')
        if np.abs(lat[tri[1]] - lat[tri[2]]) < 1e-6:
            ax.plot(x[tri[1:3]], y[tri[1:3]], z[tri[1:3]], color='red')
        if np.abs(lat[tri[2]] - lat[tri[0]]) < 1e-6:
            ax.plot(x[tri[0:3:2]], y[tri[0:3:2]], z[tri[0:3:2]], color='red')
    
        if np.abs(lon[tri[0]] - lon[tri[1]]) < 1e-6:
            ax.plot(x[tri[0:2]], y[tri[0:2]], z[tri[0:2]], color='blue')
        if np.abs(lon[tri[1]] - lon[tri[2]]) < 1e-6:
            ax.plot(x[tri[1:3]], y[tri[1:3]], z[tri[1:3]], color='blue')
        if np.abs(lon[tri[2]] - lon[tri[0]]) < 1e-6:
            ax.plot(x[tri[0:3:2]], y[tri[0:3:2]], z[tri[0:3:2]], color='blue')

    ax.set_box_aspect([1,1,1])
    ax.set_proj_type('ortho')
    set_axes_equal(ax)
    plt.show()



    dlon = np.abs(horizontal[0])
    dlat = np.abs(vertical[1])

    pdb.set_trace()


def detect_spherical_resolution(lat:np.ndarray, lon:np.ndarray) -> Optional[SphericalResolution]:
    
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

    #find edges that are either horizontal or vertical
    horizontal = edges[:, np.abs(edges[1]) < 1e-6]
    vertical = edges[:, np.abs(edges[0]) < 1e-6]

    #DEBUG plot the horizontal and vertical edges in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, marker='.')
    for tri in tri.simplices:
        if np.abs(lat[tri[0]] - lat[tri[1]]) < 1e-6:
            ax.plot(x[tri[0:2]], y[tri[0:2]], z[tri[0:2]], color='red')
        if np.abs(lat[tri[1]] - lat[tri[2]]) < 1e-6:
            ax.plot(x[tri[1:3]], y[tri[1:3]], z[tri[1:3]], color='red')
        if np.abs(lat[tri[2]] - lat[tri[0]]) < 1e-6:
            ax.plot(x[tri[0:3:2]], y[tri[0:3:2]], z[tri[0:3:2]], color='red')
    
        if np.abs(lon[tri[0]] - lon[tri[1]]) < 1e-6:
            ax.plot(x[tri[0:2]], y[tri[0:2]], z[tri[0:2]], color='blue')
        if np.abs(lon[tri[1]] - lon[tri[2]]) < 1e-6:
            ax.plot(x[tri[1:3]], y[tri[1:3]], z[tri[1:3]], color='blue')
        if np.abs(lon[tri[2]] - lon[tri[0]]) < 1e-6:
            ax.plot(x[tri[0:3:2]], y[tri[0:3:2]], z[tri[0:3:2]], color='blue')

    ax.set_box_aspect([1,1,1])
    ax.set_proj_type('ortho')
    set_axes_equal(ax)
    plt.show()

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