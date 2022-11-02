from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.spatial import Delaunay
from typing import Optional
from cartwright.schemas import LatLonResolution


import pdb

def detect_latlon_resolution(lats:np.ndarray, lons:np.ndarray) -> Optional[LatLonResolution]:
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
    n_points = 1000
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