from typing import Tuple
from scipy.stats import norm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from cartwright.space_resolution import preprocess_latlon, detect_resolution, detect_latlon_resolution

import pdb

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