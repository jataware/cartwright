from typing import Dict, Tuple
import numpy as np
from scipy.spatial import Delaunay
from ..schemas import Unit, Uniformity, AngleUnit, Resolution, GeoSpatialResolution
from enum import Enum, EnumMeta
import pandas as pd

from matplotlib import pyplot as plt
import pdb




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

    @return: GeoSpatialResolution(<TODO>) object where 
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

    @return: {'square': Resolution()} | {'lat': Resolution(), 'lon': Resolution()} | {}
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

        return {'square': Resolution(uniformity, unit, scale, error)}
    

    # rectangular grid
    dlon_uniformity = get_uniformity(dlon, dlon_avg)
    dlon_scale, dlon_unit = match_unit(AngleUnit, np.rad2deg(dlon_avg))
    dlon_error = np.rad2deg(np.abs(1 - dlon/dlon_avg).mean()) / dlon_unit

    dlat_uniformity = get_uniformity(dlat, dlat_avg)
    dlat_scale, dlat_unit = match_unit(AngleUnit, np.rad2deg(dlat_avg))
    dlat_error = np.rad2deg(np.abs(1 - dlat/dlat_avg).mean()) / dlat_unit

    return {
        'lat':Resolution(dlat_uniformity, dlat_unit, dlat_scale, dlat_error),
        'lon':Resolution(dlon_uniformity, dlon_unit, dlon_scale, dlon_error)
    }


# def detect_spherical_resolution(lat:np.ndarray, lon:np.ndarray) -> Dict[str, Resolution]:
#     """
#     Detect if points are uniformly distributed on a sphere.

#     @param lat: a numpy array of latitudes in [DEGREES]
#     @param lon: a numpy array of longitudes in [DEGREES]

#     @return: {'spherical':Resolution()} | {}
#     """
    
#     #filter out non-unique points, and convert to radians
#     lat, lon = preprocess_latlon(lat, lon, rad=True)

#     #fail if not enough points
#     if lat.size <= 2:
#         return {}

#     #convert to 3D cartesian coordinates
#     x = np.cos(lat) * np.cos(lon)
#     y = np.cos(lat) * np.sin(lon)
#     z = np.sin(lat)

#     #filter any points at the north pole
#     mask = np.abs(z) < 1.0-1e-9
#     x,y,z = x[mask], y[mask], z[mask]

#     #compute an orthographic projection of the points on the sphere
#     X = x/(1-z)
#     Y = y/(1-z)

#     #compute the Delaunay triangulation of the projected points (this is equivalent to a Delaunay triangulation directly on the sphere)
#     tri = Delaunay(np.stack([X,Y], axis=1))

#     #create a sparse adjacency matrix from the triangulation
#     adj = csr_matrix((np.ones(len(tri.simplices)), (tri.simplices[:,0], tri.simplices[:,1])), shape=(len(X), len(X)))
#     adj += csr_matrix((np.ones(len(tri.simplices)), (tri.simplices[:,1], tri.simplices[:,2])), shape=(len(X), len(X)))
#     adj += csr_matrix((np.ones(len(tri.simplices)), (tri.simplices[:,2], tri.simplices[:,0])), shape=(len(X), len(X)))

#     pdb.set_trace()


#     #collect together all edges of the triangles (in 3D space)
#     i0, i1, i2 = tri.simplices[:,0], tri.simplices[:,1], tri.simplices[:,2]
#     p0 = np.stack([x[i0], y[i0], z[i0]], axis=1)
#     p1 = np.stack([x[i1], y[i1], z[i1]], axis=1)
#     p2 = np.stack([x[i2], y[i2], z[i2]], axis=1)

#     #TODO: come up with a good way to characterize the uniformity of the points, based on the triangulation


#     #compute the areas of the triangles
#     areas = np.linalg.norm(np.cross(p1-p0, p2-p0), axis=1)
#     plt.hist(areas, bins=100)
#     plt.show()
#     pdb.set_trace()

#     #compute the great circle distance between each pair of points
#     d01 = np.arccos((p0 * p1).sum(axis=1))
#     d12 = np.arccos((p1 * p2).sum(axis=1))
#     d20 = np.arccos((p2 * p0).sum(axis=1))

#     deltas = np.concatenate([d01, d12, d20])
#     avg = np.median(deltas)
#     uniformity = get_uniformity(deltas, avg)
#     error = np.abs(1 - deltas/avg).mean()

#     #DEBUG plot points and edges in 3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x,y,z)
#     for i in range(len(tri.simplices)):
#         ax.plot([x[i0[i]], x[i1[i]]], [y[i0[i]], y[i1[i]]], [z[i0[i]], z[i1[i]]], 'k-')
#         ax.plot([x[i1[i]], x[i2[i]]], [y[i1[i]], y[i2[i]]], [z[i1[i]], z[i2[i]]], 'k-')
#         ax.plot([x[i2[i]], x[i0[i]]], [y[i2[i]], y[i0[i]]], [z[i2[i]], z[i0[i]]], 'k-')
#     set_axes_equal(ax)
#     plt.show()

#     return {'spherical': Resolution(uniformity, np.rad2deg(avg), GeoUnit.DEGREES, np.rad2deg(error))}


# def detect_categorical_resolution(lat:np.ndarray, lon:np.ndarray) -> Optional[CategoricalResolution]:
#     #edge cases/issues
#     # - how stable are the categories? countries come and go, change names, etc. would be a pain to have to make this aware of the date, but how else do you handle?
#     #    - possibly just assume the date is now?
#     #    - possibly have the user specify the date/handling process?
#     #
#     pdb.set_trace()


