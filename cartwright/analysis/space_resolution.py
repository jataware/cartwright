from typing import Optional, Tuple
import numpy as np
from scipy.spatial import Delaunay
from ..schemas import AngleUnit, Resolution, GeoSpatialResolution
from .helpers import get_uniformity, match_unit



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


def detect_latlon_resolution(lat:np.ndarray, lon:np.ndarray) -> Optional[GeoSpatialResolution]:
    """
    Detect if the lat/lon coordinates are drawn from a uniform grid.

    @param lat: a numpy array of latitudes in [DEGREES]
    @param lon: a numpy array of longitudes in [DEGREES]

    @return: (optional) GeoSpatialResolution with either 
        - square = Resolution
        - lat = Resolution, lon = Resolution
        
    where `square` indicates that the detected grid has the same resolution in both dimensions
    while `lat`/`lon` indicate that the detected grid has different resolutions for lat/lon
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
        return None

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
        error = np.rad2deg(np.abs(deltas-avg).mean()) / unit

        return GeoSpatialResolution(square=Resolution(uniformity, unit, scale, error))
    

    # rectangular grid
    dlon_uniformity = get_uniformity(dlon, dlon_avg)
    dlon_scale, dlon_unit = match_unit(AngleUnit, np.rad2deg(dlon_avg))
    dlon_error = np.rad2deg(np.abs(1 - dlon/dlon_avg).mean()) / dlon_unit

    dlat_uniformity = get_uniformity(dlat, dlat_avg)
    dlat_scale, dlat_unit = match_unit(AngleUnit, np.rad2deg(dlat_avg))
    dlat_error = np.rad2deg(np.abs(1 - dlat/dlat_avg).mean()) / dlat_unit

    return GeoSpatialResolution(
        lat=Resolution(dlat_uniformity, dlat_unit, dlat_scale, dlat_error),
        lon=Resolution(dlon_uniformity, dlon_unit, dlon_scale, dlon_error)
    )