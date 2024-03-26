# Author :udit
# Created on : 22/03/24
# Features :
import numpy as np

def tounit(vec: np.ndarray) -> np.ndarray:
    """converts into unitvectors , normalized form"""
    if vec.ndim == 1: return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]


def vec_angle(svec, tvec, normalize=True, maxang=180, signed=False, units="deg") -> np.ndarray:
    """
    Calculates the angles in rad or deg between source vectors to target vector
     Args:
        svec: (3) source vector
        tvec: (N,3) or (3) target vector/s from where to check angle
        normalize: True if normalization is needed to be done on vectors
        maxang: outputs in [0,maxang] range. Usually [0,90] default it is [0,180]
        signed: if clock or anticlockwise angles needed
        units: format of unit required as output, "rad": in radians, "deg": in degrees (default).

    Returns: (N) array of angles between vectors.
    """
    if isinstance(svec, (list, tuple)):
        svec = np.asarray(svec)
    if isinstance(tvec, (list, tuple)):
        tvec = np.asarray(tvec)
    if normalize:
        svec = svec / np.linalg.norm(svec)
        tvec = tvec / np.linalg.norm(tvec)
    dotprod = np.einsum("ij,ij->i", svec.reshape(-1, 3), tvec.reshape(-1, 3))
    angles = np.arccos(np.clip(dotprod, -1.0, 1.0))
    if units == 'deg':
        angles = np.degrees(angles)
        if maxang == 90:angles = np.where(angles > maxang, 180 - angles, angles)
    return np.around(angles, decimals=2)


def Point2PointDist(points: np.ndarray, ref: np.ndarray, positive=True) -> np.ndarray:
    # Euclidian Distance which is always positive!
    return np.linalg.norm((points - ref), axis=1)

def getpointonline(point, norm, at_distance=0.5, mode='pos'):
    # get new point in direction of line.
    if mode == 'pos':
        return point + norm * at_distance
    return point - norm * at_distance
