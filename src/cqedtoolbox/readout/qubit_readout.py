from typing import Tuple, Optional

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from labcore.analysis import Node


def rotate_complex_qubit_data(
    dset: xr.Dataset, 
    cal_dset: Optional[xr.Dataset] = None,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """rotate complex qubit readout data to real data.
    Uses sklearn PCA to find the best rotation angle.

    Parameters
    ----------
    dset : xr.Dataset
        uncalibrated data.
    cal_dset : xr.Dataset
        needs a 2D dataset, (N x 2); the two columns are for g and e.
    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        rotated data, rotated calibration data.
    """

    if cal_dset is not None:
        gr_samples = cal_dset["signal_Re"][:,1].values
        gi_samples = cal_dset["signal_Im"][:,1].values
        er_samples = cal_dset["signal_Re"][:,0].values
        ei_samples = cal_dset["signal_Im"][:,0].values

        cal_samples_Re = np.concatenate((gr_samples.flatten(), er_samples.flatten()))
        cal_samples_Im = np.concatenate((gi_samples.flatten(), ei_samples.flatten()))

    for var, vvar in Node.complex_dependents(dset).items():
        r, i = dset[vvar["real"]], dset[vvar["imag"]]
        shp = r.shape
        pca = PCA(n_components=1)
    
        if cal_dset is not None:
            cal_rotated = pca.fit_transform(
                np.vstack((cal_samples_Re, cal_samples_Im)).T,
            ).flatten()

            dset[var] = xr.DataArray(
                pca.transform(
                    np.vstack((r.values.flatten(), i.values.flatten())).T
                ).reshape(shp),
                dims=dset[vvar["real"]].dims,
            )
        else:
            dset[var] = xr.DataArray(
                pca.fit_transform(
                    np.vstack((r.values.flatten(), i.values.flatten())).T
                ).reshape(shp),
                dims=dset[vvar["real"]].dims,
            )
        
        dset[var].attrs = dset[vvar["real"]].attrs
        dset = dset.drop_vars(vvar["real"]).drop_vars(vvar["imag"])

    if cal_dset is not None:
        cal_rotated = xr.Dataset(
            {
                "g": (["repetition"], cal_rotated[: len(gr_samples)]),
                "e": (["repetition"], cal_rotated[len(gr_samples) :]),
            },
            coords={
                "repetition": np.arange(len(gr_samples)),
            },
        )
    else:
        cal_rotated = None
        
    return dset, cal_rotated


def qubit_readout_calibrated_average(
    dset: xr.Dataset,
    cal_dset: xr.Dataset,
    g_err: float = 0,
    e_err: float = 0,
    rotate: bool = True,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """transfrom qubit data to excited state probability using a calibration.

    Parameters
    ----------
    dset : xr.Dataset
        uncalibrated qubit data
    cal_dset : xr.Dataset
        calibration data, contains two data vars, "g" and "e".
        Should have only one independent, "repetition".
        "g" should contain shots taken with qubit prepared in ground state,
        "e" contains shots with qubit prepared in excited state.
    g_err : float, optional
        error probability for g preparation, by default 0
    e_err : float, optional
        error probability for e preparation, by default 0
    rotate : bool, optional
        if True, perform rotation first, by default True

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        transformed dataset, and calibration dataset (rotated)
    """
    if rotate:
        dset, cal_dset = rotate_complex_qubit_data(dset, cal_dset)

    g_mean = float(cal_dset.g.mean().values)
    e_mean = float(cal_dset.e.mean().values)

    avg_dset = Node.mean(dset, "repetition")
    ind, dep = Node.data_dims(avg_dset)
    for d in dep:
        v = avg_dset[d]
        avg_dset[d + "_Pe"] = -(
            (v - g_mean) / (g_mean - e_mean)
            + (g_err * (e_mean - v) + e_err * (g_mean - v)) / (g_mean - e_mean)
        )
        avg_dset = avg_dset.drop_vars(d)
        avg_dset[d].attrs = dset[d].attrs

    return avg_dset, cal_dset


def kmeans_calibration(data, name, g_center_guess, e_center_guess):
    reals = data[f'{name}_Re'].values.flatten()
    imags = data[f'{name}_Im'].values.flatten()
    X = np.vstack((reals, imags)).T
    km = KMeans(
        init=[g_center_guess, e_center_guess],
        n_clusters=2, 
        n_init='auto',
    ).fit(X)
    data.attrs['kmeans_cluster_centers'] = km.cluster_centers_
    return km

def apply_kmeans_calibration(data, name, km):
    dsre = data[f'{name}_Re']
    dsim = data[f'{name}_Im']
    shp = dsre.shape
    X = np.vstack((dsre.values.flatten(), dsim.values.flatten())).T
    lbls_ = km.predict(X).reshape(shp)
    lbls = xr.DataArray(
        name='label',
        data=lbls_,
        dims=dsre.dims,
    )
    data[lbls.name] = lbls
    return data

def lbl2prob(data, name='label', avg_dim='repetition', nlbls=2):
    for i in range(nlbls):
        pofi = (data[name]==i).astype(int).mean(avg_dim, keep_attrs=True)
        pofi.name = f"Pr_{i}"
        data[pofi.name] = pofi
    return data


def qubit_readout_label_and_average(data, var_name, caldata, cal_var_name, center_guesses):
    km = kmeans_calibration(
        caldata, cal_var_name,
        *center_guesses
    )
    caldata = apply_kmeans_calibration(caldata, cal_var_name, km)
    caldata = lbl2prob(caldata)
    data = apply_kmeans_calibration(data, var_name, km)
    data = lbl2prob(data)
    
    return data, caldata
