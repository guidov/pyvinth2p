from cython_vinth2p_4d import vinth2p_ecmwf_fast_4d
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Process some variables.')
parser.add_argument('--infile', type=str, required=True, help='Input file path')
parser.add_argument('--var', type=str, required=True, help='Variable name to process')

# Parse the arguments
args = parser.parse_args()

# Define functions and wrappers to perform hybrid levels to pressure coordinate transformation
# This version of the code is used on timeseries with all variables in one file (e.g. ncrcat ...)
# The required variables are: Z3,T,PS,PHIS

def vinth2p_ecmwf_wrap(
    var3d, hbcoefa, hbcoefb, p0, varps, plevo, phis, tbot, varint, intyp, extrapp, spval
):
    """
    This is a static type wrapper for the cythonized version of the vinth2p_ecmwf.
    The orginal code was a fortran version that used an NCL function.
    Some numpy operations on the model variables will change the static type.
    Original cython code created by Deepak Chandan.
    """
    var3d = np.array(var3d).astype(np.float32)
    hbcoefa = np.array(hbcoefa).astype(np.float64)
    hbcoefb = np.array(hbcoefb).astype(np.float64)
    p0 = float(p0)
    varps = np.array(varps).astype(np.float32)
    plevo = np.array(plevo).astype(np.float64)
    phis = np.array(phis).astype(np.float32)
    tbot = np.array(tbot).astype(np.float32)
    varint = int(varint)
    intyp = int(intyp)
    extrapp = int(extrapp)
    spval = float(spval)

    var_p = vinth2p_ecmwf_fast_4d(
        var3d,
        hbcoefa,
        hbcoefb,
        p0,
        varps,
        plevo,
        phis,
        tbot,
        varint,
        intyp,
        extrapp,
        spval,
    )
    return var_p


def h2pvar(cesmxr, var, plevs, intyp="lin"):
    """
    Convert hybrid coordinate level CESM variable to pressure levels using
    vinth2p_ecmwf_fast converted from ncl and cythonized NCL code.
    This function expects the cesm xarray of monthly data or a minimum
    number of a set of cesm variables in an xarray. This function can
    loop over the time variable.

    Args:
        cesmxr : input data in a xarray format
        var : the name of the 3-D variable which you want to trasform
        plevs : output pressure levels (units: hPa)
        intyp : specify the interpolation type. Either of: "lin" for linear, "log"
                for log or "loglog" for log-log interpolation (default = "lin")

    Returns:
        3D array of shape (nplevo, nlat, nlon) or 4D array of shape (time nplevo, nlat, nlon)
        where nplevo is the length of the input pressure levels variable plevo

    Default Settings:
        phis : surface geopotential height (units: m2/s2)
        tbot : temperature at level closest to ground (units: K). Only used
               if interpolating geopotential height
        varint : one of: "T"=1, "Z"=2 or None=3 to indicate whether interpolating temperature,
              geopotential height, or some other variable
        intyp : specify the interpolation type. Either of: "lin=1" for linear, "log=2"
                for log or "loglog=3" for log-log interpolation
        extrapp 0 = no extrapolation when the pressure level is outside of the range of psfc.


    """
    try:
        hbcoefa = cesmxr["hyam"].data
    except:
        raise Exception("No hyam varibale in xarray")
    try:
        hbcoefb = cesmxr["hybm"].data
    except:
        raise Exception("No hybm varibale in xarray")

    p0 = 1000.0  # hPa
    plevo = plevs
    spval = 9.96921e36 # missing value
    extrapp = 1

    # check dimensions on the dataset
    ndim = cesmxr[var].ndim
    if (ndim < 3) and (ndim > 4):
        raise ValueError(
            "Error: Working with CESM variable array that is not"
            "of dimension (time,ilev,lat,lon)"
            "or of dimension (ilev,lat,lon)"
        )
    try:
        ntime = cesmxr["time"].data.shape[0]
    except:
        ntime = None

    # why did I do this ???
    try:
        varint = ["T", "Z3"].index(var) + 1
    except:
        varint = 3

    try:
        varps = cesmxr["PS"].data / 100.0  # hPa
    except:
        raise Exception("No PS varibale in xarray")
    try:
        phis = cesmxr["PHIS"].data
    except:
        raise Exception("No PHIS varibale in xarray")
    try:
        tlev = cesmxr.coords["lev"].data.shape[0] - 1 
        tbot = cesmxr["T"].isel(lev=tlev).data
    except:
        raise Exception("problem with tlev or T varibale in xarray")
    # Get the 3D or 4D variable that needs to be converted
    try:
        varxd = np.ma.masked_invalid(cesmxr[var].data)
    except:
        raise Exception("No {} varibale in xarray".format(var))
    try:
        intyp = ["lin", "log"].index(var) + 1
    except:
        intyp = 3
    # check for file that has no time dimension
    if ntime == None:
        pvar = vinth2p_ecmwf_wrap(
            varxd,
            hbcoefa,
            hbcoefb,
            p0,
            varps,
            plevo,
            phis,
            tbot,
            varint,
            intyp,
            extrapp,
            spval,
        )
        xrda = xr.DataArray(
            pvar,
            coords=[
                ("plev", plevs),
                ("lat", cesmxr.coords["lat"].data),
                ("lon", cesmxr.coords["lon"].data),
            ],
            dims=["plev", "lat", "lon"],
            attrs=cesmxr[var].attrs,
            name=var,
        )
        timeattrs = None
    else:
        # This step needs to be reworked or it may not fit in memory for large files
        # Maybe using xarray and ufunc will do the job
        pvar = vinth2p_ecmwf_wrap(
                    varxd,
                    hbcoefa,
                    hbcoefb,
                    p0,
                    varps,
                    plevo,
                    phis,
                    tbot,
                    varint,
                    intyp,
                    extrapp,
                    spval,
                )
        # Reassemble the numpy array into an xarray DataArray
        xrda = xr.DataArray(
            pvar,
            coords=[
                ("time", cesmxr["time"].data),
                ("plev", plevs),
                ("lat", cesmxr.coords["lat"].data),
                ("lon", cesmxr.coords["lon"].data),
            ],
            dims=["time", "plev", "lat", "lon"],
            attrs=cesmxr[var].attrs,
            name=var,
        )
        timeattrs = cesmxr.time.attrs

    # convert DataArray to DataSet before return
    xrds = xrda.to_dataset(name=xrda.name)
    lattrs = {
        "long_name": "pressure",
        "units": "hPa",
        "positive": "down",
        "standard_name": "atmosphere__pressure_coordinate",
    }
    xrds.attrs = cesmxr[var].attrs
    if timeattrs is not None:
        xrds.time.attrs = timeattrs
    xrds.plev.attrs = lattrs
    xrds.lat.attrs = cesmxr[var].lat.attrs
    xrds.lon.attrs = cesmxr[var].lon.attrs
    return xrds


if __name__ == "__main__":
    in_file = args.infile
    variable_name = args.var

    # Constructing the out_file name based on in_file and variable_name
    out_file = in_file.replace(".nc", f"_{variable_name}_p.nc")

    nchyb = xr.open_dataset(in_file, engine="netcdf4")

    plvl = np.array([
        30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 
        500.0, 600.0, 700.0, 775.0, 850.0, 925.0, 1000.0,
    ])
    
    z3_on_p = h2pvar(nchyb, variable_name, plvl)

    # there is a problem with time being written as int64
    try:
        z3_on_p.time.encoding["dtype"] = "float64"
    except:
        pass

    write_job = z3_on_p.to_netcdf(
        out_file, unlimited_dims="time", engine="netcdf4", compute=False
    )
    with ProgressBar():
        print(f"Writing to {out_file}")
        write_job.compute()

