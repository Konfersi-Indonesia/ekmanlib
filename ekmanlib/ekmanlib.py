import numpy as np

def ncdump(nc_fid, verb=True):
    def print_ncattr(key):
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim)
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    nc_vars = [var for var in nc_fid.variables]
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def timeIndexToDatetime(baseTime,times):
    newTimes=[]
    for ts in times:
        newTimes.append(baseTime+datetime.timedelta(seconds=ts))

    return newTimes

def wind_stress(u, v, rho_air=1.22, cd=None):
    """Convert wind speed (u,v) to wind stress (Tx,Ty).

    It uses either wind-dependent or constant drag.

    Args:
        u, v: Wind vector components (m/s), 2d or 3d (for time series).
        rho_air: Density of air (1.22 kg/m^3).
        cd: Non-dimensional drag (wind-speed dependent).
            For constant drag use cd=1.5e-3.
    Notes:
        Function to compute wind stress from wind field data is based on Gill,
        (1982)[1]. Formula and a non-linear drag coefficient (cd) based on
        Large and Pond (1981)[2], modified for low wind speeds (Trenberth et
        al., 1990)[3]

        [1] A.E. Gill, 1982, Atmosphere-Ocean Dynamics, Academy Press.
        [2] W.G. Large & S. Pond., 1981,Open Ocean Measurements in Moderate
        to Strong Winds, J. Physical Oceanography, v11, p324-336.
        [3] K.E. Trenberth, W.G. Large & J.G. Olson, 1990, The Mean Annual
        Cycle in Global Ocean Wind Stress, J. Physical Oceanography, v20,
        p1742-1760.
    """
    w = np.sqrt(u**2 + v**2) # wind speed (m/s) 

    if not cd:
        # wind-dependent drag
        cond1 = (w<=1)
        cond2 = (w>1) & (w<=3)
        cond3 = (w>3) & (w<10)
        cond4 = (w>=10)
        cd = np.zeros_like(w)
        cd[cond1] = 2.18e-3 
        cd[cond2] = (0.62 + 1.56/w[cond2]) * 1e-3
        cd[cond3] = 1.14e-3
        cd[cond4] = (0.49 + 0.065*w[cond4]) * 1e-3

    Tx = rho_air * cd * w * u # zonal wind stress (N/m^2)
    Ty = rho_air * cd * w * v # meridional wind stress (N/m^2)
    return [Tx, Ty]


def wind_curl(u, v, x, y, ydim=0, xdim=1, tdim=2):
    """Calculate the curl of the wind vector. 
    
    Args:
        u, v: Wind vector components (m/s), 2d or 3d (for time series).
        x, y: Coordinates in lon/lat (degrees), 2d.

    Notes:
        Curl(u,v) = dv/dx - du/dy
        Units of frequency (1/s).
        The different constants come from oblateness of the ellipsoid.
    """
    dy = np.abs(y[1,0] - y[0,0]) # scalar in deg
    dx = np.abs(x[0,1] - x[0,0]) 
    dy *= 110575. # scalar in m
    dx *= 111303. * np.cos(y * np.pi/180) # array in m (varies w/lat)  # FIXME?
    # extend dimension for broadcasting (2d -> 3d)
    dvdx = []
    if u.ndim == 3:
        dx = np.expand_dims(dx, tdim)
        
    # grad[f(y,x), delta] = diff[f(y)]/delta, diff[f(x)]/delta 
    dudy = np.gradient(u, dy)[ydim] # (1/s)
    for i in range(0,len(v[:,0]),1):
        dvdx1 = np.gradient(v[i,:], dx[i,0])
        dvdx = np.concatenate((dvdx,dvdx1),axis=0)
    dvdx = np.reshape(dvdx,(len(v[:,0]),len(v[0,:])))
    curl = dvdx - dudy # (1/s)
    return curl


def wind_stress_curl(Tx, Ty, x, y, ydim=0, xdim=1, tdim=2):
    """Calculate the curl of wind stress (Tx, Ty).

    Args:
        Tx, Ty: Wind stress components (N/m^2), 2d or 3d (for time series)
        x, y: Coordinates in lon/lat (degrees), 2d.

    Notes:
        Curl(Tx,Ty) = dTy/dx - dTx/dy
        The different constants come from oblateness of the ellipsoid.
    """
    dy = np.abs(y[1,0] - y[0,0]) # scalar in deg
    dx = np.abs(x[0,1] - x[0,0]) 
    dy *= 110575. # scalar in m
    dx *= 111303. * np.cos(y * np.pi/180) # array in m (varies w/lat)
    # extend dimension for broadcasting (2d -> 3d)
    if Tx.ndim == 3:
        dx = np.expand_dims(dx, tdim)
    
    dTydx = []
    # grad[f(y,x), delta] = diff[f(y)]/delta, diff[f(x)]/delta 
    dTxdy = np.gradient(Tx, dy)[ydim] # (N/m^3)
    #dTydx = np.gradient(Ty, dx)[xdim] 
    
    for i in range(0,len(Ty[:,0]),1):
        dTydx1 = np.gradient(Ty[i,:], dx[i,0])
        dTydx = np.concatenate((dTydx,dTydx1),axis=0)
    dTydx = np.reshape(dTydx,(len(Ty[:,0]),len(Ty[0,:])))
    
    curl_tau = dTydx - dTxdy # (N/m^3)
    return curl_tau

def ekman_transport(tau_x, tau_y, y, rho_water=1028., tdim=2):
    """Calculate Ekman Mass Transport from Wind Curl.
    
    Args:
        tau_x = Wind Curl X, 2d or 3d (for time series)
        tau y = Wind Curl Y, 2d or 3d (for time series)
        y = Latitude grid, 2d
    """
    
    omega = 7.292115e-5 # rotation rate of the Earth (rad/s)
    f = 2 * omega * np.sin(y * np.pi/180) # (rad/s)

    if f.shape != curl_tau.shape:
        f = np.expand_dims(f, tdim)

    emtx = (tau_y)/(rho_water*f)
    emty = (tau_x)/(rho_water*f)

    return emtx,emty
def ekman_pumping(curl_tau, y, rho_water=1028., tdim=2):
    """Calculate Ekman pumping from wind-stress curl.

    Args:
        curl_tau: Wind stress curl (N/m^3), 2d or 3d (for time series).
        y: Latitude grid (degrees), 2d.

    Notes:
        We = Curl(tau)/rho*f (vertical velocity in m/s).
        f = Coriolis frequency (rad/s), latitude dependent.
        rho = Ocean water density (1028 kg/m^3).
    """
    # Coriolis frequency
    omega = 7.292115e-5 # rotation rate of the Earth (rad/s)
    f = 2 * omega * np.sin(y * np.pi/180) # (rad/s)

    # Expand dimension for broadcasting (2d -> 3d)
    if f.shape != curl_tau.shape:
        f = np.expand_dims(f, tdim)

    # Ekman pumping
    We = curl_tau / (rho_water * f) # vertical velocity (m/s)
    return We
def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray
