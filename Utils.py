import numpy as np 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import astropy.units as u
import data_utils
from sunpy.time import parse_time
from numba import njit, prange

def cart2sphere(x,y,z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Parameters
    ----------
    x : float or array-like
        The x-coordinate(s) in Cartesian space.
    y : float or array-like
        The y-coordinate(s) in Cartesian space.
    z : float or array-like
        The z-coordinate(s) in Cartesian space.

    Returns
    -------
    r : float or array-like
        The radial distance from the origin.
    theta : float or array-like
        The polar angle (inclination) in radians, measured from the z-axis.
    phi : float or array-like
        The azimuthal angle in radians, measured from the x-axis in the x-y plane.

    Notes
    -----
    - The output angles are in radians.
    """
    r = np.sqrt(x**2+ y**2 + z**2)           
    theta = np.arctan2(z,np.sqrt(x**2+ y**2))
    phi = np.arctan2(y,x)                    
    return r, theta, phi

@njit(fastmath=True, parallel=True)
def compute_cme_ensemble(gamma, ambient_wind, speed_ensemble, timesteps, distance0):
    """
    Compute CME ensemble propagation (r and v) with Numba acceleration.

    Parameters
    ----------
    gamma : 1D array (n_ensemble)
    ambient_wind : 1D array (n_ensemble)
    speed_ensemble : 1D array (n_ensemble)
    timesteps : 1D array (kindays_in_min)
    distance0 : float

    Returns
    -------
    cme_r_ensemble : 2D array [kindays_in_min, n_ensemble]
    cme_v_ensemble : 2D array [kindays_in_min, n_ensemble]
    """

    n_steps = timesteps.size
    n_ens = gamma.size
    cme_r_ensemble = np.empty((n_steps, n_ens), dtype=np.float32)
    cme_v_ensemble = np.empty((n_steps, n_ens), dtype=np.float32)

    for j in prange(n_ens):
        g = gamma[j]
        v_amb = ambient_wind[j]
        v0 = speed_ensemble[j]
        accsign = 1.0
        if v0 < v_amb:
            accsign = -1.0

        gfac = g * 1e-7
        for i in range(n_steps):
            t = timesteps[i]
            term = accsign * gfac * (v0 - v_amb) * t
            cme_r_ensemble[i, j] = (
                (accsign / gfac) * np.log(1.0 + term)
                + v_amb * t
                + distance0
            )
            cme_v_ensemble[i, j] = (
                (v0 - v_amb) / (1.0 + term) + v_amb
            )

    return cme_r_ensemble, cme_v_ensemble

def process_arrival(distance, obj, time1, cme_v, cme_id, t0, halfAngle, speed, cme_lon, cme_lat, label):
        """
        Processes the arrival time and related parameters for a CME.
        Args:
            distance (np.ndarray): Array of distances for each time and scenario.
            obj (float): Target distance for arrival calculation.
            time1 (list or np.ndarray): List of datetime objects corresponding to each distance entry.
            cme_v (np.ndarray): Array containing CME speed and its uncertainties.
            cme_id (np.ndarray): Array containing CME identifier(s).
            t0 (datetime): CME launch time.
            halfAngle (float): Half angular of the CME.
            speed (float): Mean speed of the CME.
            cme_lon (np.ndarray): Array containing CME longitude(s).
            cme_lat (np.ndarray): Array containing CME latitude(s).
            label (str): Target label (e.g., 'earth') to determine calculation method.
        Returns:
            dict: Dictionary containing the following keys:
                - "arrival": List with CME arrival information and uncertainties.
                - "arr_time_fin": List of final arrival times.
                - "arr_time_err0": List of lower bound arrival times.
                - "arr_time_err1": List of upper bound arrival times.
                - "arr_id": List of CME identifiers.
                - "arr_hit": List indicating if arrival was detected (1.0) or not (nan).
                - "arr_speed_list": List of arrival speeds.
                - "arr_speed_err_list": List of arrival speed uncertainties.
        """
        arr_time = []
        arrival = []
        arr_time_fin = []
        arr_time_err0 = []
        arr_time_err1 = []
        arr_id = []
        arr_hit = []
        arr_speed_list = []
        arr_speed_err_list = []

        if not np.isnan(distance).all():
            if label == 'earth':
                for t in range(3):
                    index = np.argmin(np.abs(np.ma.array(distance[:, t], mask=np.isnan(distance[:, t])) - obj))
                    arr_time.append(time1[int(index)])

            else:
                for t in range(3):
                    index = np.argmin(np.abs(distance[:,t] - obj))
                    arr_time.append(time1[int(index)])

            arr_speed = cme_v[:, 0][index]
            err_arr_speed = cme_v[:, 2][index] - cme_v[:, 1][index]
            err_arr_time = (arr_time[1] - arr_time[2]).total_seconds() / 3600.0
            arrival.append([
                cme_id[0].decode("utf-8"),
                t0.strftime('%Y-%m-%dT%H:%MZ'),
                "{:.1f}".format(cme_lon[0]),
                "{:.1f}".format(cme_lat[0]),
                "{:.1f}".format(halfAngle),
                "{:.1f}".format(speed),
                arr_time[0].strftime('%Y-%m-%dT%H:%MZ'),
                "{:.2f}".format(err_arr_time / 2),
                "{:.2f}".format(arr_speed),
                "{:.2f}".format(err_arr_speed / 2)
            ])
            arr_time_fin.append(arr_time[0])
            arr_time_err0.append(arr_time[0] - timedelta(hours=err_arr_time))
            arr_time_err1.append(arr_time[0] + timedelta(hours=err_arr_time))
            arr_id.append(cme_id[0].decode("utf-8"))
            arr_hit.append(1.0)
            arr_speed_list.append(arr_speed)
            arr_speed_err_list.append(err_arr_speed / 2)
        
        else:
            arr_time_fin.append(np.nan)
            arr_time_err0.append(np.nan)
            arr_time_err1.append(np.nan)
            arr_id.append(np.nan)
            arr_hit.append(np.nan)
            arr_speed_list.append(np.nan)
            arr_speed_err_list.append(np.nan)


        return {
            f"arrival": arrival,
            f"arr_time_fin": arr_time_fin,
            f"arr_time_err0": arr_time_err0,
            f"arr_time_err1": arr_time_err1,
            f"arr_id": arr_id,
            f"arr_hit": arr_hit,
            f"arr_speed_list": arr_speed_list,
            f"arr_speed_err_list": arr_speed_err_list,
        }

def Prediction_ELEvo(time21_5, latitude, longitude, halfAngle, speed, type, isMostAccurate, associatedCMEID, associatedCMEstartTime, note, associatedCMELink, catalog, featureCode, dataLevel, measurementTechnique, imageType, tilt, minorHalfWidth, speedMeasuredAtHeight, submissionTime, versionId, link,positions):
    print(associatedCMEID)
    distance0 = 21.5*u.solRad.to(u.km)
    t0 = time21_5
    gamma_init = 0.1
    ambient_wind_init = 400.
    kindays = 15
    n_ensemble = 50000
    halfwidth = np.deg2rad(halfAngle)
    res_in_min = 10
    f = 0.7
    kindays_in_min = int(kindays*24*60/res_in_min)

    earth = positions["l1"]
    #sta = positions["sta"]

    

    ### Just doing earth for now, if needed create generic function and call it with spacecraft pos array

    dct = mdates.date2num(time21_5) - earth.time
    earth_ind = np.argmin(np.abs(dct))
    

    if np.abs(np.deg2rad(longitude)) + np.abs(earth.lon[earth_ind][0]) > np.pi and np.sign(np.deg2rad(longitude)) != np.sign(earth.lon[earth_ind][0]):
        delta_earth = np.deg2rad(longitude) - (earth.lon[earth_ind][0] + 2 * np.pi * np.sign(np.deg2rad(longitude)))
    else:
        delta_earth = np.deg2rad(longitude) - earth.lon[earth_ind][0]


     #times for each event kinematic
    time1=[]
    tstart1=time21_5
    tend1=tstart1+timedelta(days=kindays)
    #make 30 min datetimes
    while tstart1 < tend1:

        time1.append(tstart1)  
        tstart1 += timedelta(minutes=res_in_min)    

   
    # #make kinematics
    timestep=np.zeros([kindays_in_min,n_ensemble])
    cme_r=np.zeros([kindays_in_min, 3])
    cme_v=np.zeros([kindays_in_min, 3])
    cme_lon=np.ones(kindays_in_min)*longitude
    cme_lat=np.ones(kindays_in_min)*latitude
    cme_id=np.chararray(kindays_in_min, itemsize=27)
    cme_id[:]=associatedCMEID
    cme_r_ensemble=np.zeros([kindays_in_min, n_ensemble])
    cme_v_ensemble=np.zeros([kindays_in_min, n_ensemble])
    

    cme_delta=delta_earth*np.ones([kindays_in_min,3])

    cme_hit=np.zeros(kindays_in_min)
    cme_hit[np.abs(delta_earth)<halfwidth] = 1


    distance_earth = np.empty([kindays_in_min,3])
    distance_solo = np.empty([kindays_in_min,3])
    distance_sta = np.empty([kindays_in_min,3])
    distance_earth[:] = np.nan
    distance_solo[:] = np.nan
    distance_sta[:] = np.nan


        
    kindays_in_min = int(kindays*24*60/res_in_min)
    
    gamma = np.abs(np.random.normal(gamma_init,0.025,n_ensemble))
    ambient_wind = np.random.normal(ambient_wind_init,50,n_ensemble)
    speed_ensemble = np.random.normal(speed,50,n_ensemble)
    
    timesteps = np.arange(kindays_in_min)*res_in_min*60
    timesteps = np.vstack([timesteps]*n_ensemble)
    timesteps = np.transpose(timesteps)

    accsign = np.ones(n_ensemble)
    accsign[speed_ensemble < ambient_wind] = -1.

    distance0_list = np.ones(n_ensemble)*distance0

    cme_r_ensemble, cme_v_ensemble = compute_cme_ensemble(
        gamma.astype(np.float32),
        ambient_wind.astype(np.float32),
        speed_ensemble.astype(np.float32),
        (np.arange(kindays_in_min, dtype=np.float32) * res_in_min * 60.0),
        np.float32(distance0)
    )

    cme_r_mean = cme_r_ensemble.mean(1)
    cme_r_std = cme_r_ensemble.std(1)
    cme_v_mean = cme_v_ensemble.mean(1)
    cme_v_std = cme_v_ensemble.std(1)
    cme_r[:,0]= cme_r_mean*u.km.to(u.au)
    cme_r[:,1]=(cme_r_mean - 2*cme_r_std)*u.km.to(u.au) 
    cme_r[:,2]=(cme_r_mean + 2*cme_r_std)*u.km.to(u.au)
    cme_v[:,0]= cme_v_mean
    cme_v[:,1]=(cme_v_mean - 2*cme_v_std)
    cme_v[:,2]=(cme_v_mean + 2*cme_v_std)
    
    #Ellipse parameters   
    theta = np.arctan(f**2*np.ones([kindays_in_min,3]) * np.tan(halfwidth*np.ones([kindays_in_min,3])))
    omega = np.sqrt(np.cos(theta)**2 * (f**2*np.ones([kindays_in_min,3]) - 1) + 1)   
    cme_b = cme_r * omega * np.sin(halfwidth*np.ones([kindays_in_min,3])) / (np.cos(halfwidth*np.ones([kindays_in_min,3]) - theta) + omega * np.sin(halfwidth*np.ones([kindays_in_min,3])))    
    cme_a = cme_b / f*np.ones([kindays_in_min,3])
    cme_c = cme_r - cme_b
        
    root = np.sin(cme_delta)**2 * f**2*np.ones([kindays_in_min,3]) * (cme_b**2 - cme_c**2) + np.cos(cme_delta)**2 * cme_b**2
    distance_earth[cme_hit.all() == 1] = (cme_c * np.cos(cme_delta) + np.sqrt(root)) / (np.sin(cme_delta)**2 * f**2*np.ones([kindays_in_min,3]) + np.cos(cme_delta)**2) #distance from SUN in AU for given point on ellipse
    

    #### linear interpolate to 10 min resolution

    #find next full hour after t0
    format_str = '%Y-%m-%d %H'  
    t0r = datetime.strptime(datetime.strftime(t0, format_str), format_str) +timedelta(hours=1)
    time2=[]
    tstart2=t0r
    tend2=tstart2+timedelta(days=kindays)
    #make 30 min datetimes 
    while tstart2 < tend2:
        time2.append(tstart2)  
        tstart2 += timedelta(minutes=res_in_min)  

    time2_num=parse_time(time2).plot_date        
    time1_num=parse_time(time1).plot_date
    
    results_earth = process_arrival(distance_earth, earth.r[earth_ind][0], time1, cme_v, cme_id, t0, halfAngle, speed, cme_lon, cme_lat, label="earth")
  
    #linear interpolation to time_mat times    
    cme_r = [np.interp(time2_num, time1_num,cme_r[:,i]) for i in range(3)]
    cme_v = [np.interp(time2_num, time1_num,cme_v[:,i]) for i in range(3)]
    cme_lat = np.interp(time2_num, time1_num,cme_lat )
    cme_lon = np.interp(time2_num, time1_num,cme_lon )
    cme_a = [np.interp(time2_num, time1_num,cme_a[:,i]) for i in range(3)]
    cme_b = [np.interp(time2_num, time1_num,cme_b[:,i]) for i in range(3)]
    cme_c = [np.interp(time2_num, time1_num,cme_c[:,i]) for i in range(3)]
    

    return time2_num, cme_r, cme_lat, cme_lon, cme_a, cme_b, cme_c, cme_id, cme_v, results_earth['arr_time_fin'], results_earth['arr_time_err0'], results_earth['arr_time_err1'], results_earth['arr_id'], results_earth['arr_hit'], results_earth['arr_speed_list'], results_earth['arr_speed_err_list']

if __name__ == '__main__':
    print("main function here")