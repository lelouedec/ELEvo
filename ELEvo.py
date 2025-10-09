

ffmpeg_path=''

import datetime
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import numpy as np
import time
import multiprocessing

import warnings
warnings.filterwarnings('ignore')

import data_utils
import Utils 
import plotting_utils

def fun_wrapper(dict_args):
    return Utils.Prediction_ELEvo(**dict_args)



def main(path_to_donki, path_to_positions):
    object_list = ['l1','solo','psp','sta','bepi','mercury','venus','mars']

    dates = [datetime(2024,5,1),datetime(2025,4,30)]
    data = data_utils.load_donki(path_to_donki,dates)
    positions = data_utils.load_position(path_to_positions,[mdates.date2num(dates[0]),mdates.date2num(dates[1])],object_list)


    print('Generating kinematics using ELEvo')

    start_time = time.time()
    
    results = []
    for d in data:
        d["positions"]=positions
        results.append(fun_wrapper(d))
    

    cmes = {}

    cmes["hc_time_num1"]= np.vstack(np.array(results, dtype=object)[:,0])

    cmes["hc_r1" ]= np.array(results, dtype=object)[:,1]

    cmes["hc_lat1" ]= np.vstack(np.array(results, dtype=object)[:,2])
    cmes["hc_lon1" ]= np.vstack(np.array(results, dtype=object)[:,3])

    cmes["a1_ell" ]= np.array(results, dtype=object)[:,4]
    cmes["b1_ell" ]= np.array(results, dtype=object)[:,5]
    cmes["c1_ell" ]= np.array(results, dtype=object)[:,6]

    cmes["hc_id1" ]= np.vstack(np.array(results, dtype=object)[:,7])

    cmes["hc_v1" ]= np.array(results, dtype=object)[:,8]

    cmes["hc_arr_time1"] = np.vstack(np.array(results, dtype=object)[:,9])
    cmes["hc_err_arr_time_min1"] = np.vstack(np.array(results, dtype=object)[:,10])
    cmes["hc_err_arr_time_max1"] = np.vstack(np.array(results, dtype=object)[:,11])
    cmes["hc_arr_id1"] = np.vstack(np.array(results, dtype=object)[:,12])
    cmes["hc_arr_hit1"] = np.vstack(np.array(results, dtype=object)[:,13])
    cmes["hc_arr_speed1"] = np.vstack(np.array(results, dtype=object)[:,14])
    cmes["hc_err_arr_speed1"] = np.vstack(np.array(results, dtype=object)[:,15])

    np.save('data/cmes_elevo.npy',cmes)
    #plotting_utils.make_frame_trajectories(positions,object_list,start_end=False,cmes=cmes)

    print('Done in: ',np.round((time.time()-start_time)), 'seconds')


def plot_trajectories_only(path_to_positions):
    
    positions = data_utils.load_position(path_to_positions,[mdates.date2num(datetime(2025,11,1)),mdates.date2num(datetime(2025,11,1)+timedelta(days=365))],['l1','solo','psp','sta','bepi','mercury','venus','mars'])
    plotting_utils.make_frame_trajectories(positions,punch=True,trajectories=True)

if __name__ == '__main__':
    
    path_to_donki = 'data/'
    path_to_positions = 'data/'
    main(path_to_donki, path_to_positions)
    # plot_trajectories_only()


