

ffmpeg_path=''

import datetime
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import numpy as np
import time
import multiprocessing
import os
import warnings
warnings.filterwarnings('ignore')

import utils.data_utils as data_utils
import utils.pred_utils as pred_utils 
import utils.plotting_utils as plotting_utils
import copy
import sys

def fun_wrapper(dict_args):
    return pred_utils.Prediction_ELEvo(**dict_args)

def save_test_output(dat, res):

    nam = dat['associatedCMEID'].replace(':','_').replace('-','_')
    save_test_dict = copy.deepcopy(dat)

    (time2_num, cme_r, cme_lat,
     cme_lon, cme_a, cme_b,
     cme_c, cme_id, cme_v,
     halfwidth, arr_time_fin, arr_time_err0,
     arr_time_err1, arr_id, arr_hit,
     arr_speed_list, arr_speed_err_list) = res

    save_results_dict = {}
    save_results_dict["hc_time_num1"]= time2_num
    save_results_dict["hc_r1" ]= cme_r
    save_results_dict["hc_lat1" ]= cme_lat
    save_results_dict["hc_lon1" ]= cme_lon
    save_results_dict["a1_ell" ]= cme_a
    save_results_dict["b1_ell" ]= cme_b
    save_results_dict["c1_ell" ]= cme_c
    save_results_dict["hc_id1" ]= cme_id
    save_results_dict["hc_v1" ]= cme_v
    save_results_dict["halfwidth" ]= halfwidth
    save_results_dict["hc_arr_time1"] = arr_time_fin
    save_results_dict["hc_err_arr_time_min1"] = arr_time_err0
    save_results_dict["hc_err_arr_time_max1"] = arr_time_err1
    save_results_dict["hc_arr_id1"] = arr_id
    save_results_dict["hc_arr_hit1"] = arr_hit
    save_results_dict["hc_arr_speed1"] = arr_speed_list
    save_results_dict["hc_err_arr_speed1"] = arr_speed_err_list

    np.save('data/test_elevo_input_'+nam+'.npy',save_test_dict)

    np.save('data/test_elevo_output_'+nam+'.npy',save_results_dict)


def main(path_to_donki, path_to_positions, dates, strudl_path=None):
    # TODO: Make strudl_path keyword work properly - is supposed to allow plotting STRUDL tracks along with ELEvo results

    object_list = list(path_to_positions.keys())
    data = data_utils.load_donki(path_to_donki,dates)

    positions = {}

    for sc_obj in object_list:
        _, sc_name = data_utils.parse_space_obj_names(sc_obj)

        if not os.path.isfile(path_to_positions[sc_obj]):
            print(f'Position file for {sc_name} not found at {path_to_positions[sc_obj]}')
            print(f'Attempting to download positions for {sc_name} from JPL Horizons...')
            data_utils.create_positions_file(sc_obj, start='2010-01-01', stop='2030-12-31', step='10m', save_path=path_to_positions[sc_obj])
            path_to_positions[sc_obj] = os.path.join(path_to_positions[sc_obj], f'positions_{sc_name}_from_2010_to_2030_HEEQ_10m_rad_mb.p')

        positions[sc_name] = data_utils.load_positions_jpl(path_to_positions[sc_obj],[mdates.date2num(dates[0]),mdates.date2num(dates[1])],sc_name)
    
    if strudl_path is not None:
        # TODO: make this work with multiple CMEs
        strudl_tracks = np.load(strudl_path, allow_pickle=True).item()
        interesting_cme = 'CME_60'
        strudl_track = strudl_tracks[interesting_cme]


    print('Generating kinematics using ELEvo')

    start_time = time.time()
    
    results = []
    for d in data:
        d["positions"]=positions
        # d["seed_value"]=42  # for reproducibility in tests
        # dat = copy.deepcopy(d) # for saving input/output

        results.append(fun_wrapper(d))

        # save_test_output(dat, results[-1])  # Uncomment to save test input/output for each CME
        # sys.exit()  # Uncomment to stop after first CME for testing purposes

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

    cmes["halfwidth" ]= np.vstack(np.array(results, dtype=object)[:,9])

    cmes["hc_arr_time1"] = np.array(results, dtype=object)[:,10]
    cmes["hc_err_arr_time_min1"] = np.array(results, dtype=object)[:,11]
    cmes["hc_err_arr_time_max1"] = np.array(results, dtype=object)[:,12]
    cmes["hc_arr_id1"] = np.array(results, dtype=object)[:,13]
    cmes["hc_arr_hit1"] = np.array(results, dtype=object)[:,14]
    cmes["hc_arr_speed1"] = np.array(results, dtype=object)[:,15]
    cmes["hc_err_arr_speed1"] = np.array(results, dtype=object)[:,16]

    np.save('data/cmes_elevo_'+datetime.strftime(dates[0], '%Y%m%d')+'_'+datetime.strftime(dates[1], '%Y%m%d')+'.npy',cmes)
    #plotting_utils.make_frame_trajectories(positions,object_list,start_end=False,cmes=cmes,cme_tracks=(strudl_track if strudl_path is not None else None))

    print('Done in: ',np.round((time.time()-start_time)), 'seconds')


def plot_trajectories_only(path_to_positions):
    
    positions = data_utils.load_position(path_to_positions,[mdates.date2num(datetime(2025,11,1)),mdates.date2num(datetime(2025,11,1)+timedelta(days=365))],['l1','solo','psp','sta','bepi','mercury','venus','mars'])

if __name__ == '__main__':

    dates = [datetime(2025,5,12),datetime(2025,5,13)]
    path_to_donki = 'data/'
    strudl_path = None#'data/tracks_with_parameters_mean_45_2024_05_01_2025_04_30_earth_pa_6h_cleaned.npy'

    # for downloading positions, specify folder to save the file in
    # for using existing files, specify full path to the file

    path_to_positions = {'l1': 'data/positions_l1_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'solo': 'data/positions_solo_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'psp': 'data/positions_psp_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'sta': 'data/positions_sta_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'bepi': 'data/positions_bepi_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'mercury': 'data/positions_mercury_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'venus': 'data/positions_venus_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'mars': 'data/positions_mars_from_2010_to_2030_HEEQ_10m_rad_mb.p'}

    # path_to_positions = {'l1': 'data/',
    #                      'solo': 'data/',
    #                      'psp': 'data/',
    #                      'sta': 'data/',
    #                      'bepi': 'data/',
    #                      'mercury': 'data/',
    #                      'venus': 'data/',
    #                      'mars': 'data/'}
    
    main(path_to_donki, path_to_positions, dates, strudl_path)
    # plot_trajectories_only()


