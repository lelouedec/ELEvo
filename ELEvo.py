

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



# today = datetime.today()
# date_today = datetime.now().strftime('%Y-%m-%d')
# arr_outputdirectory=arrival_path+str(date_today)



# date_today_hours = datetime.now().strftime('%Y-%m-%d_%H')
# date_today_minutes = datetime.now().strftime('%Y-%m-%d %H:%M')

# header = r'ID, time 21.5 [UT, at 21.5 R_Sun], lon [deg], lat [deg], half width [deg], initial speed [km/s], arrival time [UTC], error arrival time [h], arrival speed [km/s], error arrival speed [km/s]'
# with open(arr_outputdirectory+'/icme_arrival_'+date_today_hours+'.txt', "a") as f:
#     f.write('ASWO, GeoSphere Austria - created ' + str(today.strftime('%A'))[0:3] + ' ' + date_today_minutes + ' UTC \n')
#     f.write(header + '\n')
#     f.close
    
# header = r'ID, time 21.5 [UT, at 21.5 R_Sun], lon [deg], lat [deg], half width [deg], initial speed [km/s], arrival time @SolarOrbiter [UTC], error arrival time [h], arrival speed @SolarOrbiter [km/s], error arrival speed [km/s]'
# with open(arr_outputdirectory+'/icme_arrival_solo_'+date_today_hours+'.txt', "a") as f:
#     f.write('ASWO, GeoSphere Austria - created ' + str(today.strftime('%A'))[0:3] + ' ' + date_today_minutes + ' UTC \n')
#     f.write(header + '\n')
#     f.close
    
# header = r'ID, time 21.5 [UT, at 21.5 R_Sun], lon [deg], lat [deg], initial speed [km/s], arrival time [UT], error arrival time @STEREO-A [h], arrival speed @STEREO-A [km/s], error arrival speed [km/s]'
# with open(arr_outputdirectory+'/icme_arrival_sta_'+date_today_hours+'.txt', "a") as f:
#     f.write('ASWO, GeoSphere Austria - created ' + str(today.strftime('%A'))[0:3] + ' ' + date_today_minutes + ' UTC \n')
#     f.write(header + '\n')
#     f.close



def fun_wrapper(dict_args):
    return Utils.Prediction_ELEvo(**dict_args)



def main(path_to_donki, path_to_positions):
    liste_spc = ['l1','solo','psp','sta','bepi','mercury','venus','mars']
    dates = [datetime(2024,5,1),datetime(2024,5,5)]
    data = data_utils.load_donki(path_to_donki,dates)
    positions = data_utils.load_position(path_to_positions,[mdates.date2num(dates[0]),mdates.date2num(dates[1])],liste_spc)

    # for d in data:
    #    d["positions"]=positions
   

    print('Generating kinematics using ELEvo')

    start_time = time.time()

    # if len(data) >= 5:
    #     used=5
    # else:
    #     used=1
        
    # pool = multiprocessing.Pool(used)
    # results = pool.map(fun_wrapper, data)
    # pool.close()
    # pool.join()
    
    results = []
    for d in data:
        d["positions"]=positions
        results.append(fun_wrapper(d))
    
    

    cmes = {}
    cmes["hc_time_num1"]= np.concatenate(np.array(results, dtype=object)[:,0])
    cmes["hc_r1" ]= np.concatenate(np.array(results, dtype=object)[:,1],1)
    cmes["hc_lat1" ]= np.concatenate(np.array(results, dtype=object)[:,2])
    cmes["hc_lon1" ]= np.concatenate(np.array(results, dtype=object)[:,3])
    cmes["a1_ell" ]= np.concatenate(np.array(results, dtype=object)[:,4],1)
    cmes["b1_ell" ]= np.concatenate(np.array(results, dtype=object)[:,5],1)
    cmes["c1_ell" ]= np.concatenate(np.array(results, dtype=object)[:,6],1)
    cmes["hc_id1" ]= np.concatenate(np.array(results, dtype=object)[:,7])
    cmes["hc_v1" ]= np.concatenate(np.array(results, dtype=object)[:,8],1)


    plotting_utils.make_frame_trajectories(positions,start_end=False,cmes=cmes)


    print('Done in: ',np.round((time.time()-start_time)), 'seconds')


def plot_trajectories_only(path_to_positions):
    
    positions = data_utils.load_position(path_to_positions,[mdates.date2num(datetime(2025,11,1)),mdates.date2num(datetime(2025,11,1)+timedelta(days=365))],['l1','solo','psp','sta','bepi','mercury','venus','mars'])
    plotting_utils.make_frame_trajectories(positions,punch=True,trajectories=True)

if __name__ == '__main__':
    
    path_to_donki = 'data/'
    path_to_positions = 'data/'
    main(path_to_donki, path_to_positions)
    # plot_trajectories_only()


