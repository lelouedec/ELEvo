import pickle
from datetime import datetime,timedelta
import urllib
import matplotlib.dates as mdates
import json 
import requests
import pandas as pd 
import numpy as np





def load_position(data_path,date_range,spc=["l1","solo"]):
    with open(data_path+'positions_from_2010_to_2030_HEEQ_10min_rad_ed.p', "rb") as f:
        pos = pickle.load(f)
    columns = pos['l1'].dtype.names
    positions = {}
    for s in spc:
        idx = np.argwhere(np.logical_and(pos[s][columns[0]]>=date_range[0],pos[s][columns[0]]<=date_range[1]))
        positions[s] = pos[s][idx]
     
    return positions


def load_donki(results_path,dates):

    url_donki='https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis?startDate='+dates[0].strftime('%Y-%m-%d')+'&endDate='+dates[1].strftime('%Y-%m-%d')+'&mostAccurateOnly=true'
    try:
        r = requests.get(url_donki)
        with open(results_path+'DONKI.json','wb') as f:
            f.write(r.content)
    except:
        print(url_donki)
        print('DONKI not loaded')   

    f = open(results_path+'DONKI.json')
    data = json.load(f)
    CMEs = {}   
    for d in data:
        if(d["associatedCMEID"] in CMEs.keys()):
            d["time21_5"] = datetime.strptime(d["time21_5"],"%Y-%m-%dT%H:%MZ")
            for k in d.keys():
                if(isinstance(d[k], datetime)):
                    diff = timedelta(seconds=(CMEs[d["associatedCMEID"]][k]-d[k] ).total_seconds())
                    
                    if(CMEs[d["associatedCMEID"]][k]>d[k]):
                        CMEs[d["associatedCMEID"]][k] = CMEs[d["associatedCMEID"]][k] + diff/2
                    else:
                        CMEs[d["associatedCMEID"]][k] = d[k] + diff/2
                   
                elif(isinstance(d[k], (int, float, complex))):
                    # TODO: this fix is temporary, need to check why there are None entries
                    try:
                        CMEs[d["associatedCMEID"]][k] = np.nanmean([CMEs[d["associatedCMEID"]][k],d[k]])
                    except:
                        CMEs[d["associatedCMEID"]][k] = d[k]
                        
        else:
            d["time21_5"] = datetime.strptime(d["time21_5"],"%Y-%m-%dT%H:%MZ")
            CMEs[d["associatedCMEID"]] = d


    return list(CMEs.values())

def load_strudl_tracks(path, return_parameters=False):
    
    times  = []
    elongs = []

    if return_parameters:
        params = []

    converted_strudl_dict = np.load(path, allow_pickle=True).item()

    for cme_key in converted_strudl_dict.keys():
        
        times.append([pd.to_datetime(d) for d in converted_strudl_dict[cme_key]["time"]])
        elongs.append(list(converted_strudl_dict[cme_key]["elongation"]))

        if return_parameters:
            phi = converted_strudl_dict[cme_key]["phi"]
            halfwidth = converted_strudl_dict[cme_key]["halfwidth"]
            L1_ist_obs = converted_strudl_dict[cme_key]["L1_ist_obs"]
            cmeID_elevo = converted_strudl_dict[cme_key]["cmeID_elevo"]

            params.append({'phi':phi, 'halfwidth':halfwidth, 'L1_ist_obs':L1_ist_obs, 'cmeID_strudl':cme_key, 'cmeID_elevo':cmeID_elevo})

    if return_parameters:
        return times,elongs,params
    else:
        return times,elongs
    
if __name__ == "__main__":
    print(len(load_donki("./")))