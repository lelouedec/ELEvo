
import numpy as np 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import Utils
import matplotlib.pyplot as plt 
from sunpy.time import parse_time
import matplotlib as mpl
from scipy.stats import norm
import seaborn as sns
from numba import njit, prange
import time
from matplotlib.patches import Polygon
from matplotlib.path import Path
# mpl.rcParams['path.simplify'] = True
# mpl.rcParams['path.simplify_threshold'] = 0.5
# mpl.rcParams['lines.antialiased'] = False
# mpl.rcParams['patch.antialiased'] = False
# mpl.rcParams['agg.path.chunksize'] = 10000  # helps for large polygons

def angle_to_coord_line(angle,x0,y0,x1,y1):
    #rotate by 4 deg for HI1 FOV
    ang=np.deg2rad(angle)
    rot=np.array([[np.cos(ang), -np.sin(ang)], 
                  [np.sin(ang), np.cos(ang)]
                  ])    
    [x2,y2]=np.dot(rot,[x1,y1])

    #add to sta position
    x2f=x0+x2
    y2f=y0+y2    
    
    return Utils.cart2sphere(x2f,y2f,0.0)    

def draw_punch_fov(pos, time_num, timeind,ax):
    
    lcolor='green'

    #sta position
    x0=pos['x'][timeind]
    y0=pos['y'][timeind]
    z0=0

    x1=-pos['x'][timeind]
    y1=-pos['y'][timeind]
    z1=0

    
    r2,t2,lon2=angle_to_coord_line(45,x0,y0,x1,y1)
    r3,t3,lon3=angle_to_coord_line(-45,x0,y0,x1,y1)


    r4,t4,lon4=angle_to_coord_line(1.5,x0,y0,x1,y1)
    r5,t5,lon5=angle_to_coord_line(-1.5,x0,y0,x1,y1)

    r6,t6,lon6=angle_to_coord_line(4.4,x0,y0,x1,y1)
    r7,t7,lon7=angle_to_coord_line(-4.4,x0,y0,x1,y1)

    r8,t8,lon8=angle_to_coord_line(7.4,x0,y0,x1,y1)
    r9,t9,lon9=angle_to_coord_line(-7.4,x0,y0,x1,y1)
    # r5,t5,lon5=angle_to_coord_line(ang4d,x0,y0,x1,y1)



    r0,t0,lon0 =Utils.cart2sphere(x0,y0,z0)   
    ax.plot([lon0,lon2],[r0,r2],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon3],[r0,r3],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon6],[r0,r6],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon7],[r0,r7],linestyle='-',color=lcolor,alpha=0.5, lw=1.2,label="PUNCH WFI Field of View")

    # ax.plot([lon0,lon4],[r0,r4],linestyle='-',color=lcolor,alpha=0.45, lw=1)
    # ax.plot([lon0,lon5],[r0,r5],linestyle='-',color=lcolor,alpha=0.45, lw=1)
    # ax.plot([lon0,lon8],[r0,r8],linestyle='-',color=lcolor,alpha=0.45, lw=1)
    ax.fill([lon0,lon9,lon5],[r0,r9,r5],color=lcolor,alpha=0.3)
    ax.fill([lon0,lon8,lon4],[r0,r8,r4],color=lcolor,alpha=0.3,label="PUNCH NFI Field of View")

def calculate_stereo_fov_lines(pos, sc):
    #plots the STA FOV HI1 HI2
    
    time_num = mdates.date2num(pos['time'])
    #STB never flipped the camera:
    sc = sc.lower()

    if sc=='stb': 
        ang1d=-4
        ang2d=-24
        ang3d=-18
        ang4d=-88
        # lcolor='blue'
    
    if sc=='sta': 
        ang1d=4
        ang2d=24
        ang3d=18
        ang4d=88
        # lcolor='red'

        #STA flipped during conjunction
        if mdates.date2num(datetime(2015,11,1))<time_num<mdates.date2num(datetime(2023,8,12)):  
            ang1d=-4
            ang2d=-24
            ang3d=-18
            ang4d=-88

    #calculate endpoints
    
    #sta position
    x0=pos['x']
    y0=pos['y']
    z0=0
    
    #sta position 180° rotated    
    x1=-pos['x']
    y1=-pos['y']
    z1=0
    
    r2,t2,lon2=angle_to_coord_line(ang1d,x0,y0,x1,y1)
    r3,t3,lon3=angle_to_coord_line(ang2d,x0,y0,x1,y1)   
    r4,t4,lon4=angle_to_coord_line(ang3d,x0,y0,x1,y1)

    r5,t5,lon5=angle_to_coord_line(ang4d,x0,y0,x1,y1)
    
    #convert to polar coordinates and plot
    [r0,t0,lon0]=Utils.cart2sphere(x0,y0,z0)    
    #[r1,t1,lon1]=hd.cart2sphere(x1,y1,z1)    


    rc11,tc21,lonc11=angle_to_coord_line(0.7,x0,y0,x1,y1)
    rc21,tc21,lonc21=angle_to_coord_line(4.0,x0,y0,x1,y1)

    rc12,tc22,lonc12=angle_to_coord_line(-0.7,x0,y0,x1,y1)
    rc22,tc22,lonc22=angle_to_coord_line(-4.0,x0,y0,x1,y1)

    fov_values = [lon0, r0, lon2, r2, lon3, r3, lonc11, rc11, lonc21, rc21, lonc12, rc12, lonc22, rc22]
    
    for num_val, value in enumerate(fov_values):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError("Expected scalar values for FOV calculations.")
            else:
                value = value[0]  # Convert single-element array to scalar
                fov_values[num_val] = value

    return np.array(fov_values)

def calculate_elongation_lines(pos, sc, track_elongation, track_time):
    #plots the STA FOV HI1 HI2
    
    #STB never flipped the camera:
    sc = sc.lower()
    track_elongation_calc = track_elongation[0]
    if sc=='sta': 

        #STA flipped during conjunction
        if mdates.date2num(datetime(2015,11,1))<track_time<mdates.date2num(datetime(2023,8,12)):  
            track_elongation_calc = -track_elongation_calc

    #calculate endpoints
    
    #sta position
    x0=pos['x']
    y0=pos['y']
    z0=0
    
    #sta position 180° rotated    
    x1=-pos['x']
    y1=-pos['y']
    z1=0
    
    r_elon = np.nan
    t_elon = np.nan
    lon_elon = np.nan

    
    #convert to polar coordinates and plot
    [r0,t0,lon0]=Utils.cart2sphere(x0,y0,z0)    
    #[r1,t1,lon1]=hd.cart2sphere(x1,y1,z1)    

    if not np.isnan(track_elongation_calc):
        r_temp,t_temp,lon_temp=angle_to_coord_line(track_elongation_calc,x0,y0,x1,y1)
        r_elon=r_temp
        t_elon=t_temp
        lon_elon=lon_temp
    
    else:
        r0=np.nan
        lon0=np.nan


    fov_values = [lon0, r0, lon_elon, r_elon]
    
    for num_val, value in enumerate(fov_values):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError("Expected scalar values for FOV calculations.")
            else:
                value = value[0]  # Convert single-element array to scalar
                fov_values[num_val] = value

    return np.array(fov_values)

def plot_elongation_line(ax, fov_data, sc, label_display=False):
    if sc.lower() == 'sta':
        lcolor = 'darkred'
    else:
        lcolor = 'darkblue'

    if(label_display):
        label="Elongation Line"
    else:
        label=""

    lon0, r0, lon_elon, r_elon = fov_data
    artists = []

    line, = ax.plot([lon0, lon_elon], [r0, r_elon], linestyle='--', color=lcolor, alpha=0.5, lw=1.2, label=label)
    artists.append(line)

    return artists
def plot_stereo_hi_fov(ax, fov_data, sc, label_display=False):    
    
    if sc.lower() == 'sta':
        lcolor = 'red'
    else:
        lcolor = 'blue'

    if(label_display):
        label1="STEREO-A/HI1 Field of View"
        label2="STEREO-A/COR2 Field of View"
    else:
        label1=""
        label2=""

    lon0, r0, lon2, r2, lon3, r3, lonc11, rc11, lonc21, rc21, lonc12, rc12, lonc22, rc22 = fov_data
    artists = []

    line1, = ax.plot([lon0, lon2], [r0, r2], linestyle='-', color=lcolor, alpha=0.5, lw=1.2)
    line2, = ax.plot([lon0, lon3], [r0, r3], linestyle='-', color=lcolor, alpha=0.5, lw=1.22, label=label1)
    artists.extend([line1, line2])

    poly1 = ax.fill([lon0, lonc11, lonc21], [r0, rc11, rc21], color=lcolor, alpha=0.3)
    poly2 = ax.fill([lon0, lonc12, lonc22], [r0, rc12, rc22], color=lcolor, alpha=0.3,label=label2)
    artists.extend([poly1[0], poly2[0]])  # fill returns a list of PolyCollections, take first

    return artists


def get_object_color(obj_name):

    obj_name = obj_name.lower()

    COLOR_MAP = {
    "sun": '#F9F200FF',
    "psp": '#052E37FF',
    "bepi": '#5833FEFF',
    "solo": '#F29707FF',
    "earth": '#75CC41FF',
    "l1": '#75CC41FF',
    "sta": '#E75C13FF',
    "mercury": '#9DABAEFF',
    "venus": '#8C11AAFF',
    "mars": '#E75C13B3',
    "cme": '#8C99FDFF'
    }

    if obj_name in COLOR_MAP:
        obj_color = COLOR_MAP[obj_name]
    else:
        raise ValueError(f"Unknown object: {obj_name}")

    return obj_color

def get_object_type(obj_name):

    obj_name = obj_name.lower()

    TYPE_MAP = {
    "sun": 'planet',
    "psp": 'spacecraft',
    "bepi": 'spacecraft',
    "solo": 'spacecraft',
    "earth": 'planet',
    "l1": 'planet',
    "sta": 'spacecraft',
    "mercury": 'planet',
    "venus": 'planet',
    "mars": 'planet'
    }

    if obj_name in TYPE_MAP:
        obj_type = TYPE_MAP[obj_name]
    else:
        raise ValueError(f"Unknown object: {obj_name}")

    return obj_type

def plot_spacecraft_planets(ax, pos_data, obj_list):

    symsize_planet=110
    symsize_spacecraft=80

    marker_spacecraft='s'
    marker_planet='o'

    zorder=3

    scatter_artists = []

    for pos_ind, pos in enumerate(pos_data):
        obj_name = obj_list[pos_ind]
        obj_type = get_object_type(obj_name)

        marker_color = get_object_color(obj_name)

        if obj_type == 'planet':
            marker_type = marker_planet
            marker_size = symsize_planet
            label = None
        
        elif obj_type == 'spacecraft':
            marker_type = marker_spacecraft
            marker_size = symsize_spacecraft
            label = obj_name.upper() + ":  " + pos['time'].strftime('%Y-%m-%d')

        scatter = ax.scatter(pos['lon'], pos['r']*np.cos(pos['lat']), s=marker_size, c=marker_color, marker=marker_type, lw=0, zorder=zorder, label=label)
        scatter_artists.append(scatter)
    
    return scatter_artists

@njit(fastmath=True, parallel=True)
def compute_cme_ellipses(hc_time_num1, hc_lon1, hc_lat1, a1_ell, b1_ell, c1_ell, num_gridpoints=200):
    num_entries = len(hc_time_num1)
    num_steps = np.shape(hc_time_num1[0])[0]
    longcirc = np.zeros((num_entries, 3, num_steps, num_gridpoints+1), dtype=np.float32)
    rcirc = np.zeros((num_entries, 3, num_steps, num_gridpoints+1), dtype=np.float32)
    alpha = np.zeros((num_entries, num_steps), dtype=np.float32)
    
    grid_rot = ((np.arange(num_gridpoints+1)-10) * np.pi / 180).astype(np.float32)

    for e in range(num_entries):
        for t in range(num_steps):
            lon_rad = hc_lon1[e][t] * np.pi / 180.0
            base = grid_rot - lon_rad
            lat_factor = np.abs(hc_lat1[e][t]) / 100.0
            alpha[e][t] = 1.0 - lat_factor

            for i in range(3):
                a = a1_ell[e][i][t]
                b = b1_ell[e][i][t]
                c = c1_ell[e][i][t]

                denom = np.sqrt((b * np.cos(grid_rot))**2 + (a * np.sin(grid_rot))**2)
                r = (a * b) / denom
                xc = c * np.cos(lon_rad) + r * np.sin(base)
                yc = c * np.sin(lon_rad) + r * np.cos(base)

                longcirc[e, i, t, :] = np.arctan2(yc, xc)
                rcirc[e, i, t, :] = np.sqrt(xc**2 + yc**2)

    return longcirc, rcirc, alpha

@njit(fastmath=True, parallel=True)
def compute_cme_ellipses_slow(hc_time_num1, hc_lon1, hc_lat1, a1_ell, b1_ell, c1_ell, num_gridpoints=200):
    num_steps = len(hc_time_num1)
    longcirc = np.zeros((3, num_steps, num_gridpoints+1), dtype=np.float32)
    rcirc = np.zeros((3, num_steps, num_gridpoints+1), dtype=np.float32)
    alpha = np.zeros(num_steps, dtype=np.float32)
    
    grid_rot = ((np.arange(num_gridpoints+1)-10) * np.pi / 180).astype(np.float32)

    for t in range(num_steps):
        lon_rad = hc_lon1[t] * np.pi / 180.0
        base = grid_rot - lon_rad
        lat_factor = np.abs(hc_lat1[t]) / 100.0
        alpha[t] = 1.0 - lat_factor

        for i in range(3):
            a = a1_ell[i][t]
            b = b1_ell[i][t]
            c = c1_ell[i][t]

            denom = np.sqrt((b * np.cos(grid_rot))**2 + (a * np.sin(grid_rot))**2)
            r = (a * b) / denom
            xc = c * np.cos(lon_rad) + r * np.sin(base)
            yc = c * np.sin(lon_rad) + r * np.cos(base)

            longcirc[i, t, :] = np.arctan2(yc, xc)
            rcirc[i, t, :] = np.sqrt(xc**2 + yc**2)

    return longcirc, rcirc, alpha

def update_spacecraft_artists(artists, pos_data):
    for scatter, pos in zip(artists, pos_data):
        scatter.set_offsets([[pos['lon'], pos['r'] * np.cos(pos['lat'])]])

def update_stereo_hi_fov(artists, fov_data):

    line1, line2, poly1, poly2 = artists

    lon0, r0, lon2, r2, lon3, r3, lonc11, rc11, lonc21, rc21, lonc12, rc12, lonc22, rc22 = fov_data
    line1.set_data([lon0, lon2], [r0, r2])
    line2.set_data([lon0, lon3], [r0, r3])

    # Update polygon vertices
    poly1.set_xy(np.array([[lon0, r0], [lonc11, rc11], [lonc21, rc21]]))
    poly2.set_xy(np.array([[lon0, r0], [lonc12, rc12], [lonc22, rc22]]))

def update_elongation_line(artists, fov_data):

    line1 = artists[0]

    lon0, r0, lon_elon, r_elon = fov_data
    line1.set_data([lon0, lon_elon], [r0, r_elon])


def update_cmes(artists, idx, ellipse_data):

    line, poly = artists
    longcirc, rcirc, alpha = ellipse_data

    if idx == -1:
        line.set_data([], [])
        line.set_alpha(0.0)
        poly.set_xy(np.column_stack([[],[]]))
        poly.set_alpha(0.0)

    else:
        lon_line = longcirc[0, idx, :]
        r_line   = rcirc[0, idx, :]

        line.set_data(lon_line, r_line)
        line.set_alpha(float(alpha[idx]))

        theta = longcirc[2][idx]
        r_inner = rcirc[2][idx]
        r_outer = rcirc[1][idx]

        # Concatenate outer curve + reversed inner curve
        theta_full = np.concatenate([theta, theta[::-1]])
        r_full = np.concatenate([r_outer, r_inner[::-1]])

        verts = np.column_stack([theta_full, r_full])

        poly.set_xy(verts)
        poly.set_alpha(0.05)

def make_frame_trajectories(positions,object_list,start_end=True,cmes=None,plot_stereo_fov=True,punch=True,trajectories=True, cme_tracks=None):

    gridcolor = '#052E37'
    fontsize = 13

    threshold_cme_in_frame = 60.0 # in seconds
    time_array = np.array(mdates.date2num(positions[object_list[0]]['time']))
    #time_array = np.array([item for items in time_array for item in items])

    for obj in object_list:
        if obj not in positions.keys():
            raise ValueError(f"Object {obj} not found in positions data.")
    
    # compute CME ellipses if CME data is provided
    if cmes is not None:
        if not all(key in cmes for key in ["hc_time_num1", "hc_r1", "hc_lat1", "hc_lon1", "a1_ell", "b1_ell", "c1_ell"]):
            raise ValueError("CME data is missing required keys.")

        # longcirc, rcirc, alpha = compute_cme_ellipses(cmes, num_gridpoints=200)
        longcirc = []
        rcirc = []
        alpha = []
        cme_times = []
        
        num_cmes = np.shape(cmes["hc_time_num1"])[0]
        cme_times = [np.asarray(cmes['hc_time_num1'][i], dtype=np.float64) for i in range(num_cmes)]

        time_array = np.asarray(time_array, dtype=np.float64)  # frame times
        frame_to_cme_idx = -np.ones((len(time_array), num_cmes), dtype=np.int32)

        # compute all ellipse parameters outside of loop
        # longcirc, rcirc, alpha = compute_cme_ellipses(
        #     cmes["hc_time_num1"].astype(np.float32),
        #     cmes["hc_lon1"].astype(np.float32),
        #     cmes["hc_lat1"].astype(np.float32),
        #     np.array(cmes["a1_ell"], dtype=np.float32),
        #     np.array(cmes["b1_ell"], dtype=np.float32),
        #     np.array(cmes["c1_ell"], dtype=np.float32),
        #     num_gridpoints=200
        # )

        # compute time difference for each CME and frame outside of loop

        longcirc = []
        rcirc = []
        alpha = []

        for cme_index in range(num_cmes):

            lc, rc, al = compute_cme_ellipses_slow(
                np.asarray(cmes["hc_time_num1"][cme_index], dtype=np.float32),
                np.asarray(cmes["hc_lon1"][cme_index], dtype=np.float32),
                np.asarray(cmes["hc_lat1"][cme_index], dtype=np.float32),
                np.asarray(cmes["a1_ell"][cme_index], dtype=np.float32),
                np.asarray(cmes["b1_ell"][cme_index], dtype=np.float32),
                np.asarray(cmes["c1_ell"][cme_index], dtype=np.float32),
                num_gridpoints=200
            )

            longcirc.append(lc)
            rcirc.append(rc)
            alpha.append(al)

            ctimes = cme_times[cme_index]

            diffs = np.abs(time_array[:, None] - ctimes[None, :]) * 24*3600.0
            valid = diffs < threshold_cme_in_frame

            if valid.any():
                nearest = np.argmin(diffs, axis=1)
                mask = valid.any(axis=1)
                frame_to_cme_idx[mask, cme_index] = nearest[mask]

        cme_color = get_object_color("cme")

    # initialize static plot elements
    fig,ax=plt.subplots(1,1,figsize = (10,10),dpi=100,subplot_kw={'projection': 'polar'}) #full hd

    ax.set_theta_zero_location('E')
    # plot the Sun in the center
    ax.scatter(0,0,s=100,c='#F9F200',alpha=1, edgecolors='black', linewidth=0.3)

    # plot the longitude grid
    ax.set_thetagrids(
        range(0,360,45),
        (u'0\u00b0',u'45\u00b0',u'90\u00b0',u'135\u00b0',u'+/- 180\u00b0       ',u'- 135\u00b0',u'- 90\u00b0',u'- 45\u00b0'),
        ha='center',
        fmt='%d',
        fontsize=fontsize-1,
        color=gridcolor,
        alpha=0.9,
        zorder=4)
    
    # plot the radial grid
    ax.set_rgrids(
        (0.1,0.3,0.5,0.7,1.0),
        ('0.10','0.3','0.5','0.7','1.0 AU'),
        angle=180,
        fontsize=fontsize-3,
        alpha=0.5,
        color=gridcolor)

    ax.set_ylim(0, 1.2)

    ax.set_autoscale_on(False)
    ax.autoscale(enable=False)
    fig.tight_layout(pad=0.0)  # once, outside loop

    if(start_end):
        ks = [0,-1]
    else:
        ks = np.arange(0,len(positions["l1"]))

    if plot_stereo_fov and ('sta' in object_list or 'stb' in object_list):
        stereo_sc = 'sta' if 'sta' in object_list else 'stb'
        fov_lines_at_step_k = calculate_stereo_fov_lines(dict(zip(list(positions[stereo_sc].keys()),np.array(list(positions[stereo_sc].values()))[:,0])), stereo_sc)
        fov_artists = plot_stereo_hi_fov(ax, fov_lines_at_step_k, stereo_sc, label_display=False)

    # TODO: implement CME tracks plotting
    # if cme_tracks is not None and ('sta' in object_list or 'stb' in object_list):
    #     stereo_sc = 'sta' if 'sta' in object_list else 'stb'
    #     elongation_lines_at_step_k = []
    #     elongation_artists = []
    #     track_times, track_elongations = cme_tracks
    #     for cme_number in range(len(track_times)):
    #         elongation_lines_at_step_k = calculate_elongation_lines(positions[stereo_sc], stereo_sc, track_elongations[cme_number], track_times[cme_number])
    #         elongation_artists = plot_elongation_line(ax, elongation_lines_at_step_k, stereo_sc, label_display=True)

    if cme_tracks is not None and ('sta' in object_list or 'stb' in object_list):
        stereo_sc = 'sta' if 'sta' in object_list else 'stb'
        elongation_lines_at_step_k = []
        elongation_artists = []
        track_times = cme_tracks['time']
        track_elongations = cme_tracks['elongation']

        time_num = [positions[stereo_sc]['time'][i] for i in range(len(positions[stereo_sc]['time']))]

        # interpolate tracks to time_num grid
        track_time_mdates = mdates.date2num(track_times)

        track_elongation_interp = np.interp(time_num, track_time_mdates, track_elongations, left=np.nan, right=np.nan)

        elongation_lines_at_step_k = calculate_elongation_lines(dict(zip(list(positions[stereo_sc].keys()),np.array(list(positions[stereo_sc].values()))[:,0])), stereo_sc, track_elongation_interp[0], track_time_mdates[0])
        elongation_artists = plot_elongation_line(ax, elongation_lines_at_step_k, stereo_sc, label_display=False)

    pos_at_step_k = [dict(zip(list(positions[obj].keys()),np.array(list(positions[obj].values()))[:,0])) for obj in object_list]
    # plot planets and spacecraft in object_list
    spacecraft_artists = plot_spacecraft_planets(ax, pos_at_step_k, object_list)

    if cmes is not None:
        cme_artists = []
        for cme_index in range(num_cmes):
            # initial plotting off-screen (or empty)
            line, = ax.plot([], [], color=cme_color, lw=1.5, alpha=0)  # empty line
            poly = Polygon(xy=np.column_stack([[],[]]), facecolor=cme_color, alpha=0.05, edgecolor='none', transform=ax.transData)
            ax.add_patch(poly)
            cme_artists.append((line, poly))

    for k in ks:
        start_timer = time.time()
        #plot all positions including text R lon lat for some 

        # get position at current step
        pos_at_step_k = [dict(zip(list(positions[obj].keys()),np.array(list(positions[obj].values()))[:,k])) for obj in object_list]
        # Plot time at step k outside of plot in top middle
        current_time = mdates.num2date(time_array[k]).strftime('%Y-%m-%d %H:%M:%S')
        plt.title(current_time, fontsize=fontsize+2, color=gridcolor)
        # plot planets and spacecraft in object_list
        update_spacecraft_artists(spacecraft_artists, pos_at_step_k)

        # plot stereoa fov hi1/2 and cor
        if plot_stereo_fov and ('sta' in object_list or 'stb' in object_list):
            fov_lines_at_step_k = calculate_stereo_fov_lines(dict(zip(list(positions[stereo_sc].keys()),np.array(list(positions[stereo_sc].values()))[:,k])), stereo_sc)
            update_stereo_hi_fov(fov_artists, fov_lines_at_step_k)

        if cmes is not None:
            for cme_index in range(num_cmes):
                idx = frame_to_cme_idx[k, cme_index]
                update_cmes(cme_artists[cme_index], idx, (longcirc[cme_index], rcirc[cme_index], alpha[cme_index]))

        if cme_tracks is not None and ('sta' in object_list or 'stb' in object_list):
            elongation_lines_at_step_k = calculate_elongation_lines(dict(zip(list(positions[stereo_sc].keys()),np.array(list(positions[stereo_sc].values()))[:,k])), stereo_sc, track_elongation_interp[k], time_num[k])
            update_elongation_line(elongation_artists, elongation_lines_at_step_k)
        plt.savefig('plots/ELEvo_'+str(k)+'.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor(), compress_level=1)

        end_timer = time.time()
        print(f"Frame {k+1}/{len(time_array)} done in {np.round(end_timer - start_timer, 2)} seconds.")
