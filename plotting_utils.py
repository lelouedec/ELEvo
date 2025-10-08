
import numpy as np 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import Utils
import matplotlib.pyplot as plt 
from sunpy.time import parse_time
import matplotlib as mpl
from scipy.stats import norm
import seaborn as sns

# animation settings 

# fadeind = 200*24 #if s/c positions are given in hourly resolution

# #for parker spiral   
# theta=np.arange(0,np.deg2rad(180),0.01)
# cme_color='#8C99FD'

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
    x0=pos.x[timeind]
    y0=pos.y[timeind]
    z0=0

    x1=-pos.x[timeind]
    y1=-pos.y[timeind]
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

def plot_stereo_hi_fov(ax, pos,sc,label_display=False):    
    
    #plots the STA FOV HI1 HI2
    
    time_num = pos.time
    #STB never flipped the camera:
    sc = sc.lower()
    if sc=='stb': 
        ang1d=-4
        ang2d=-24
        ang3d=-18
        ang4d=-88
        lcolor='blue'
    
    if sc=='sta': 
        ang1d=4
        ang2d=24
        ang3d=18
        ang4d=88
        lcolor='red'

        #STA flipped during conjunction
        if mdates.date2num(datetime(2015,11,1))<time_num<mdates.date2num(datetime(2023,8,12)):  
            ang1d=-4
            ang2d=-24
            ang3d=-18
            ang4d=-88

    #calculate endpoints
    
    #sta position
    x0=pos.x
    y0=pos.y
    z0=0
    
    #sta position 180° rotated    
    x1=-pos.x
    y1=-pos.y
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
    # r2,t2,lon2=angle_to_coord_line(45,x0,y0,x1,y1) 

    #ax.plot([lon0,lon1],[r0,r1],'--r',alpha=0.5)

    if(label_display):
        label1="STEREO-A/HI1 Field of View"
        label2="STEREO-A/COR2 Field of View"
    else:
        label1=""
        label2=""


    ax.plot([lon0,lon2],[r0,r2],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon3],[r0,r3],linestyle='-',color=lcolor,alpha=0.5, lw=1.2,label=label1)

    ax.fill([lon0,lonc11,lonc21],[r0,rc11,rc21],color=lcolor,alpha=0.3)
    ax.fill([lon0,lonc12,lonc22],[r0,rc12,rc22],color=lcolor,alpha=0.3,label=label2)
    # ax.plot([lon0,lon4],[r0,r4],linestyle='--',color=lcolor,alpha=0.3, lw=0.8)
    # ax.plot([lon0,lon5],[r0,r5],linestyle='--',color=lcolor,alpha=0.3, lw=0.8)




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
                label = obj_name.upper() + ":  " + mdates.num2date(pos.time[0]).strftime('%Y-%m-%d')

            ax.scatter(pos.lon, pos.r*np.cos(pos.lat), s=marker_size, c=marker_color, marker=marker_type, lw=0, zorder=zorder, label=label)

def compute_cme_ellipses(cmes, num_gridpoints=200):

    num_steps = len(cmes["hc_time_num1"])
    longcirc = np.zeros((3, num_steps, num_gridpoints+1))
    rcirc = np.zeros((3, num_steps, num_gridpoints+1))
    alpha = np.zeros(num_steps)

    for t_ind in range(num_steps):

        grid_base = ((np.arange(num_gridpoints+1)-10)*np.pi/180)-(cmes["hc_lon1"][t_ind]*np.pi/180)
        grid_rot = ((np.arange(num_gridpoints+1)-10)*np.pi/180)

        for i in range(3):

            xc = cmes["c1_ell"][i][t_ind]*np.cos(cmes["hc_lon1"][t_ind]*np.pi/180)+((cmes["a1_ell"][i][t_ind]*cmes["b1_ell"][i][t_ind])/np.sqrt((cmes["b1_ell"][i][t_ind]*np.cos(grid_rot))**2+(cmes["a1_ell"][i][t_ind]*np.sin(grid_rot))**2))*np.sin(grid_base)
            yc = cmes["c1_ell"][i][t_ind]*np.sin(cmes["hc_lon1"][t_ind]*np.pi/180)+((cmes["a1_ell"][i][t_ind]*cmes["b1_ell"][i][t_ind])/np.sqrt((cmes["b1_ell"][i][t_ind]*np.cos(grid_rot))**2+(cmes["a1_ell"][i][t_ind]*np.sin(grid_rot))**2))*np.cos(grid_base)

            longcirc[i][t_ind] = np.arctan2(yc, xc)
            rcirc[i][t_ind] = np.sqrt(xc**2+yc**2)
            alpha[t_ind] = 1 - abs(cmes["hc_lat1"][t_ind]/100)
    
    return longcirc, rcirc, alpha


def make_frame_trajectories(positions,object_list,start_end=True,cmes=None,plot_stereo_fov=True,punch=True,trajectories=True):

    gridcolor = '#052E37'
    fontsize = 13

    threshold_cme_in_frame = 60.0 # in seconds
    time_array = positions[object_list[0]]['time']
    time_array = np.array([item for items in time_array for item in items])

    for obj in object_list:
        if obj not in positions.keys():
            raise ValueError(f"Object {obj} not found in positions data.")
    
    # compute CME ellipses if CME data is provided
    if cmes is not None:
        if not all(key in cmes for key in ["hc_time_num1", "hc_r1", "hc_lat1", "hc_lon1", "a1_ell", "b1_ell", "c1_ell"]):
            raise ValueError("CME data is missing required keys.")

        longcirc, rcirc, alpha = compute_cme_ellipses(cmes, num_gridpoints=200)
        
        cme_times = cmes["hc_time_num1"]
        cme_color = get_object_color("cme")

    fig,ax=plt.subplots(1,1,figsize = (10,10),dpi=100,subplot_kw={'projection': 'polar'}) #full hd

    if(start_end):
        ks = [0,-1]
    else:
        ks = np.arange(0,len(positions["l1"]))

    for k in ks:
        current_time = mdates.num2date(time_array[k])
        #plot all positions including text R lon lat for some 

        # get position at current step
        pos_at_step_k = [positions[obj][k] for obj in object_list]


        # plot planets and spacecraft in object_list
        plot_spacecraft_planets(ax, pos_at_step_k, object_list)

        # plot stereoa fov hi1/2 and cor
        if plot_stereo_fov and ('sta' in object_list or 'stb' in object_list):
            stereo_sc = 'sta' if 'sta' in object_list else 'stb'
            plot_stereo_hi_fov(ax, positions[stereo_sc][k], stereo_sc, label_display=False)

        if cmes is not None:
            #plot_cmes_old(ax,cmes,k,current_time,res_in_days)

            # calculate differene between current time and cme start time
            cme_time_diff_to_current = [np.abs((mdates.num2date(cme_times[i])- current_time).total_seconds()) for i in range(0,len(cme_times))]
            # get indices of cmes that are in the frame
            cmes_at_step_k_ind = np.where(np.array(cme_time_diff_to_current)<threshold_cme_in_frame)[0]

            if len(cmes_at_step_k_ind) > 0:
                for cme_ind in cmes_at_step_k_ind:
                    ax.plot(longcirc[0][cme_ind], rcirc[0][cme_ind], color=cme_color, ls='-', lw=1.5, alpha=alpha[cme_ind])
                    ax.fill_between(longcirc[2][cme_ind], rcirc[2][cme_ind], rcirc[1][cme_ind], color=cme_color, alpha=0.05)

        ax.set_theta_zero_location('E')
        # plot the Sun in the center
        ax.scatter(0,0,s=100,c='#F9F200',alpha=1, edgecolors='black', linewidth=0.3)

        # plot the longitude grid
        plt.thetagrids(
            range(0,360,45),
            (u'0\u00b0',u'45\u00b0',u'90\u00b0',u'135\u00b0',u'+/- 180\u00b0       ',u'- 135\u00b0',u'- 90\u00b0',u'- 45\u00b0'),
            ha='center',
            fmt='%d',
            fontsize=fontsize-1,
            color=gridcolor,
            alpha=0.9,
            zorder=4)
        
        # plot the radial grid
        plt.rgrids(
            (0.1,0.3,0.5,0.7,1.0),
            ('0.10','0.3','0.5','0.7','1.0 AU'),
            angle=180,
            fontsize=fontsize-3,
            alpha=0.5,
            color=gridcolor)

        ax.set_ylim(0, 1.2)

        plt.savefig('plots/trajectories_test_'+str(k)+'.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        if (not start_end):
            fig,ax=plt.subplots(1,1,figsize = (10,10),dpi=100,subplot_kw={'projection': 'polar'})    


    # draw_punch_fov(earth,frame_time_num,-1,ax)
    # angle = np.deg2rad(67.5)
    # ax.legend(loc="lower left",
    #         bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))

    # plt.savefig('plots/trajectories'++'.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    # plt.close(fig)