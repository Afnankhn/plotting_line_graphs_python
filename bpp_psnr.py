# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:14:51 2023

@author: Image903
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, average
from scipy.interpolate import make_interp_spline, BSpline

# create data
bpp1 = [0.2507 ,   0.3531 ,   0.4412 ,  0.6248 ,  0.8690  ,  1.0179  ,  1.2292 ,   1.4274  ,  1.6002 ,   1.7980]
psnrhvs1 =[35.3277 ,  38.5016 ,  40.6429 ,  44.3029 ,  47.9778   ,49.6100  , 51.2810  , 52.5509  , 53.3105 ,  53.6986]

bpp2 =[0.3336 ,   0.4732 ,   0.5957  ,  0.7145  ,  0.8454   , 1.0447  ,  1.1725   , 1.3682 ,   1.6422,    1.8908]
psnrhvs2 =[36.0627  , 39.8157 ,  42.0753 ,  44.1341  , 45.5953   ,47.4200,   48.4954,   49.4124  , 50.5740  , 51.2555]

# psnravg = [[ 40.6429  , 42.5944 ,  44.3029,   46.6102 ,  47.9778   ,49.6100  , 51.2810  , 52.5509  , 53.3105 ,  53.6986],
#              [36.0627  , 39.8157 ,  42.0753 ,  44.1341  , 45.5953   ,47.4200,   48.4954,   49.4124  , 50.5740  , 51.2555]]

mean_bpp = np.mean((bpp1,bpp2), axis=0)
mean_psnr = np.mean((psnrhvs1,psnrhvs2), axis=0)


k=1
smooth = 20
bpp1_smooth = np.linspace(np.array(bpp1).min(), np.array(bpp1).max(), smooth) 
bpp1_spl = make_interp_spline(bpp1, psnrhvs1, k=k)
psnrhvs1_smooth = bpp1_spl(bpp1_smooth)

bpp2_smooth = np.linspace(np.array(bpp2).min(), np.array(bpp2).max(), smooth) 
bpp2_spl = make_interp_spline(bpp2, psnrhvs2, k=k)
psnrhvs2_smooth = bpp2_spl(bpp2_smooth)


mean_bpp_smooth = np.linspace(np.array(mean_bpp).min(), np.array(mean_bpp).max(), smooth) 
mean_bpp_spl = make_interp_spline(mean_bpp, mean_psnr, k=k)
mean_psnr_smooth = mean_bpp_spl(mean_bpp_smooth)

# Plot a simple line chart
plt.title("Rate_distortion")
plt.plot(bpp1_smooth, psnrhvs1_smooth, label='psnrhvs1', color='b', linewidth=1.0, marker='o')
plt.plot(bpp2_smooth, psnrhvs2_smooth, label='psnrhvs2', color='r', linewidth=1.0, marker='*')
plt.plot(mean_bpp_smooth, mean_psnr_smooth, label='Average', color='g', linewidth=1.0, marker='+')
plt.xlabel( 'BPP')
plt.ylabel( 'PSNR-HVS')
plt.legend()
#plt.grid( color = 'black', which='major', linestyle = ':', linewidth = 0.3)
plt.savefig("plot.png", dpi=600)
plt.show()

