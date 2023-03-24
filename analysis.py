# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import os, json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
myColor = list(mcolors.CSS4_COLORS)

from pygama.dsp.errors import DSPFatal
import pygama.lgdo.lh5_store as lh5


# %%
# DIRECTORY FOR PHYSICS DATA:
raw_dir='/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/raw/phy/p01/r027/'
dsp_dir='/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/dsp/phy/p01/r027/'
hit_dir='/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/hit/phy/p01/r027/'
tcm_dir='/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/tcm/phy/p01/r027/'
# DIRECTORY FOR CALIBRATION DATA:
# raw_dir = '/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/raw/cal/p01/r027/'
# dsp_dir = '/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/dsp/cal/p01/r027/'
# hit_dir = '/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/hit/cal/p01/r027/'
#tcm_dir = '/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/tcm/cal/p01/r027/'
# # SAMPLE DATA FILES
# raw_file = raw_dir+"l60-p01-r027-cal-20220923T165106Z-tier_raw.lh5" 
# dsp_file = raw_dir+"l60-p01-r026-phy-20220923T095129Z-tier_raw.lh5"
# hit_file = raw_dir+"l60-p01-r026-phy-20220923T095129Z-tier_raw.lh5"
# tcm_file = raw_dir+"l60-p01-r026-phy-20220923T095129Z-tier_raw.lh5

# %%
#List of all available channels, where ch000=pulse, ch001=muon, ch003=Ge, ...
channel= ['ch000', 'ch001', 
          'ch002', 'ch003', 'ch004', 'ch005', 'ch006', 'ch007', 'ch008', 'ch009', 'ch010', 
          'ch011', 'ch012', 'ch013', 'ch014', 'ch015', 'ch016', 'ch017', 'ch018', 'ch019', 'ch020', 'ch021', 
          'ch022', 'ch023', 'ch024', 'ch025', 'ch026', 'ch027', 'ch028', 'ch029', 'ch030', 'ch031', 'ch032', 
          'ch033', 'ch034', 'ch035', 'ch036', 'ch037', 'ch038', 'ch039', 'ch040', 'ch041', 'ch042', 'ch043', 
          'ch044', 'ch045', 'ch046', 'ch047', 'ch048', 'ch049', 'ch050', 'ch051', 'ch052', 'ch053', 'ch054', 
          'ch055', 'ch056', 'ch057', 'ch058', 'ch059', 'ch060', 'ch061', 'ch062', 'ch063', 'ch064', 'ch065', 
          'ch066', 'ch067', 'ch068', 'ch069', 'ch070', 'ch071', 'ch072', 'ch073', 'ch074', 'ch075', 'ch076', 
          'ch077', 'ch078', 'ch079', 'ch080', 'ch081', 'ch082', 'ch083', 'ch084', 'ch085', 'ch086', 'ch087', 
          'ch088', 'ch089', 'ch090', 'ch091', 'ch092', 'ch093', 'ch094', 'ch095', 'ch096', 'ch097', 'ch098', 
          'ch099', 'ch100', 'ch101', 'ch102', 'ch103', 'ch104', 'ch105', 'ch106', 'ch107', 'ch108', 'ch109']
string1= ["ch023","ch024","ch025","ch026","ch027","ch028","ch029","ch016"]
string2= ["ch009","ch010","ch011","ch012","ch013","ch014","ch015"]
string7= ["ch037","ch038","ch039","ch040","ch041","ch042","ch007","ch043"]
string8= ["ch002","ch003","ch008","ch005","ch006"]

list_of_Ge_channels= ["ch023","ch024","ch025","ch026","ch027","ch028","ch029","ch016",
                      "ch009","ch010","ch011","ch012","ch013","ch014","ch015",
                      "ch037","ch038","ch039","ch040","ch041","ch042","ch007","ch043",
                      "ch002","ch003","ch008","ch005","ch006"]
print(len(list_of_Ge_channels))

# %%
# get the raw files
raw_files=os.listdir(raw_dir) # list of all phy lh5 files
raw_files.sort() # sort them according to time
for i,file in enumerate(raw_files): 
    raw_files[i] =os.path.join(raw_dir,file) # get the full path
    #get the dsp files
dsp_files=os.listdir(dsp_dir) # list of all phy lh5 files
dsp_files.sort() # sort them according to time
for i,file in enumerate(dsp_files): 
    dsp_files[i] =os.path.join(dsp_dir,file) # get the full path

    #get the hit files
hit_files=os.listdir(hit_dir) # list of all phy lh5 files
hit_files.sort() # sort them according to time
for i,file in enumerate(hit_files): 
    hit_files[i] =os.path.join(hit_dir,file) # get the full path

    #get the tcm files
tcm_files=os.listdir(tcm_dir) # list of all phy lh5 files
tcm_files.sort() 
# sort them according to time
for i,file in enumerate(tcm_files): 
    tcm_files[i] =os.path.join(tcm_dir,file) # get the full path

# print(len(raw_files), len(dsp_files), len(hit_files), len(tcm_files))

# %%
raw_channels = lh5.ls(raw_files[0])

# %%
print(raw_channels)

# %%
lh5.ls(raw_files[0],'ch016/raw/')

# %%
# print("In RAW: \n ")
# show_raw = lh5.show(raw_files[0],"ch000")
# # choosing to show only the contents for ch000
# show_raw = lh5.show(raw_files[0])
# print("In DSP: \n ")
# show_dsp = lh5.show(dsp_files[0],"ch000")
# print("In HIT: \n ")
# show_hit = lh5.show(hit_files[0])
# print(" In TCM: \n ")
# show_tcm = lh5.show(tcm_files[0])

# %%
hit_channels = lh5.ls(hit_files[0])
# hit_channels

# %%
ge_channels_hit = hit_channels[:25]
sipm_channels_hit = hit_channels[25:]
# sipm_channels_hit

# %%
## Kernel dies when runnig this block, need to fix it.

# hit_channels = lh5.ls(hit_files[0])
# ge_channels_hit = hit_channels[:25]
# sipm_channels_hit = hit_channels[25:]

# energy = np.array([])
# aoe = np.array([])
# aoe_pass = np.array([], dtype=bool)
# energy_in_pe = np.array([])
# qual_cut = np.array([], dtype=bool)
# trig_pos = np.array([])
# for channel in ge_channels_hit:
#     ge_hit_data = lh5.load_nda(hit_files, 
#                         ["cuspEmax_ctc_cal", "AoE_Double_Sided_Cut", "AoE_Classifier"], f'{channel}/hit')
#     pulser_events = ge_hit_data["AoE_Classifier"]>10
#     energy=np.append(energy, ge_hit_data["cuspEmax_ctc_cal"])#[~pulser_events]
#     aoe_pass=np.append(aoe_pass, ge_hit_data["AoE_Double_Sided_Cut"])
#     aoe=np.append(aoe, ge_hit_data["AoE_Classifier"])
#     print(f"Ge channel: {channel} loaded")
# for channel in sipm_channels_hit:
#     sipm_hit_data = lh5.load_nda(hit_files, 
#                         ['energy_in_pe','quality_cut','trigger_pos'], f'{channel}/hit')
#     energy_in_pe=np.append(energy_in_pe, sipm_hit_data["energy_in_pe"])
#     qual_cut=np.append(qual_cut, sipm_hit_data["quality_cut"])
#     trig_pos=np.append(trig_pos, sipm_hit_data["trigger_pos"])
#     print(f" SiPM channel: {channel} loaded")


# %%
hit_channels = lh5.ls(hit_files[0])
ge_channels_hit = hit_channels[:25]
sipm_channels_hit = hit_channels[25:]

energy = np.array([])
aoe = np.array([])
aoe_pass = np.array([], dtype=bool)

for channel in ge_channels_hit:
    ge_hit_data = lh5.load_nda(hit_files, 
                        ["cuspEmax_ctc_cal", "AoE_Double_Sided_Cut", "AoE_Classifier"], f'{channel}/hit')
    pulser_events = ge_hit_data["AoE_Classifier"]>10
    energy=np.append(energy, ge_hit_data["cuspEmax_ctc_cal"])#[~pulser_events]
    aoe_pass=np.append(aoe_pass, ge_hit_data["AoE_Double_Sided_Cut"])
    aoe=np.append(aoe, ge_hit_data["AoE_Classifier"])
    print(f"Ge channel: {channel} loaded")

# %%
fig,ax = plt.subplots()
bins = np.linspace(0,3000,3000)
counts, edges, _ = ax.hist(energy, bins=bins, histtype='step', label='After quality cuts')
counts_pass, edges_pass, _ = ax.hist(energy[aoe_pass], bins=bins, histtype='step', label="After quality cuts and PSD cut")
ax.set_yscale('log')
plt.xlabel("Energy (keV)")
plt.ylabel("Counts / keV")
plt.legend()
plt.xlim(1400,1800)
# Count the number of events in the 1460 keV region
region_counts = np.sum((energy >= 1455) & (energy <= 1465))
region_counts_pass = np.sum((energy[aoe_pass] >= 1455) & (energy[aoe_pass] <= 1465))
print(f"Number of events in the 1460 +- 5 keV region: {region_counts}")
print(f"Number of events passing the PSD cut in the 1460 +- 5 keV region: {region_counts_pass}")

plt.show()


# %%
sto=lh5.LH5Store()

# %%
sto=lh5.LH5Store()
wfs = sto.read_object("ch087/raw/waveform", raw_files[0])

# %%
wfs

# %%
type(wfs)

# %%
len(wfs)

# %%
print(type(wfs[0]),type(wfs[1]))

# %%
len(wfs[0])

# %%
print(len(wfs[0]['t0']), len(wfs[0]['dt']),len(wfs[0]['values']))

# %%
wfs[0]['values']

# %%
plt.figure()
for i in range(1000):
    plt.plot(wfs[0]["values"].nda[i])
plt.show()

# %%
hit_phy_channel = lh5.ls(hit_files[0])
print(hit_phy_channel)

# %%
#### listing the parameters saved in raw file
ls_raw=lh5.ls(raw_files[0],'ch003/raw/')
ls_dsp=lh5.ls(dsp_files[0],'ch003/dsp/')
ls_hit=lh5.ls(hit_files[0],'ch003/hit/')
ls_tcm=lh5.ls(tcm_files[0])
raw_params= [x[10:] for x in ls_raw]
dsp_params= [x[10:] for x in ls_dsp]
hit_params= [x[10:] for x in ls_hit]
tcm_params= [x for x in ls_tcm]
print('Parameters in raw file are : ',raw_params,'\n')
print('Parameters in dsp file are : ',dsp_params,'\n')
print('Parameters in hit file are : ',hit_params,'\n')
print('Parameters in tcm file are : ',tcm_params,'\n')

# %%
#### listing the parameters saved in raw file
ls_raw=lh5.ls(raw_files[0],'ch103/raw/')
ls_dsp=lh5.ls(dsp_files[0],'ch103/dsp/')
ls_hit=lh5.ls(hit_files[0],'ch103/hit/')
ls_tcm=lh5.ls(tcm_files[0])
raw_params= [x[10:] for x in ls_raw]
dsp_params= [x[10:] for x in ls_dsp]
hit_params= [x[10:] for x in ls_hit]
tcm_params= [x for x in ls_tcm]
print('Parameters in raw file are : ',raw_params,'\n')
print('Parameters in dsp file are : ',dsp_params,'\n')
print('Parameters in hit file are : ',hit_params,'\n')
print('Parameters in tcm file are : ',tcm_params,'\n')


# %%
def dump_channel_info(raw_file):
    my_params = ['fcid','card','ch_orca','channel','crate']
    raw_info_dict = {}
    raw_channels = lh5.ls(raw_file)
    for iCh in raw_channels:
        raw_values = lh5.load_nda(raw_file, my_params, f'{iCh}/raw/')
        raw_info_dict[iCh]=raw_values 
    raw_info_table=pd.DataFrame.from_dict({k:[v['card'][0],v['fcid'][0],v['channel'][0],v['crate'][0]] for k,v in raw_info_dict.items()}, 
                                      orient='index',columns=['card','fcid','channel','crate'])
    print(raw_info_table.to_string())
    
# raw_file = "/data1/shared/l60/l60-prodven-v1/prod-ref/v06.00/generated/tier/raw/phy/p01/r027\
# /l60-p01-r027-phy-20220923T235251Z-tier_raw.lh5"
# dump_channel_info(raw_file)
    


# %%
my_params = ['fcid','card','ch_orca','channel','crate','baseline','daqenergy','timestamp']

# %%
values = lh5.load_nda(raw_files[10],my_params,'ch087/raw/')
values

# %%
print("Number of files in this run: ",len(raw_files))
print("Number of waveforms saved in this file",len(values["baseline"]))

# %%
plt.hist(values['daqenergy'])
plt.show()

# %%
values["timestamp"][-1] - values["timestamp"][0]

# %%
from pygama.vis.waveform_browser import WaveformBrowser

# %%
browser = WaveformBrowser(raw_files[10],'ch087/raw/',x_unit='us',styles={'color':list(mcolors.TABLEAU_COLORS)})
browser.draw_entry([1,2,3,4,5,6,7,8,9,10])

# %%
event_num = sto.read_object("ch001/raw/eventnumber", raw_files[0])

# %%
type(event_num)

# %%
event_num

# %%
sto=lh5.LH5Store()
sipm_attr = sto.read_object("ch087/hit", hit_files[0])

# %%
sipm_attr

# %%
type(sipm_attr)

# %%
type(sipm_attr[0])

# %%
sipm_attr[0]['energy_in_pe'].nda[10]

# %%
plt.figure()
for i in range(3080):
    plt.plot(sipm_attr[0]["energy_in_pe"].nda[i])
plt.show()

# %%
for i in range(3080):
    pe = sipm_attr[0]['energy_in_pe'].nda[i][0]
    if pe>30:
        print(i, pe)


# %%
wfs

# %%
maxwf=(1543,1886,2361)
for i in maxwf:
    plt.plot(wfs[0]['values'].nda[i], label=f"event number {i}")
    plt.legend()
plt.show()

# %% [markdown]
# ## Checking on Channel 087 only

# %%
sipm_hit_data = lh5.load_nda(hit_files,['energy_in_pe','trigger_pos','timestamp','quality_cut'],'ch087/hit')

# %% [markdown]
# ### PE histogram

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,500,500)
npe = sipm_hit_data["energy_in_pe"]
npe_without_nan = npe[~np.isnan(npe)]
ax.hist(npe_without_nan, bins=bins, histtype='step', label='number of PE')
ax.set_xlabel("Number of photo electrons")
ax.legend()
ax.set_yscale("log")
# ax.set_xticks(np.arange(0,21))
ax.grid()
plt.show()

# %% [markdown]
# ### Checking trigger postion

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,10000,10000)
trig = sipm_hit_data["trigger_pos"]
trig_without_nan = trig[~np.isnan(trig)]
ax.hist(trig_without_nan, bins=bins, histtype='step', label='trigger position')
ax.legend()
ax.set_yscale("log")
# ax.set_xticks(np.arange(0,21))
ax.grid()
plt.show()

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,10000,10000)
trig = sipm_hit_data["trigger_pos"]
trig_without_nan = trig[~np.isnan(trig)]
ax.hist(trig_without_nan, bins=bins, histtype='step', label='trigger position')
ax.set_xlim(2500,4000)
ax.legend()
ax.set_yscale("log")
# ax.set_xticks(np.arange(0,21))
ax.grid()
plt.show()

# %%
trig_mask = (trig_withou_nan>2900) &(trig_withou_nan<3200)

# %%
plt.hist(trig_withou_nan[trig_mask])

# %%
# trig_pos = sipm_data["trigger_pos"]
# trig_pos = trig_pos[~np.isnan(trig_pos)]
# trig_pos.shape

# %% [markdown]
# ### Read DSP files and extract energy using the trigger postion

# %%
lh5.show(dsp_files[0],'ch087/dsp')

# %%
sipm_dsp_data = lh5.load_nda(dsp_files,['energies','timestamp','trigger_pos'],'ch087/dsp')

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,200,200)
ax.hist(sipm_dsp_data["energies"][~np.isnan(sipm_dsp_data["energies"])], bins=bins, histtype='step',label='energy')
ax.set_xlabel("")
ax.legend()
ax.set_yscale("log")
# ax.set_xticks(np.arange(0,21))
ax.grid()
plt.show()

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,9000,9000)
ax.hist(sipm_dsp_data["trigger_pos"][~np.isnan(sipm_dsp_data["trigger_pos"])], bins=bins, histtype='step',label='trigger position')
ax.set_xlabel("")
ax.legend()
ax.set_yscale("log")
# ax.set_xticks(np.arange(0,21))
ax.grid()
plt.show()

# %%
trig_withou_nan = sipm_dsp_data["trigger_pos"][~np.isnan(sipm_dsp_data["trigger_pos"])]
trig_mask = (trig_withou_nan>2900) &(trig_withou_nan<3200)
trig_mask

# %%
print("Total: ", len(sipm_dsp_data["trigger_pos"]))
print("after trigger position cut:", len(trig_mask))

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,500,500)
energies = sipm_dsp_data["energies"][~np.isnan(sipm_dsp_data["energies"])]
trigger_pos = sipm_dsp_data["trigger_pos"][~np.isnan(sipm_dsp_data["trigger_pos"])]
trig_mask = (trigger_pos>2900) &(trigger_pos<3200)

ax.hist(energies, bins=bins, histtype='step',label='energy')
ax.hist(energies[trig_mask], bins=bins, histtype='step',label='after trig. pos. cut')
ax.set_ylabel("Counts")
ax.legend()
ax.set_yscale("log")
ax.grid()
plt.show()

# %% [markdown]
# ### For all SiPM channels:

# %%
for channel in sipm_channels_hit:
    sipm_dsp_data = lh5.load_nda(dsp_files,['energies','timestamp','trigger_pos'],f'{channel}/dsp')   

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,500,500)
energies = sipm_dsp_data["energies"][~np.isnan(sipm_dsp_data["energies"])]
trigger_pos = sipm_dsp_data["trigger_pos"][~np.isnan(sipm_dsp_data["trigger_pos"])]
trig_mask = (trigger_pos>2900) &(trigger_pos<3200)

ax.hist(energies, bins=bins, histtype='step',label='energy')
ax.hist(energies[trig_mask], bins=bins, histtype='step',label='after trig. pos. cut')
ax.set_ylabel("Counts")
ax.legend()
ax.set_yscale("log")
ax.grid()
plt.show()
