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
# channel= ['ch000', 'ch001', 
#           'ch002', 'ch003', 'ch004', 'ch005', 'ch006', 'ch007', 'ch008', 'ch009', 'ch010', 
#           'ch011', 'ch012', 'ch013', 'ch014', 'ch015', 'ch016', 'ch017', 'ch018', 'ch019', 'ch020', 'ch021', 
#           'ch022', 'ch023', 'ch024', 'ch025', 'ch026', 'ch027', 'ch028', 'ch029', 'ch030', 'ch031', 'ch032', 
#           'ch033', 'ch034', 'ch035', 'ch036', 'ch037', 'ch038', 'ch039', 'ch040', 'ch041', 'ch042', 'ch043', 
#           'ch044', 'ch045', 'ch046', 'ch047', 'ch048', 'ch049', 'ch050', 'ch051', 'ch052', 'ch053', 'ch054', 
#           'ch055', 'ch056', 'ch057', 'ch058', 'ch059', 'ch060', 'ch061', 'ch062', 'ch063', 'ch064', 'ch065', 
#           'ch066', 'ch067', 'ch068', 'ch069', 'ch070', 'ch071', 'ch072', 'ch073', 'ch074', 'ch075', 'ch076', 
#           'ch077', 'ch078', 'ch079', 'ch080', 'ch081', 'ch082', 'ch083', 'ch084', 'ch085', 'ch086', 'ch087', 
#           'ch088', 'ch089', 'ch090', 'ch091', 'ch092', 'ch093', 'ch094', 'ch095', 'ch096', 'ch097', 'ch098', 
#           'ch099', 'ch100', 'ch101', 'ch102', 'ch103', 'ch104', 'ch105', 'ch106', 'ch107', 'ch108', 'ch109']
# string1= ["ch023","ch024","ch025","ch026","ch027","ch028","ch029","ch016"]
# string2= ["ch009","ch010","ch011","ch012","ch013","ch014","ch015"]
# string7= ["ch037","ch038","ch039","ch040","ch041","ch042","ch007","ch043"]
# string8= ["ch002","ch003","ch008","ch005","ch006"]

# list_of_Ge_channels= ["ch023","ch024","ch025","ch026","ch027","ch028","ch029","ch016",
#                       "ch009","ch010","ch011","ch012","ch013","ch014","ch015",
#                       "ch037","ch038","ch039","ch040","ch041","ch042","ch007","ch043",
#                       "ch002","ch003","ch008","ch005","ch006"]
# print(len(list_of_Ge_channels))

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

# %% [markdown]
# #### Inspect the channels in a raw file

# %%
raw_channels = lh5.ls(raw_files[0])
print(raw_channels)
print("Number of raw channels: ",len(raw_channels))

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

# %% [markdown]
# #### Inspect the channels in a dsp file

# %%
dsp_channels = lh5.ls(dsp_files[0])
print(dsp_channels)
print("Number of dsp channels: ",len(dsp_channels)-1)

# %% [markdown]
# #### Inspect the channels in a hit file

# %%
hit_channels = lh5.ls(hit_files[0])
print(hit_channels)
print("Number of hit channels: ",len(hit_channels))

# %% [markdown]
# #### Look at the parameters saved in hit file. It provides which channel has what information. In a way this also help to identify the Ge channels and SiPM channels

# %%
lh5.show(hit_files[0])

# %% [markdown]
# #### Separating the Ge-channels and SiPM channels (This might not be a good way but it works for run27)

# %%
ge_channels_hit = hit_channels[:25]
print(ge_channels_hit)
print("Number of Ge-channels in a hit file in run027: ", len(ge_channels_hit))

# %%
sipm_channels_hit = hit_channels[25:]
print(sipm_channels_hit)
print("Number of SiPM-channels in a hit file in run027: ", len(sipm_channels_hit))

# %% [markdown]
# #### Listing the parameters saved in Ge-channels (jsut take one channel from ge_channels_list, all channels have same information)

# %%
#### listing the parameters saved in raw file
ls_raw=lh5.ls(raw_files[0],f'{ge_channels_hit[0]}/raw/')
ls_dsp=lh5.ls(dsp_files[0],f'{ge_channels_hit[0]}/dsp/')
ls_hit=lh5.ls(hit_files[0],f'{ge_channels_hit[0]}/hit/')
ls_tcm=lh5.ls(tcm_files[0])
raw_params= [x[10:] for x in ls_raw]
dsp_params= [x[10:] for x in ls_dsp]
hit_params= [x[10:] for x in ls_hit]
tcm_params= [x for x in ls_tcm]
print('Parameters in raw file are : ',raw_params,'\n')
print('Parameters in dsp file are : ',dsp_params,'\n')
print('Parameters in hit file are : ',hit_params,'\n')
print('Parameters in tcm file are : ',tcm_params,'\n')

# %% [markdown]
# #### Listing the parameters saved in SiPM-channels (jsut take one channel from SiPM_channels_list, all channels have same information)

# %%
#### listing the parameters saved in raw file
ls_raw=lh5.ls(raw_files[0],f'{sipm_channels_hit[0]}/raw/')
ls_dsp=lh5.ls(dsp_files[0],f'{sipm_channels_hit[0]}/dsp/')
ls_hit=lh5.ls(hit_files[0],f'{sipm_channels_hit[0]}/hit/')
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
def plot_energy_spectrum_run27(hit_files, xlo=1400, xhi=1700):
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
        
    fig,ax = plt.subplots(figsize=(10,6))
    bins = np.linspace(0,3000,3001)
    counts, edges, _ = ax.hist(energy, bins=bins, histtype='step', label='After quality cuts')
    counts_pass, edges_pass, _ = ax.hist(energy[aoe_pass], bins=bins, histtype='step', label="After quality cuts and PSD cut")
    ax.set_yscale('log')
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts / keV")
    plt.legend()
    # plt.xlim(xlo,xhi)
    # Count the number of events in the 1460 keV region
    region_counts = np.sum((energy >= 1455) & (energy <= 1465))
    region_counts_pass = np.sum((energy[aoe_pass] >= 1455) & (energy[aoe_pass] <= 1465))
    print(f"Number of events in the (1460 ± 5) keV region: {region_counts}")
    print(f"Number of events passing the PSD cut in the (1460 ± 5) keV region: {region_counts_pass}")
    plt.show()
    


# %%
plot_energy_spectrum_run27(hit_files)

# %%
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, InsetPosition, mark_inset

def plot_energy_spectrum_and_zoomin_run27(hit_files, xlo=1400, xhi=1700):
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
        
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=4, ncols=2, hspace=0, wspace=0)
    ax1 = fig.add_subplot(gs[:-1, :])
    ax2 = fig.add_subplot(gs[-1, -1])

    # Main plot
    bins = np.linspace(0, 3000, 3001)
    counts, edges, _ = ax1.hist(energy, bins=bins, histtype='step', label='After quality cuts')
    counts_pass, edges_pass, _ = ax1.hist(energy[aoe_pass], bins=bins, histtype='step', label="After quality and PSD cut")
    ax1.set_yscale('log')
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Counts / keV")
    ax1.legend(loc="upper left")
    # ax1.set_xlim(xlo, xhi)
    
    # Count the number of events in the 1460 keV region
    region_counts = np.sum((energy >= 1455) & (energy <= 1465))
    region_counts_pass = np.sum((energy[aoe_pass] >= 1455) & (energy[aoe_pass] <= 1465))
    print(f"Number of events in the (1460 ± 5) keV region: {region_counts}")
    print(f"Number of events passing the PSD cut in the (1460 ± 5) keV region: {region_counts_pass}")
    
    # Zoomed in plot
    ax2.hist(energy, bins=bins, histtype='step', label='After quality cuts')
    ax2.hist(energy[aoe_pass], bins=bins, histtype='step', label="After quality cuts and PSD cut")
    ax2.set_yscale("log")
    ax2.set_xlim(1425, 1575)
    ax2.tick_params(labelsize=8)
    # ax2.legend()
    
    # Adjust the size and position of the zoomed in plot
    bbox = ax1.get_position()
    x0, y0, width, height = bbox.bounds
    # ax2.set_position([x0+width-0.3, y0+height-0.2, 0.3*width, 0.3*height])
    ax2.set_position([x0+width-0.4, y0+height-0.3, 0.4*width, 0.4*height])
    plt.savefig('energy_spectrum.png', dpi=100)
    plt.show()
    


# %%
plot_energy_spectrum_and_zoomin_run27(hit_files)


# %% [markdown]
# #### t0: the waveform time offsets (ralative to a certain global reference), optionally with units; 
# #### dt: the waveform sampling periods, optionally with units; 
# #### values: the waveform values. May be Array of equal-sized arrays, Vector of vectors, etc; 
# #### t0 and dt must have one dimension (the last) less than values.

# %% tags=[]
def plot_wf(raw_file, channel, event_num=None):
    sto=lh5.LH5Store()
    wfs = sto.read_object(f"{channel}/raw/waveform", raw_file)
    print("Number of events in the raw file are: ", wfs[1])
    # print("t0 value is: ", wfs[0]["t0"].nda[event_num])
    # print("dt value is: ", wfs[0]["dt"].nda[event_num])
    print("generating the plot for event number: ",event_num)
    plt.plot(wfs[0]["values"].nda[event_num])
    plt.xlabel("ns")
    plt.ylabel("ADC count")
    plt.tight_layout()
    plt.show()
    
def plot_energy_in_pe(hit_file, channel, event_num=None, nbins=10):
    sipms_hit = lh5.load_nda(hit_file, ['energy_in_pe'],f'{channel}/hit/' )
    print("Number of events in the hit file are: ", len(sipms_hit["energy_in_pe"]))
    # convert nan values to 0
    # np.nan_to_num(sipms_hit["energy_in_pe"],nan=0.0)
    # discarding the nan values
    if event_num is None:
        pe_energy = sipms_hit["energy_in_pe"]
    else:
        pe_energy = sipms_hit["energy_in_pe"][event_num]

    pe_ene_event = pe_energy[~np.isnan(pe_energy)]  # this is required if you are selecting all events instead of specific event (to flatten the multi-dimension array)
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(pe_ene_event, bins=nbins,histtype='step', label='energy in pe')
    ax.set_xlabel('Energy in PE')
    ax.set_ylabel('Counts')
    ax.legend(loc='best')
    # ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    return n, bins


def plot_energy_in_pe_qcut(hit_file, channel, event_num=None, nbins=10):
    sipms_hit = lh5.load_nda(hit_file, ['energy_in_pe','quality_cut'],f'{channel}/hit/' )
    print("Number of events in the hit file are: ", len(sipms_hit["energy_in_pe"]))
    # convert nan values to 0
    # np.nan_to_num(sipms_hit["energy_in_pe"],nan=0.0)
    # discarding the nan values
    if event_num is None:
        pe_ene= sipms_hit["energy_in_pe"]
        q_cut = sipms_hit["quality_cut"]
    else:
        pe_ene= sipms_hit["energy_in_pe"][event_num]
        q_cut = sipms_hit["quality_cut"][event_num]
    ene_qcut = pe_ene[q_cut]
    pe_ene_qcut = ene_qcut[~np.isnan(ene_cut)]  # this is required if you are selecting all events instead of specific event (to flatten the multi-dimension array)
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(pe_ene_qcut, bins=nbins, histtype='step', label='pe after trig cut')
    ax.set_xlabel('Energy in PE')
    ax.set_ylabel('Counts')
    ax.legend(loc='best')
    # ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    return n, bins

def plot_energy_in_pe_trig_cut(hit_file, channel, event_num=None, trig_low=2900, trig_hi= 3200, nbins=10):
    sipms_hit = lh5.load_nda(hit_file, ['energy_in_pe','trigger_pos'],f'{channel}/hit/' )
    print("Number of events in the hit file are: ", len(sipms_hit["trigger_pos"]))
    # convert nan values to 0
    # np.nan_to_num(sipms_hit["energy_in_pe"],nan=0.0)
    # discarding the nan values
    if event_num is None:
        ene = sipms_hit["energy_in_pe"]
        trig = sipms_hit["trigger_pos"]
    else:
        ene = sipms_hit["energy_in_pe"][event_num]
        trig = sipms_hit["trigger_pos"][event_num]
    trig_mask = (trig>trig_low) & (trig<trig_hi)
    trig_cut = ene[trig_mask]
    ene_trig_cut = trig_cut[~np.isnan(trig_cut)]  # this is required if you are selecting all events instead of specific event (to flatten the multi-dimension array)
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(ene_trig_cut, bins=nbins, histtype='step', label = "pe after trig_pos cut")
    ax.set_xlabel('energy in pe')
    ax.set_ylabel('Counts')
    ax.legend(loc='best')
    # ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    return n, bins


def plot_trigger_pos(hit_file, channel, event_num=None, nbins=50):
    sipms_hit = lh5.load_nda(hit_file, ['trigger_pos'],f'{channel}/hit/' )
    print("Number of events in the hit file are: ", len(sipms_hit["trigger_pos"]))
    # convert nan values to 0
    # np.nan_to_num(sipms_hit["energy_in_pe"],nan=0.0)
    # discarding the nan values
    if event_num is None:
        trig = sipms_hit["trigger_pos"]
    else:
        trig = sipms_hit["trigger_pos"][event_num]
    trigpos = trig[~np.isnan(trig)]  # this is required if you are selecting all events instead of specific event (to flatten the multi-dimension array)
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(trigpos, bins=nbins, histtype='steo', label='trigge pos')
    ax.set_xlabel('trigger position')
    ax.set_ylabel('Counts')
    ax.legend(loc='best')
    # ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    return n, bins

def plot_sipm_energies(dsp_file, channel, event_num=None, nbins=10):
    sipms_dsp = lh5.load_nda(dsp_file,["energies"],f'{channel}/dsp/')
    energies = sipms_dsp["energies"]
    print("Number of events in this dsp file are: ", energies.shape[0])
    if event_num is None:
        ene = energies
    else:
        ene = energies[event_num]
    sipm_ene = ene[~np.isnan(ene)] # this is required if you are selecting all events instead of specific event (to flatten the multi-dimension array)
    fig, ax =  plt.subplots()
    n, bins, patches = ax.hist(sipm_ene, bins=nbins, histtype='step', label='energies')
    ax.set_xlabel('energies')
    ax.set_ylabel('Counts')
    ax.legend(loc='best')
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    return n, bins

def plot_sipm_energies_trig_cut(dsp_file, channel, event_num=None,trig_low=2900, trig_hi= 3200, nbins=10):
    sipms_dsp = lh5.load_nda(dsp_file, ['energies','trigger_pos'],f'{channel}/dsp/' )
    print("Number of events in the dsp file are: ", len(sipms_dsp["trigger_pos"]))
    if event_num is None:
        ene = sipms_dsp["energies"]
        trig = sipms_dsp["trigger_pos"]
    else:
        ene = sipms_dsp["energies"][event_num]
        trig = sipms_dsp["trigger_pos"][event_num]
    trig_mask = (trig>trig_low) & (trig<trig_hi)
    trig_cut = ene[trig_mask]
    ene_trig_cut = trig_cut[~np.isnan(trig_cut)]  # this is required if you are selecting all events instead of specific event (to flatten the multi-dimension array)
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(ene_trig_cut, bins=nbins, histtype='step', label = "energies after trig_pos cut")
    ax.set_xlabel('energies')
    ax.set_ylabel('Counts')
    ax.legend(loc='best')
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()
    return n, bins

def plot_channel_vs_summed_pe(hit_file, channel_list, event_num=None):
    ch_pe = {}
    for iCh in channel_list:
        sipms_hit = lh5.load_nda(hit_file, ['energy_in_pe','timestamp','trigger_pos'], f'{iCh}/hit/')
        if event_num is None:
            pe = sipms_hit["energy_in_pe"]
            ch_pe[iCh] = np.nan_to_num(pe,nan=0.0) # making all nan values 0, you cannot plot with nan values
        else:
            pe = sipms_hit["energy_in_pe"][event_num]
            ch_pe[iCh] = np.nan_to_num(pe,nan=0.0)
    # print(ch_pe.values())      
    pes = np.array(list(ch_pe.values())) # shape of pes is (channels, events, 100)
    print(pes.shape)
    if event_num is None:
        total_pe = np.sum(pes, axis=2)  # shape of total_pe is (channels, events)
        sum_pe = np.sum(total_pe, axis=1)
    else:
        sum_pe = np.sum(pes, axis=1)
    fig, ax =  plt.subplots(figsize=(20,7))
    ax.plot(list(ch_pe.keys()), sum_pe, marker="o", linestyle="", label = "energy in pe")
    ax.set_xlabel("Channels")
    ax.set_ylabel("summed energy in pe")
    ax.grid(which="both")
    # ax.set_yscale("log")
    # ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_channel_vs_pe(hit_file, channel_list, event_num):
    ch_pe = {}
    for iCh in channel_list:
        sipms_hit = lh5.load_nda(hit_file, ['energy_in_pe','timestamp','trigger_pos'], f'{iCh}/hit/')
        pe = sipms_hit["energy_in_pe"][event_num]
        ch_pe[iCh] = np.nan_to_num(pe,nan=0.0)
    fig, ax =  plt.subplots(figsize=(15,5))
    ax.plot(list(ch_pe.keys()), list(ch_pe.values()), marker="o", linestyle="", label = "energy in pe")
    ax.set_xlabel("Channels")
    ax.set_ylabel("energy in pe")
    ax.grid()
    # ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


# %%
plot_wf(raw_files[1], "ch068",1)

# %%
# def plot_energy_in_pe(hit_file, channel, event_num):
#     sto=lh5.LH5Store()
#     sipm_obj = sto.read_object(f'{channel}/hit/', hit_file)
#     sipms = sipm_obj[0]
#     print("Number of events in the hit file are: ", sipm_obj[1])
#     print("quality cut are : ", sipms["quality_cut"].nda[event_num])
#     print("trigger position is: ", sipms["trigger_pos"].nda[event_num])
#     print("timestamp is at: ", sipmm["timestamp"].nda[event_num])
#     print("generating the pe distribution for event number: ",event_num)
#     print(sipms["quality_cut"].nda[event_num])
#     # plt.hist(sipms['energy_in_pe'].nda[event_num][sipms["quality_cut"].nda[event_num]])
#     plt.hist(sipms['energy_in_pe'].nda[event_num])
#     plt.xlabel("number of pe")
#     plt.ylabel("counts")
#     plt.tight_layout()
#     plt.show()

# %%
# from pygama.vis.waveform_browser import WaveformBrowser
# browser = WaveformBrowser(raw_files[10],'ch087/raw/',x_unit='us',styles={'color':list(mcolors.TABLEAU_COLORS)})
# browser.draw_entry([1,2,3,4,5,6,7,8,9,10])

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
trig_mask = (trig_without_nan>2900) &(trig_without_nan<3200)

# %%
plt.hist(trig_without_nan[trig_mask])

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

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,200,200)
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

# %%

# %%
for channel in sipm_channels_hit:
    sipm_dsp_data1 = lh5.load_nda(dsp_files[0],['energies','timestamp','trigger_pos'],f'{channel}/dsp')
    sipm_hit_data1 = lh5.load_nda(hit_files[0],['energy_in_pe','timestamp','trigger_pos'],f'{channel}/hit')

# %%
sipm_dsp_data1["energies"]

# %%
sipm_hit_data1["energy_in_pe"].shape

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,200,200)
energies = sipm_dsp_data1["energies"][~np.isnan(sipm_dsp_data1["energies"])]
trigger_pos = sipm_dsp_data1["trigger_pos"][~np.isnan(sipm_dsp_data1["trigger_pos"])]
trig_mask = (trigger_pos>2900) &(trigger_pos<3200)

ax.hist(energies, bins=bins, histtype='step',label='energy')
ax.hist(energies[trig_mask], bins=bins, histtype='step',label='after trig. pos. cut')
ax.set_ylabel("Counts")
ax.legend()
ax.set_yscale("log")
ax.grid()
plt.show()

# %%
sipm_dsp_data1_ch087 = lh5.load_nda(dsp_files[0],['energies','timestamp','trigger_pos'],'ch087/dsp')  
sipm_hit_data1_ch087 = lh5.load_nda(hit_files[0],['energy_in_pe','timestamp','trigger_pos'],'ch087/hit')  

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,200,200)
energies = sipm_dsp_data1_ch087["energies"][~np.isnan(sipm_dsp_data1_ch087["energies"])]
trigger_pos = sipm_dsp_data1_ch087["trigger_pos"][~np.isnan(sipm_dsp_data1_ch087["trigger_pos"])]
trig_mask = (trigger_pos>2900) &(trigger_pos<3200)

ax.hist(energies, bins=bins, histtype='step',label='energy')
ax.hist(energies[trig_mask], bins=bins, histtype='step',label='after trig. pos. cut')
ax.set_ylabel("Counts")
ax.legend()
ax.set_yscale("log")
ax.grid()
plt.show()

# %%
sipm_dsp_data1_ch087["energies"]

# %%
sipm_dsp_data1_ch087["energies"].shape

# %%
sipm_dsp_data1_ch087["energies"][~np.isnan(sipm_dsp_data1_ch087["energies"])]

# %%
sipm_dsp_data1_ch087["energies"][~np.isnan(sipm_dsp_data1_ch087["energies"])].shape

# %%
np.nan_to_num(sipm_dsp_data1_ch087["energies"],nan=0.0)

# %%
np.nan_to_num(sipm_dsp_data1_ch087["energies"],nan=0.0).sum(axis=1)

# %%
np.nan_to_num(sipm_hit_data1_ch087["energy_in_pe"],nan=0.0)

# %%
np.nan_to_num(sipm_hit_data1_ch087["energy_in_pe"],nan=0.0).sum(axis=1)

# %%
np.nan_to_num(sipm_hit_data1_ch087["energy_in_pe"],nan=0.0).shape

# %%

# %%
r = np.random.rand(3,3)
q = np.random.randint(10, size=(3,3))

# %%
print(r ,'\n')
print(q)

# %%
sum_before_cut = np.sum(q, axis=1)
sum_before_cut

# %%
# Apply the trigger position cut 
mask = np.logical_and(r > 0,r < 0.5)
fltr = q[mask]
fltr = fltr.reshape(-1, 1)
sum_after_cut = fltr.sum(axis=1)
sum_after_cut

# %%
# energies = np.nan_to_num(sipm_dsp_data1_ch087["energies"], nan=0.0)
# trigger_pos = np.nan_to_num(sipm_dsp_data1_ch087["trigger_pos"], nan=0.0)

# sum_before_cut = np.sum(energies, axis=1)

# # Apply the trigger position cut 
# mask = np.logical_and(trigger_pos > 2900, trigger_pos < 3200)
# e_filtered = energies[mask]
# print("Sum of energies before cut:", sum_before_cut)

# # Reshape e_filtered to have shape (N, 1)
# e_filtered = pe_filtered.reshape(-1, 1)

# sum_after_cut = e_filtered.sum(axis=1)
# print("Sum of energies after trigger position cut:", sum_after_cut)

# %%
# fig,ax = plt.subplots(figsize=(12,7))
# bins = np.linspace(0,200,200)
# energies = np.nan_to_num(sipm_dsp_data1_ch087["energies"],nan=0.0)
# trigger_pos = np.nan_to_num(sipm_dsp_data1_ch087["trigger_pos"],nan=0.0)
# sum_before_cut = np.sum(energies, axis=1)

# # Apply the trigger position cut 
# mask = np.logical_and(trigger_pos > 2900, trigger_pos < 3200)
# e_filtered = energies[mask]
# e_filtered = pe_filtered.reshape(-1, 1)
# sum_after_cut = e_filtered.sum(axis=1)

# ax.hist(sum_before_cut, bins=bins, histtype='step',label='sum-energy')
# ax.hist(sum_after_cut, bins=bins, histtype='step',label='sum-energy after trig-pos. cut')
# ax.set_ylabel("Counts")
# ax.legend()
# ax.set_yscale("log")
# ax.grid()
# plt.show()

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,200,200)
energy_pe = np.nan_to_num(sipm_hit_data1_ch087["energy_in_pe"],nan=0.0)
trigger_pos = np.nan_to_num(sipm_hit_data1_ch087["trigger_pos"],nan=0.0)
sum_before_cut = np.sum(energy_pe, axis=1)

# Apply the trigger position cut 
mask = np.logical_and(trigger_pos > 2900, trigger_pos < 3200)
pe_filtered = energy_pe[mask]
pe_filtered = pe_filtered.reshape(-1, 1)
sum_after_cut = pe_filtered.sum(axis=1)

ax.hist(sum_before_cut, bins=bins, histtype='step',label='sum of pe')
ax.hist(sum_after_cut, bins=bins, histtype='step',label='sum of pe after trig-pos. cut')
ax.set_ylabel("Counts")
ax.legend()
ax.set_yscale("log")
# ax.set_xlim(0,20)
ax.grid()
plt.show()

# %%
len(sipm_channels_hit)

# %%
for channel in sipm_channels_hit:
    sipm_dsp_data = lh5.load_nda(dsp_files[0],['energies','timestamp','trigger_pos'],f'{channel}/dsp/')
    sipm_hit_data = lh5.load_nda(hit_files[0],['energy_in_pe','timestamp','trigger_pos'],f'{channel}/hit/')

# %%
fig,ax = plt.subplots(figsize=(12,7))
bins = np.linspace(0,200,200)
energy_pe = np.nan_to_num(sipm_hit_data["energy_in_pe"],nan=0.0)
trigger_pos = np.nan_to_num(sipm_hit_data["trigger_pos"],nan=0.0)
sum_before_cut = np.sum(energy_pe, axis=1)

# Apply the trigger position cut 
mask = np.logical_and(trigger_pos > 2900, trigger_pos < 3200)
pe_filtered = energy_pe[mask]
pe_filtered = pe_filtered.reshape(-1, 1)
sum_after_cut = pe_filtered.sum(axis=1)

ax.hist(sum_before_cut, bins=bins, histtype='step',label='sum of pe')
ax.hist(sum_after_cut, bins=bins, histtype='step',label='sum of pe after trig-pos. cut')
ax.set_ylabel("Counts")
ax.legend()
ax.set_yscale("log")
# ax.set_xlim(0,20)
ax.grid()
plt.show()

# %%
sipm_hit_data["energy_in_pe"]

# %%
