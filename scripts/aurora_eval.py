import numpy as np
import pandas as pd
import xarray as xr
import yaml

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from aurora import Batch, Metadata
from aurora import Aurora, AuroraSmallPretrained, AuroraPretrained
from aurora import rollout
from aurora.normalisation import locations, scales, normalise_atmos_var, normalise_surf_var

from pathlib import Path
import time
import argparse

# from huggingface_hub import hf_hub_download

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from time import perf_counter

from aurora import rollout, rollout_updated, rollout_nudged

from src.data.batch import make_batch_era5, make_batch_merra2, trim_batch_lat
from src.training.loss_funcs import AuroraL1Loss, AuroraL1Loss_NoWeight, AuroraL1Loss_PolarMask
from src.training.scheduler import make_lr_lambda, lr_lambda, lr_lambda_nowarm, EarlyStopping
from src.utils.utils import memcheck, load_config

aur_levs = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

def one_step_eval(run_name,
                  model, 
                  roll_start,
                  roll_end, 
                  species_list, 
                  interp_scheme_gc,
                  met_type,
                  met_list,
                  patch_size=1,
                  device=0):

    ## Initialize dicts for outputs
    preds_dict = {}
    trues_dict= {}

    surf_vars=("2t", "10u", "10v", "msl")
    atmos_vars = ("z", "u", "v", "t", "q") + tuple(species_list)

    for var in surf_vars:
        preds_dict[var] = []
        trues_dict[var] = []

    for var in atmos_vars:
        for lvl in aur_levs:
            preds_dict[f"{var}_{lvl}"] = []
            trues_dict[f"{var}_{lvl}"] = []

    ## Make list of rollout years
    years = list(range(roll_start.year, roll_end.year + 1))

    ## Compile GEOS-Chem fields
    chem_data_test = {}
    chem_weights = {}
    for species in species_list:
        
        gc_filled_all_list = []
        for year in years:
            gc_filled_test = xr.open_dataset(f"{config['data_dir']}/GEOS_Chem/{gc_version}/time_avg/4x5/{year}/{species}_filled_{year}_interp{interp_scheme_gc}.nc")
            gc_filled_all_list.append(gc_filled_test)
        gc_filled_all = xr.concat(gc_filled_all_list, dim="time")
        
        gc_filled_all = gc_filled_all.sel(time=slice(roll_start, roll_end))
        gc_filled_all = gc_filled_all[species][::2]
        chem_data_test[species] = torch.from_numpy(gc_filled_all.values[None])

        ## Normalize GC data
        # gc_scales = 0.5*gc_filled_all.max(axis=(-2, -1)).mean(axis=0).values
        # gc_maes = np.mean(np.abs(gc_filled_all.values[:-1] - gc_filled_all.values[1:]), axis=(0, 2, 3))

        norm_stats = np.load(f"{config['output_dir']}/Weights/{run_name}_norm_stats.npz")
        gc_scales = norm_stats[f"{species}_scales"]
        gc_maes = norm_stats[f"{species}_maes"]

        gc_weights = torch.from_numpy(gc_scales / gc_maes).to(torch.device(f"cuda:{device}"))

        chem_weights[species] = gc_weights
        
        pressure_lvls = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
        for idx, lev in enumerate(pressure_lvls):
            locations[f"{species}_{str(int(lev))}"] = 0.0
            scales[f"{species}_{str(int(lev))}"] = gc_scales[idx]

    print(chem_data_test["O3"].shape)

    ## Compile static ERA5 fields
    static_data_test = {}
    for year in years:
        static_data_test[year] = xr.open_dataset(f"{config['data_dir']}/era5_interp/{year}/static_{year}_interp{interp_scheme}.nc", engine="netcdf4")
    
    ## Compile MERRA2 fields
    if met_type == "merra2":
        
        merra2_data = {}
        merra2_path = f"{config['data_dir']}/MERRA2/time_avg/4x5"

        for met in met_list:

            print(f"Loading {met}.", flush=True)

            merra2_filled = xr.open_mfdataset([f"{merra2_path}/{train_year}/{met}_filled_{train_year}_interp_uniform_fix.nc" for train_year in years],
                                            combine="by_coords",
                                            engine="netcdf4")
    
            if met in ["TS", "U10M", "V10M", "PSC2WET"]:
                merra2_filled = merra2_filled[f"Met_{met}"][::2]
            else:
                merra2_filled = merra2_filled[met][::2]

            print(merra2_filled.shape, flush=True)

            if met == "SPHU":
                merra2_filled = merra2_filled / 1000
            if met == "Z":
                merra2_filled = merra2_filled * 9.81

            merra2_data[met] = torch.from_numpy(merra2_filled.values[None])
    
    ## Make list of datetimes for full test domain
    n_test = (roll_end - roll_start) // timedelta(hours=6) - 1
    test_idxs = (np.arange(0, n_test)).astype(int)

    # Build full datetime sequence (not just dates), extended by 2 extra steps
    datetimes_test = [roll_start + timedelta(hours=6*i) for i in range(n_test + 2)]

    ## Make lists of test set days and hours
    test_days_t1 = np.array([datetimes_test[i].date() for i in test_idxs])
    test_days_t2 = np.array([datetimes_test[i+1].date() for i in test_idxs])
    test_days_t3 = np.array([datetimes_test[i+2].date() for i in test_idxs])

    test_hours_t1 = np.array([datetimes_test[i].hour // 6 for i in test_idxs])
    test_hours_t2 = np.array([datetimes_test[i+1].hour // 6 for i in test_idxs])
    test_hours_t3 = np.array([datetimes_test[i+2].hour // 6 for i in test_idxs])
    
    ## Specify loss function
    loss_func = AuroraL1Loss_PolarMask()

    ## Testing Loop
    for i, day_idx in enumerate(test_idxs):

        start = time.time()

        ## Select correct day and hour
        day_t1 = test_days_t1[i]
        day_t2 = test_days_t2[i]
        day_t3 = test_days_t3[i]
        hour_t1 = test_hours_t1[i]
        hour_t2 = test_hours_t2[i]
        hour_t3 = test_hours_t3[i]

        ## Construct input and true batches
        if met_type == "merra2":
            input_batch, true_batch = make_batch_merra2(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                                        day_idx, chem_data_test, merra2_data, 
                                                        static_data_test, interp_scheme, new_static, config['data_dir'])
        else:
            input_batch, true_batch = make_batch_era5(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                                        day_idx, chem_data_test, 
                                                        static_data_test, interp_scheme, new_static, config['data_dir'])

        input_batch = input_batch.to(torch.device(f"cuda:{device}"))
        input_batch = trim_batch_lat(input_batch, "input")
        true_batch = trim_batch_lat(true_batch, "true")
    
        model.eval()      
        start_forward = time.time()

        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            pred_test = model.forward(input_batch)
        peak_bytes = torch.cuda.max_memory_allocated()
        
        pred_test = pred_test.to(torch.device(f"cuda:{device}"))
        time_forward = time.time() - start_forward
        true_batch = true_batch.to(torch.device(f"cuda:{device}"))
        loss = loss_func(pred_test, true_batch, chem_weights, patch_size=1)
 
        for var in pred_test.surf_vars:
            preds_dict[var].append(pred_test.surf_vars[var][-1, 0].cpu().detach().numpy())
            if patch_size % 2 == 0:
                trues_dict[var].append(true_batch.surf_vars[var][0, :-1].cpu().detach().numpy())
            else:
                trues_dict[var].append(true_batch.surf_vars[var][0].cpu().detach().numpy())

        for var in pred_test.atmos_vars:
            for lvl_idx, lvl in enumerate(pressure_lvls):
                preds_dict[f"{var}_{lvl}"].append(pred_test.atmos_vars[var][-1, 0, lvl_idx].cpu().detach().numpy())
                if patch_size % 2 == 0:    
                    trues_dict[f"{var}_{lvl}"].append(true_batch.atmos_vars[var][0, lvl_idx, :-1].cpu().detach().numpy())
                else:
                    trues_dict[f"{var}_{lvl}"].append(true_batch.atmos_vars[var][0, lvl_idx].cpu().detach().numpy())

        print(f"MODEL EVALUATED FOR DAY {day_idx}, Runtime: {np.round(time.time() - start, 2)}, Forward: {np.round(time_forward, 2)}, TEST LOSS: {np.round(loss.item(), 3)}, MEM USED: {peak_bytes / 1024**2}", flush=True)

    ## Construct metadata arrays for output files
    time_dim = [datetime.combine(test_days_t3[i], 
                                 datetime.min.time()) + timedelta(hours=int(test_hours_t3[i]*6)) for i in range(len(test_days_t3))]

    if patch_size % 2 == 0:
        coords = {
            "time":time_dim,
            "lat":input_batch.metadata.lat.cpu().detach().numpy()[:-1],
            "lon":input_batch.metadata.lon.cpu().detach().numpy()
        }
    else:
        coords = {
            "time":time_dim,
            "lat":input_batch.metadata.lat.cpu().detach().numpy(),
            "lon":input_batch.metadata.lon.cpu().detach().numpy()
        }

    preds_dict_final = {}
    trues_dict_final = {}
    
    for k in preds_dict:
        preds_dict_final[k] = (("time", "lat", "lon"), np.array(preds_dict[k]))
        trues_dict_final[k] = (("time", "lat", "lon"), np.array(trues_dict[k]))
    
    ds_preds = xr.Dataset(preds_dict_final, coords=coords)
    ds_trues = xr.Dataset(trues_dict_final, coords=coords)

    ds_preds.to_netcdf(f"{config['output_dir']}/{run_name}_{roll_start:%Y%m%d}_{roll_end:%Y%m%d}_ONESTEP.nc")
    ds_trues.to_netcdf(f"{config['output_dir']}/{run_name}_{roll_start.year}{roll_start.month}_{roll_end.year}{roll_end.month}_TRUES.nc")

def rollout_eval(run_name,
                 model, 
                 rollout_type, 
                 roll_start,
                 roll_end,
                 nudge_int,
                 met_nudge_list,
                 chem_nudge_list,
                 sat_nudge_list,
                 species_list,
                 interp_scheme_gc,
                 met_type,
                 met_list,
                 patch_size=1,
                 device=0):
    
    file_suffix_dict = {
        "free": "ROLLOUTS_MATCHED_EVAL_MAES",
        "naive": "ROLLOUTS_NAIVE",
        "nudged": f"ROLLOUTS_{nudge_int}_NUDGED",
        "nudged_met": f"ROLLOUTS_{nudge_int}_NUDGED_METONLY_MATCHED_EVAL_MAES_DOY",
        "nudged_chem": f"ROLLOUTS_{nudge_int}_NUDGED_CHEMONLY_MATCHED_EVAL_MAES_DOY",
        "nudged_time": "ROLLOUTS_TIMEONLY",
        "nudged_emis": "ROLLOUTS_EMISONLY",
        "nudged_tropomi_no2": "ROLLOUTS_TROPOMI_NO2_CORRECTED_MASK",
        "nudged_tropomi_no2_ak": "ROLLOUTS_TROPOMI_NO2_AK_CORRECTED_MASK",
        "nudged_free": "ROLLOUTS_FREE_MATCHED_EVAL_DOY",
    }

    chem_nudge = met_nudge = sat_nudge = avg_kernel = False
    if rollout_type == "nudged":
        chem_nudge = True
        met_nudge = True
        sat_nudge = False
        avg_kernel = False
    elif rollout_type == "nudged_met":
        chem_nudge = False
        met_nudge = True
        sat_nudge = False
        avg_kernel = False
    elif rollout_type == "nudged_chem":
        chem_nudge = True
        met_nudge = False
        sat_nudge = False
        avg_kernel = False
    elif rollout_type == "nudged_partial":
        file_suffix_dict[rollout_type] = f"ROLLOUTS_{nudge_int}_NUDGED_" + "_".join(met_nudge_list + chem_nudge_list)
        chem_nudge = True
        met_nudge = True
        sat_nudge = False
        avg_kernel = False
    elif rollout_type == "nudged_tropomi_no2":
        sat_nudge = True
        chem_nudge = False
        met_nudge = False
        avg_kernel = False
    elif rollout_type == "nudged_tropomi_no2_ak":
        sat_nudge = True
        chem_nudge = False
        met_nudge = False
        avg_kernel = True
    elif rollout_type == "nudged_free":
        sat_nudge = False
        chem_nudge = False
        met_nudge = False
        avg_kernel = False

    ## Initialize dicts for outputs
    rollouts_dict= {}

    surf_vars=("2t", "10u", "10v", "msl")
    atmos_vars = ("z", "u", "v", "t", "q") + tuple(species_list)

    for var in surf_vars:
        rollouts_dict[var] = []

    for var in atmos_vars:
        for lvl in aur_levs:
            rollouts_dict[f"{var}_{lvl}"] = []

    ## Make list of rollout years
    years = list(range(roll_start.year, roll_end.year + 1))

    print(years)

    ## Make list of datetimes for full test domain
    n_test = (roll_end - roll_start) // timedelta(hours=6)
    test_idxs = (np.arange(0, n_test)).astype(int)

    # Build full datetime sequence (not just dates), extended by 2 extra steps
    datetimes_test = [roll_start + timedelta(hours=6*i) for i in range(n_test + 2)]

    ## Make lists of test set days and hours
    test_days_t1 = np.array([datetimes_test[i].date() for i in test_idxs])
    test_days_t2 = np.array([datetimes_test[i+1].date() for i in test_idxs])
    test_days_t3 = np.array([datetimes_test[i+2].date() for i in test_idxs])

    print(test_days_t1[:2])

    test_hours_t1 = np.array([datetimes_test[i].hour // 6 for i in test_idxs])
    test_hours_t2 = np.array([datetimes_test[i+1].hour // 6 for i in test_idxs])
    test_hours_t3 = np.array([datetimes_test[i+2].hour // 6 for i in test_idxs])

    i=0
    day_t1 = test_days_t1[i]
    day_t2 = test_days_t2[i]
    day_t3 = test_days_t3[i]
    hour_t1 = test_hours_t1[i]
    hour_t2 = test_hours_t2[i]
    hour_t3 = test_hours_t3[i]
    day_idx = test_idxs[i]

    ## Compile GEOS-Chem fields
    chem_data_test = {}
    chem_weights = {}
    for species in species_list:
        
        gc_filled_all = xr.open_mfdataset([f"{config['data_dir']}/GEOS_Chem/{gc_version}/time_avg/4x5/{year}/{species}_filled_{year}_interp{interp_scheme_gc}.nc" for year in years],
                                            combine="by_coords",
                                            engine="netcdf4")

        gc_filled_all = gc_filled_all.sel(time=slice(roll_start, roll_end))
        gc_filled_all = gc_filled_all[species][::2]
        
        chem_data_test[species] = torch.from_numpy(gc_filled_all.values[None])

        ## Normalize GC data
        # gc_scales = 0.5*gc_filled_all.max(axis=(-2, -1)).mean(axis=0).values
        # gc_maes = np.mean(np.abs(gc_filled_all.values[:-1] - gc_filled_all.values[1:]), axis=(0, 2, 3))

        norm_stats = np.load(f"{config['output_dir']}/Weights/{run_name}_norm_stats.npz")
        gc_scales = norm_stats[f"{species}_scales"]
        gc_maes = norm_stats[f"{species}_maes"]

        gc_weights = torch.from_numpy(gc_scales / gc_maes).to(torch.device(f"cuda:{device}"))

        chem_weights[species] = gc_weights
        
        pressure_lvls = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
        for idx, lev in enumerate(pressure_lvls):
            locations[f"{species}_{str(int(lev))}"] = 0.0
            scales[f"{species}_{str(int(lev))}"] = gc_scales[idx]

    ## Compile static ERA5 fields
    static_data_test = {}
    for year in years:
        static_data_test[year] = xr.open_dataset(f"{config['data_dir']}/era5_interp/{year}/static_{year}_interp{interp_scheme}.nc", engine="netcdf4")
        
    ## Compile MERRA2 fields
    merra2_data = {}
    if met_type == "merra2":
        
        merra2_path = f"{config['data_dir']}/MERRA2/time_avg/4x5"

        for met in met_list:

            print(f"Loading {met}.", flush=True)

            merra2_filled = xr.open_mfdataset([f"{merra2_path}/{train_year}/{met}_filled_{train_year}_interp_uniform_fix.nc" for train_year in years],
                                            combine="by_coords",
                                            engine="netcdf4")
            
            merra2_filled = merra2_filled.sel(time=slice(roll_start, roll_end))
    
            if met in ["TS", "U10M", "V10M", "PSC2WET"]:
                merra2_filled = merra2_filled[f"Met_{met}"][::2]
            else:
                merra2_filled = merra2_filled[met][::2]

            print(merra2_filled.time.values, flush=True)

            if met == "SPHU":
                merra2_filled = merra2_filled / 1000
            if met == "Z":
                merra2_filled = merra2_filled * 9.81

            merra2_data[met] = torch.from_numpy(merra2_filled.values[None])

        input_batch, _ = make_batch_merra2(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                           day_idx, chem_data_test, merra2_data, static_data_test, 
                                           interp_scheme, new_static, config['data_dir'])
    else: # ERA5 

        input_batch, _ = make_batch_era5(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                         day_idx, chem_data_test, static_data_test,
                                         interp_scheme, new_static, config['data_dir'])

    if sat_nudge:

        tropomi_no2_mask_ds = xr.open_dataset(f"{config['data_dir']}/TROPOMI/tropomi_no2_mask_2019_6hourly.nc")
    
        # Convert mask latitudes to the batch's convention
        tropomi_no2_mask_ds = tropomi_no2_mask_ds.isel(lat=slice(None, None, -1))

        # Convert mask longitudes to the batch's convention, then sort.
        tropomi_no2_mask_ds = tropomi_no2_mask_ds.assign_coords(lon=(((tropomi_no2_mask_ds.lon + 180) % 360) - 180)).sortby("lon")

        tropomi_no2_mask = tropomi_no2_mask_ds["tropomi_valid_mask"].values
        print(tropomi_no2_mask.shape, flush=True)

    else:

        tropomi_no2_mask = None

    print("BEFORE TRIM: ", input_batch.atmos_vars["t"].shape)
    input_batch = trim_batch_lat(input_batch, "input")
    print("AFTER TRIM: ", input_batch.atmos_vars["t"].shape)
    
    ## Construct metadata arrays for output files
    time_dim = [datetime.combine(test_days_t3[i], 
                                 datetime.min.time()) + timedelta(hours=int(test_hours_t3[i]*6)) for i in range(len(test_days_t3))]

    if rollout_type == "naive":
        
        pred_one_step = model.forward(input_batch).surf_vars["2t"]
        with torch.inference_mode():
            pred_rollout_first = [pred.to("cpu") for pred in rollout(model, input_batch, steps=1)]
        pred_rollout_first = pred_rollout_first[0].surf_vars["2t"]
        
        print("ROLLOUT CHECK", 
              torch.allclose(
                pred_one_step.to("cpu"),
                pred_rollout_first,
                atol=1e-6
              ), flush=True)
    
        with torch.inference_mode():
            rollouts = [pred.to("cpu") for pred in rollout(model, input_batch, steps=n_test)]
    
    elif rollout_type == "free":
        
        pred_one_step = model.forward(input_batch).surf_vars["2t"]
        with torch.inference_mode():
            pred_rollout_first = [pred.to("cpu") for pred in rollout_updated(model, 
                                                                             input_batch, 
                                                                             steps=1, 
                                                                             test_years=years, 
                                                                             nudge_time=True, 
                                                                             nudge_emis=True,
                                                                             data_dir=config["data_dir"])]
        pred_rollout_first = pred_rollout_first[0].surf_vars["2t"]
        
        print("ROLLOUT CHECK", 
              torch.allclose(
                pred_one_step.to("cpu"),
                pred_rollout_first,
                atol=1e-6
              ), flush=True)
    
        with torch.inference_mode():
            rollouts = [pred.to("cpu") for pred in rollout_updated(model, 
                                                                   input_batch, 
                                                                   steps=n_test, 
                                                                   test_years=years, 
                                                                   nudge_time=False, 
                                                                   nudge_emis=False,
                                                                   data_dir=config["data_dir"])]

    elif rollout_type == "nudged_time":
        
        pred_one_step = model.forward(input_batch).surf_vars["2t"]
        pred_rollout_first = [pred.to("cpu") for pred in rollout_updated(model, 
                                                                         input_batch, 
                                                                         steps=1, 
                                                                         test_years=years, 
                                                                         nudge_time=True, 
                                                                         nudge_emis=False,
                                                                         data_dir=config["data_dir"])]
        pred_rollout_first = pred_rollout_first[0].surf_vars["2t"]
        
        print("ROLLOUT CHECK", 
              torch.allclose(
                pred_one_step.to("cpu"),
                pred_rollout_first,
                atol=1e-6
              ), flush=True)
    
        with torch.inference_mode():
            rollouts = [pred.to("cpu") for pred in rollout_updated(model, 
                                                                   input_batch, 
                                                                   steps=n_test, 
                                                                   test_years=years, 
                                                                   nudge_time=True, 
                                                                   nudge_emis=False,
                                                                   data_dir=config["data_dir"])]

    elif rollout_type == "nudged_emis":
        
        pred_one_step = model.forward(input_batch).surf_vars["2t"]
        pred_rollout_first = [pred.to("cpu") for pred in rollout_updated(model, 
                                                                         input_batch, 
                                                                         steps=1, 
                                                                         test_years=years, 
                                                                         nudge_time=False, 
                                                                         nudge_emis=True,
                                                                         data_dir=config["data_dir"])]
        pred_rollout_first = pred_rollout_first[0].surf_vars["2t"]
        
        print("ROLLOUT CHECK", 
              torch.allclose(
                pred_one_step.to("cpu"),
                pred_rollout_first,
                atol=1e-6
              ), flush=True)

        with torch.inference_mode():
            rollouts = [pred.to("cpu") for pred in rollout_updated(model, 
                                                                   input_batch, 
                                                                   steps=n_test, 
                                                                   test_years=years, 
                                                                   nudge_time=False, 
                                                                   nudge_emis=True,
                                                                   data_dir=config["data_dir"])]

    else:
        
        with torch.inference_mode():
            
            pred_one_step = model.forward(input_batch).surf_vars["2t"]
            pred_rollout_first = [pred.to("cpu") for pred in rollout_nudged(model, 
                                                                            input_batch, 
                                                                            steps=1, 
                                                                            test_years=years,
                                                                            roll_start=roll_start,
                                                                            roll_end=roll_end,
                                                                            static_data_test=static_data_test,
                                                                            chem_data_test=chem_data_test,
                                                                            met_data_test=merra2_data,
                                                                            sat_mask_test=tropomi_no2_mask,
                                                                            nudge_int=nudge_int,
                                                                            chem_nudge=chem_nudge,
                                                                            met_nudge=met_nudge,
                                                                            sat_nudge=sat_nudge,
                                                                            avg_kernel=avg_kernel,
                                                                            chem_nudge_list=chem_nudge_list,
                                                                            met_nudge_list=met_nudge_list,
                                                                            sat_nudge_list=sat_nudge_list,
                                                                            rollout_type=rollout_type,
                                                                            gc_version=gc_version,
                                                                            met_type=met_type,
                                                                            device=device,
                                                                            data_dir=config["data_dir"])]
            pred_rollout_first = pred_rollout_first[0].surf_vars["2t"]
            
            print("ROLLOUT CHECK", 
                torch.allclose(
                    pred_one_step.to("cpu"),
                    pred_rollout_first,
                    atol=1e-6
                ), flush=True)
    
        with torch.inference_mode():
            print("CHECK 12", flush=True)
            rollouts = [pred.to("cpu") for pred in rollout_nudged(model, 
                                                                  input_batch,
                                                                  steps=n_test, 
                                                                  test_years=years,
                                                                  roll_start=roll_start,
                                                                  roll_end=roll_end,
                                                                  static_data_test=static_data_test,
                                                                  chem_data_test=chem_data_test,
                                                                  met_data_test=merra2_data,
                                                                  sat_mask_test=tropomi_no2_mask,
                                                                  nudge_int=nudge_int,
                                                                  chem_nudge=chem_nudge,
                                                                  met_nudge=met_nudge,
                                                                  sat_nudge=sat_nudge,
                                                                  avg_kernel=avg_kernel,
                                                                  chem_nudge_list=chem_nudge_list,
                                                                  met_nudge_list=met_nudge_list,
                                                                  sat_nudge_list=sat_nudge_list,
                                                                  rollout_type=rollout_type,
                                                                  gc_version=gc_version,
                                                                  met_type=met_type,
                                                                  device=device,
                                                                  data_dir=config["data_dir"])]
            
        print("ROLLOUT VS ROLLOUT CHECK",
            torch.allclose(
                pred_rollout_first,  # from steps=1
                rollouts[0].surf_vars["2t"].to("cpu"),  # from steps=n_test
                atol=1e-6
            ), flush=True)
        
    if patch_size % 2 == 0:
        coords = {
            "time":time_dim,
            "lat":input_batch.metadata.lat.cpu().detach().numpy()[:-1],
            "lon":input_batch.metadata.lon.cpu().detach().numpy()
        }
    else:
        coords = {
            "time":time_dim,
            "lat":input_batch.metadata.lat.cpu().detach().numpy(),
            "lon":input_batch.metadata.lon.cpu().detach().numpy()
        }
        
    for r in rollouts:
        for var in r.surf_vars:
            rollouts_dict[var].append(r.surf_vars[var][-1, 0].cpu().detach().numpy())
        
        for var in r.atmos_vars:
            for lvl_idx, lvl in enumerate(pressure_lvls):
                rollouts_dict[f"{var}_{lvl}"].append(r.atmos_vars[var][-1, 0, lvl_idx].cpu().detach().numpy())
        
    rollouts_dict_final = {}
    
    for k in rollouts_dict:
        rollouts_dict_final[k] = (("time", "lat", "lon"), np.array(rollouts_dict[k]))
        
    ds_rollouts = xr.Dataset(rollouts_dict_final, coords=coords)
    ds_rollouts.to_netcdf(f"{config['output_dir']}/{run_name}_{roll_start:%Y%m%d}_{roll_end:%Y%m%d}_{file_suffix_dict[rollout_type]}.nc")
    ds_rollouts.close()

if __name__ == "__main__":

    ## Parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "Config file.", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    Path(config["output_dir"], "Weights").mkdir(parents=True, exist_ok=True)

    run_name = config["run_name"]
    device = config["device"]
    species_list = config["model_config"]["species_list"]
    met_type = config["model_config"]["met_type"]
    met_list = config["model_config"]["met_list"]

    patch_size = config["patch_size"] # 1
    interp_scheme = config["interp_scheme"] # "_uniform"
    interp_scheme_gc = config["interp_scheme_gc"] # "_uniform_fix2"
    new_static = config["new_static"] # True
    gc_version = config["gc_version"] # "fetch"

    ## Load model
    if config["model_config"]["transform"]:
        positive_atmos_vars = tuple(species_list)
    else:
        positive_atmos_vars = ()
    
    model_kwargs = dict(
        surf_vars=("2t", "10u", "10v", "msl"),
        static_vars=("lsm", "z", "slt",
                    "tod_cos", "tod_sin",
                    "dow_cos", "dow_sin",
                    "doy_cos", "doy_sin",
                    "nh3_emis", "co_emis", "nox_emis", "so2_emis"),
        atmos_vars=("z", "u", "v", "t", "q") + tuple(species_list),
        patch_size=patch_size,
        positive_atmos_vars=positive_atmos_vars,
    )

    if config["model_config"]["separate_perc"]:
        model_kwargs["separate_perceiver"] = tuple(species_list)

    if config["model_config"]["level_cond"]:
        model_kwargs["level_condition"] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

    model_train = AuroraPretrained(**model_kwargs)

    model_train = model_train.to(f"cuda:{device}")
    model_train.load_state_dict(torch.load(f"{config['output_dir']}/Weights/{run_name}_best_weights.pth",
                                           map_location=f"cuda:{device}"))
    model_train.eval()  

    ## Run one-step and/or rollout evaluation
    if config["one_step"]["one_step_bool"]:
        one_step_eval(run_name,
                      model_train,
                      datetime.strptime(config["start"], "%Y-%m-%d %H"),
                      datetime.strptime(config["end"], "%Y-%m-%d %H"),
                      species_list,
                      interp_scheme_gc,
                      met_type,
                      met_list,
                      device=device)
    
    if config["rollout"]["rollout_bool"]:
        rollout_eval(run_name,
                     model_train,
                     config["rollout"]["rollout_type"],
                     datetime.strptime(config["start"], "%Y-%m-%d %H"),
                     datetime.strptime(config["end"], "%Y-%m-%d %H"),
                     config["rollout"]["nudge_int"],
                     config["rollout"]["met_nudge_fields"],
                     config["rollout"]["chem_nudge_fields"],
                     config["rollout"]["sat_nudge_fields"],
                     species_list,
                     interp_scheme_gc,
                     met_type,
                     met_list,
                     device=device) 