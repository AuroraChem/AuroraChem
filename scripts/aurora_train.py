import numpy as np
import xarray as xr
import yaml

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from aurora import Batch, Metadata
from aurora.normalisation import locations, scales, normalise_atmos_var, normalise_surf_var
from aurora import AuroraPretrained

# import matplotlib.pyplot as plt

from pathlib import Path
import time
import argparse

from datetime import date, datetime, timedelta
from datetime import time as time_dt
from dateutil.relativedelta import relativedelta

from src.data.batch import make_batch_era5, make_batch_merra2, trim_batch_lat
from src.training.loss_funcs import AuroraL1Loss, AuroraL1Loss_NoWeight, AuroraL1Loss_PolarMask
from src.training.scheduler import make_lr_lambda, lr_lambda, lr_lambda_nowarm, EarlyStopping
from src.utils.utils import memcheck, load_config

if __name__ == "__main__":

    start_full_run = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config_name = args.config

    config = load_config(args.config)
    Path(config["output_dir"], "Weights").mkdir(parents=True, exist_ok=True)
    
    run_name = config["run_name"]
    gpu_num = config["gpu"]

    # Run config
    save_weight_flag = config["save_weights"]
    new_static = config["new_static"]
    interp_scheme = config["interp_scheme"]
    interp_gc = config["interp_gc"]
    interp_scheme_gc = config["interp_scheme_gc"]
    gc_version = config["gc_version"]
    met_type = config["met_type"]
    adamw_flag = config["adamw"]

    # Field config
    phys_only_flag = config["phys_only_flag"]
    species_list = config["species_list"]
    met_list = config["met_list"]

    # Architecture config
    seed = config["seed"]
    patch_size = config["patch_size"]
    transform_flag = config["transform"]
    separate_perc_flag = config["separate_perc"]
    level_cond_flag = config["level_cond"]

    # LR schedule config
    early_stopping_flag = config["early_stopping"]
    patience = config["patience"]
    lr_decay_flag = config["lr_decay_flag"]
    lr_warm_flag = config["lr_warm_flag"]

    # Loss function config
    polar_mask_flag = config["polar_mask_flag"]

    # Training config
    train_years = config["training"]["years"]
    n_train = config["training"]["samples"]
    n_epochs = config["training"]["epochs"]
    chem_lr = config["training"]["chem_lr"]
    met_lr = config["training"]["met_lr"]
    weighting_flag = config["training"]["chem_weight"]

    # Validation config
    n_valid = config["validation"]["samples"]

    # Testing config
    test_years = config["testing"]["years"]
    n_test = config["testing"]["samples"]

    ## Use specified GPU
    torch.cuda.set_device(gpu_num)
    
    ## Model Set-Up
    if transform_flag:
        positive_atmos_vars = tuple(species_list)
    else:
        positive_atmos_vars = ()
    
    if new_static:
        if level_cond_flag:
            if separate_perc_flag:
                model_train = AuroraPretrained(
                    surf_vars=("2t", "10u", "10v", "msl"),
                    static_vars=("lsm", "z", "slt", 
                                 "tod_cos", "tod_sin",
                                 "dow_cos", "dow_sin", 
                                 "doy_cos", "doy_sin",
                                 "nh3_emis", "co_emis", "nox_emis", "so2_emis"),
                    atmos_vars=("z", "u", "v", "t", "q") + tuple(species_list),
                    patch_size=patch_size,
                    positive_atmos_vars=positive_atmos_vars,
                    separate_perceiver=tuple(species_list),
                    level_condition=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                )
            else: 
                model_train = AuroraPretrained(
                    surf_vars=("2t", "10u", "10v", "msl"),
                    static_vars=("lsm", "z", "slt", 
                                 "tod_cos", "tod_sin",
                                 "dow_cos", "dow_sin", 
                                 "doy_cos", "doy_sin",
                                 "nh3_emis", "co_emis", "nox_emis", "so2_emis"),
                    atmos_vars=("z", "u", "v", "t", "q") + tuple(species_list),
                    patch_size=patch_size,
                    positive_atmos_vars=positive_atmos_vars,
                    level_condition=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                )
        else:
            if separate_perc_flag:
                model_train = AuroraPretrained(
                    surf_vars=("2t", "10u", "10v", "msl"),
                    static_vars=("lsm", "z", "slt", 
                                 "tod_cos", "tod_sin",
                                 "dow_cos", "dow_sin", 
                                 "doy_cos", "doy_sin",
                                 "nh3_emis", "co_emis", "nox_emis", "so2_emis"),
                    atmos_vars=("z", "u", "v", "t", "q") + tuple(species_list),
                    patch_size=patch_size,
                    positive_atmos_vars=positive_atmos_vars,
                    separate_perceiver=tuple(species_list),
                )
            else:
                model_train = AuroraPretrained(
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
    else:
        model_train = AuroraPretrained(
            surf_vars=("2t", "10u", "10v", "msl"),
            static_vars=("lsm", "z", "slt"),
            atmos_vars=("z", "u", "v", "t", "q") + tuple(species_list),
            positive_atmos_vars=positive_atmos_vars,
        )

    ## Model Loading  
    model_train.load_checkpoint(strict=False)
    model_train = model_train.cuda()
    model_train.train()
    model_train.configure_activation_checkpointing()
    print("MODEL LOADED.", flush=True)

    ## Load MERRA2 data
    if met_type == "merra2":
        
        merra2_data = {}
        merra2_path = f"{config['data_dir']}/MERRA2/time_avg/4x5"

        for met in met_list:

            print(f"Loading {met}.", flush=True)

            merra2_filled = xr.open_mfdataset([f"{merra2_path}/{train_year}/{met}_filled_{train_year}_interp_uniform_fix.nc" for train_year in train_years],
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

    ## Load GEOS-Chem data
    chem_data = {}
    chem_weights = {}
    gc_path = f"{config['data_dir']}/GEOS_Chem/{gc_version}/time_avg/4x5"

    ## TODO: Check
    norm_stats = {}

    ## Compile GEOS-Chem fields for training
    gc_neg_time_idxs = []
    for species in species_list:

        print(f"Loading {species}.", flush=True)

        #TODO: make combined gc_filled for all years
        if interp_gc:
            gc_filled = xr.open_mfdataset([f"{gc_path}/{train_year}/{species}_filled_{train_year}_interp{interp_scheme_gc}.nc" for train_year in train_years], 
                                               combine="by_coords",
                                               engine="netcdf4")
        else:
            gc_filled = xr.open_mfdataset([f"{gc_path}/{train_year}/{species}_filled_{train_year}.nc" for train_year in train_years], 
                                               combine="by_coords",
                                               engine="netcdf4")
        
        gc_filled = gc_filled[species][::2]
        print(gc_filled.shape, flush=True)

        ## Check for erroneus negative GEOS-Chem concentrations
        gc_filled_values = gc_filled.values
        bad_mask = (gc_filled_values.min(axis=(-2, -1)) < 0).any(axis=1)      # (T,) any level negative
        gc_bad_times = list(np.where(bad_mask)[0])
        print(f"{species}: {len(gc_bad_times)} contaminated timesteps of {gc_filled_values.shape[0]}", flush=True)
          
        gc_neg_times = list(np.where(gc_filled[:, 3].mean(axis=(-2, -1)) < 0)[0])
        for idx in gc_neg_times:
            if idx >= 2:
                gc_neg_time_idxs.append(idx)
                gc_neg_time_idxs.append(idx - 1)
                gc_neg_time_idxs.append(idx - 2)
        print(gc_neg_time_idxs)
        
        gc_scales = 0.5*gc_filled.max(axis=(-2, -1)).mean(axis=0).values

        # Old, incorrect gc_maes
        # gc_maes = np.mean(np.abs(gc_filled[:-1] - gc_filled[1:, 0, 0]), axis=(0, 2, 3)).values

        bad = np.zeros(gc_filled_values.shape[0], dtype=bool)
        bad[gc_bad_times] = True
        valid_pairs = ~(bad[:-1] | bad[1:])               # drop any pair touching a bad step
        
        gc_diffs = np.diff(gc_filled.values, axis=0)
        gc_maes = np.abs(gc_diffs[valid_pairs]).mean(axis=(0, 2, 3))

        gc_weights = torch.from_numpy(gc_scales / gc_maes).to(torch.device(f"cuda:{gpu_num}"))

        norm_stats[f"{species}_scales"] = gc_scales
        norm_stats[f"{species}_maes"] = gc_maes
        norm_stats[f"{species}_weights"] = gc_scales / gc_maes
        
        ## Normalize GC data
        pressure_lvls = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
        for idx, lev in enumerate(pressure_lvls):
            locations[f"{species}_{str(int(lev))}"] = 0.0
            scales[f"{species}_{str(int(lev))}"] = gc_scales[idx]

        chem_data[species] = torch.from_numpy(gc_filled.values[None])
        chem_weights[species] = gc_weights  

    np.savez(f"{config['output_dir']}/Weights/{run_name}_norm_stats.npz",
                pressure_lvls=np.array(pressure_lvls),
                train_years=np.array(train_years),
                species=np.array(species_list),
                **norm_stats)
    
    # assert (gc_maes > 0).all() and gc_maes.max() / gc_maes.min() < 100, f"{species}: suspicious MAE profile {gc_maes}"

    print(f"NEGATIVE CONC IDXS: {gc_neg_time_idxs}")
    
    ## Compile static ERA5 fields for training
    static_data = {}
    for year in train_years:
        static_data[year] = xr.open_dataset(f"{config['data_dir']}/era5_interp/{year}/static_{year}_interp{interp_scheme}.nc", engine="netcdf4")
    
    ######## Construct arrays for indexing data ########
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Select validation and test indices
    n_total_examples = gc_filled.time.shape[0]

    all_candidates = np.arange(int(n_total_examples - 3))
    available = np.setdiff1d(all_candidates, gc_neg_time_idxs)
    valid_idxs = np.random.choice(available, size=n_valid, replace=False)
    
    test_idxs = (np.arange(0, n_test)).astype(int)

    ## Make list of datetimes for full training domain
    start_date = date(train_years[0], 1, 1)
    end_date = start_date + relativedelta(years=len(train_years))
    days_train = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]
    days_train = np.repeat(days_train, 4)

    ## Make lists of validation set days and hours
    valid_days_t1 = days_train[valid_idxs]
    valid_days_t2 = days_train[valid_idxs + 1]
    valid_days_t3 = days_train[valid_idxs + 2]
    
    valid_hours_t1 = valid_idxs % 4
    valid_hours_t2 = (valid_idxs + 1) % 4
    valid_hours_t3 = (valid_idxs + 2) % 4

    ## Make list of datetimes for full test domain
    start_date = date(test_years[0], 1, 1)
    end_date = start_date + relativedelta(years=len(test_years))
    days_test = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]
    days_test = np.repeat(days_test, 4)

    ## Make lists of test set days and hours
    test_days_t1 = days_test[test_idxs]
    test_days_t2 = days_test[test_idxs + 1]
    test_days_t3 = days_test[test_idxs + 2]
    
    test_hours_t1 = test_idxs % 4
    test_hours_t2 = (test_idxs + 1) % 4
    test_hours_t3 = (test_idxs + 2) % 4

    if phys_only_flag:
        chem_data = {}
        chem_weights = {}
        species_list = []

    ## Set model options
    params_chem = [p for n, p in model_train.named_parameters() if any(s in n for s in species_list)]
    params_pretrained = [p for n, p in model_train.named_parameters() if not any(s in n for s in species_list)]

    print(f"CHEM PARAMS: {len(params_chem)}")
    print(f"OTHER PARAMS: {len(params_pretrained)}")

    if adamw_flag:
        optimiser = AdamW([
            {"params": params_chem, "lr": chem_lr},
            {"params": params_pretrained, "lr": met_lr}],
            lr=met_lr)
    else:
        optimiser = Adam([
            {"params": params_chem, "lr": chem_lr},
            {"params": params_pretrained, "lr": met_lr}],
            lr=met_lr)
    
    if lr_decay_flag:
        if lr_warm_flag:
            lr_lambda_decay = make_lr_lambda(n_train, n_epochs)
            scheduler = LambdaLR(optimiser, lr_lambda=lr_lambda_decay)
        else:
            lr_lambda_decay = make_lr_lambda(n_train, n_epochs, warmup_ratio=1e-6)
            scheduler = LambdaLR(optimiser, lr_lambda=lr_lambda_decay)
    else:
        if lr_warm_flag:
            scheduler = LambdaLR(optimiser, lr_lambda=lr_lambda)
        else:
            scheduler = LambdaLR(optimiser, lr_lambda=lr_lambda_nowarm)

    if weighting_flag:
        if polar_mask_flag:
            loss_func = AuroraL1Loss_PolarMask()
        else:
            loss_func = AuroraL1Loss()
    else:
        loss_func = AuroraL1Loss_NoWeight()

    ################ TRAINING AND VALIDATION ################ 
    
    train_metrics, valid_metrics = [], []

    ## Early Stopping Set-Up
    if early_stopping_flag:
        delta = 0.002  # minimum change in the monitored metric
        best_val_loss = float("inf")  # best validation loss to compare against
        no_improvement_count = 0  # count of epochs with no improvement
        early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

    print("STARTING TRAINING LOOP.", flush=True)

    step_count = 0
    for epoch in range(n_epochs):

        available = np.setdiff1d(available, valid_idxs)
        train_idxs = np.random.choice(available, size=n_train, replace=False)
    
        print("TRAIN IDXS: ", train_idxs)
        
        train_days_t1 = days_train[train_idxs]
        train_days_t2 = days_train[train_idxs + 1]
        train_days_t3 = days_train[train_idxs + 2]
    
        train_hours_t1 = train_idxs % 4
        train_hours_t2 = (train_idxs + 1) % 4
        train_hours_t3 = (train_idxs + 2) % 4
        
        memcheck()
    
        start = time.time()       
    
        ## Training Loop
        temp_train_metrics = []
        for i, day_idx in enumerate(train_idxs):

            step_count += 1
    
            start_train = time.time()
    
            day_t1 = train_days_t1[i]
            day_t2 = train_days_t2[i]
            day_t3 = train_days_t3[i]
            hour_t1 = train_hours_t1[i]
            hour_t2 = train_hours_t2[i]
            hour_t3 = train_hours_t3[i]
    
            start_batching = time.time()
            if met_type == "merra2":
                input_batch, true_batch = make_batch_merra2(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                                            day_idx, chem_data, merra2_data, 
                                                            static_data, interp_scheme, new_static, config["data_dir"])
            else:
                input_batch, true_batch = make_batch_era5(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                                          day_idx, chem_data, 
                                                          static_data, interp_scheme, new_static, config["data_dir"])
            batch_time = time.time() - start_batching
    
            input_batch = trim_batch_lat(input_batch, "input")
            true_batch = trim_batch_lat(true_batch, "true")
    
            input_batch = input_batch.to(torch.device(f"cuda:{gpu_num}"))
            true_batch = true_batch.to(torch.device(f"cuda:{gpu_num}"))
            
            model_train.train()
            optimiser.zero_grad()
            
            start_forward = time.time()
            pred = model_train.forward(input_batch)
            forward_time = time.time() - start_forward

            # print(pred.metadata.time)
            # print(pred.atmos_vars["t"].shape)
            # exit()
            
            loss = loss_func(pred, true_batch, chem_weights, patch_size)
            start_loss = time.time()
            loss.backward()
            loss_time = time.time() - start_loss
            optimiser.step()
            scheduler.step()
            temp_train_metrics.append(loss.item())
    
            print(f"TRAIN BATCH: {len(temp_train_metrics)} / {n_train} for {np.round(time.time() - start_train, 2)} seconds (BATCHING {np.round(batch_time, 2)}s, TRAIN {np.round(forward_time, 2)}s, LOSS {np.round(loss_time, 2)}s). ", flush=True)
    
        train_metrics.append(np.array(temp_train_metrics).mean())
    
    
        ## Validation Loop
        temp_valid_metrics = []
        with torch.no_grad():
            for i, day_idx in enumerate(valid_idxs):
    
                start_valid = time.time()
        
                day_t1 = valid_days_t1[i]
                day_t2 = valid_days_t2[i]
                day_t3 = valid_days_t3[i]
                hour_t1 = valid_hours_t1[i]
                hour_t2 = valid_hours_t2[i]
                hour_t3 = valid_hours_t3[i]
    
                start_batch = time.time()
                if met_type == "merra2":
                    input_batch, true_batch = make_batch_merra2(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                                                day_idx, chem_data, merra2_data, 
                                                                static_data, interp_scheme, new_static, config["data_dir"])
                else:
                    input_batch, true_batch = make_batch_era5(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                                                            day_idx, chem_data, 
                                                            static_data, interp_scheme, new_static, config["data_dir"])
                batch_time = time.time() - start_batch
    
                input_batch = trim_batch_lat(input_batch, "input")
                true_batch = trim_batch_lat(true_batch, "true")            
                
                input_batch = input_batch.to(torch.device(f"cuda:{gpu_num}"))
                true_batch = true_batch.to(torch.device(f"cuda:{gpu_num}"))
                
                model_train.eval()
                
                start_forward = time.time()
                pred = model_train.forward(input_batch)
                forward_time = time.time() - start_forward
                
                true_batch = true_batch.to(torch.device(f"cuda:{gpu_num}"))
                loss = loss_func(pred, true_batch, chem_weights, patch_size)
                temp_valid_metrics.append(loss.item())
    
                print(f"VALID BATCH: {len(temp_valid_metrics)} / {n_valid} for {time.time() - start_valid} seconds (BATCHING {batch_time}s, TRAIN {forward_time}s). ", flush=True)
    
        valid_metrics.append(np.array(temp_valid_metrics).mean())
    
        current_lrs = [group['lr'] for group in optimiser.param_groups]
    
        print(
            f"EPOCH {epoch}, "
            f"Train Loss: {train_metrics[-1]:.4f}, "
            f"Valid Loss: {valid_metrics[-1]:.4f}, "
            f"LRs: {current_lrs}, "
            f"Time: {time.time() - start:.4f}s\n",
            flush=True
        )
        if config["early_stopping"]:
            early_stopping.check_early_stop(valid_metrics[-1], model_train)
            if early_stopping.stop_training:
                print(f"Early stopping at Epoch {epoch}.")
                model_train.load_state_dict(early_stopping.best_state)
                break

    ## Save model weights
    if save_weight_flag:
        if early_stopping_flag:
            torch.save(model_train.state_dict(), f"{config['output_dir']}/Weights/{run_name}_best_weights.pth")
        else:
            torch.save(model_train.state_dict(), f"{config['output_dir']}/Weights/{run_name}_final_weights.pth")
        print(f"WEIGHTS {run_name} SAVED.", flush=True)

    ## Save loss curves
    np.savez(f"{config['output_dir']}/Weights/{run_name}_loss_metrics.npz",
                train_metrics=np.array(train_metrics),
                valid_metrics=np.array(valid_metrics))
    print(f"LOSS METRICS {run_name} SAVED.", flush=True)