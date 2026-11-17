"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses 
import datetime as dt
import time
from typing import Generator

import numpy as np
import xarray as xr
import torch

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from pathlib import Path

from aurora.batch import Batch, Metadata
from aurora.model.aurora import Aurora
from aurora.normalisation import locations, scales, normalise_atmos_var, normalise_surf_var

__all__ = ["rollout_updated", "rollout_time", "rollout"]

def memcheck():
    """Print GPU memory usage statistics."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.1f} MiB")
            print(f"  Reserved:  {torch.cuda.memory_reserved(i) / 1024 ** 2:.1f} MiB")
            print(f"  Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024 ** 2:.1f} MiB")
            print(f"  Max Reserved:  {torch.cuda.max_memory_reserved(i) / 1024 ** 2:.1f} MiB") 
    else:
        print("CUDA not available")

def compute_layer_thickness(levels):
    """Pressure-layer thickness (hPa) for each level, via midpoint boundaries.
    levels: descending list, e.g. [1000, 925, ..., 50]
    """
    levels = np.array(levels, dtype=float)
    edges = np.zeros(len(levels) + 1)
    edges[1:-1] = (levels[:-1] + levels[1:]) / 2      # midpoints between adjacent levels
    edges[0] = levels[0] + (levels[0] - edges[1])      # bottom: extrapolate below 1000 hPa
    edges[-1] = max(levels[-1] - (edges[-2] - levels[-1]), 0.0)  # top: extrapolate above 50 hPa, floor at 0
    return np.abs(np.diff(edges))  # (n_levels,) thickness per layer, hPa

def rollout_updated(model: Aurora, 
                    batch: Batch, 
                    steps: int, 
                    test_years: list,
                    nudge_time,
                    nudge_emis,
                    data_dir: str) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions. Updates with correct time and emissions fields.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)
    
    ## Construct TIME fields
    tod = np.array([0, 6, 12, 18])
    tod_norm = tod / 24
    tod_mat = np.array([np.tile(t, [45,72]) for t in tod_norm])
    tod_cos = np.cos(2*np.pi*tod_mat)
    tod_sin = np.sin(2*np.pi*tod_mat)

    dow = np.arange(7)
    dow_norm = dow / 7
    dow_mat = np.array([np.tile(d, [45,72]) for d in dow_norm])
    dow_cos = np.cos(2*np.pi*dow_mat)
    dow_sin = np.sin(2*np.pi*dow_mat)

    doy = np.arange(365)
    doy_norm = doy / 365
    doy_mat = np.array([np.tile(d, [45,72]) for d in doy_norm])
    doy_cos = np.cos(2*np.pi*doy_mat)
    doy_sin = np.sin(2*np.pi*doy_mat)

    time_var_list = ["tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin"]
    for var in time_var_list:
        locations[var] = 0.0
        scales[var] = 1.0

    ## Load EMISSIONS fields
    nh3_emis_dict = {}
    co_emis_dict = {}
    nox_emis_dict = {}
    so2_emis_dict = {}
    for year in test_years:
        nh3_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/ammonia_{year}.nc", decode_times=False)
        co_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/carbon-monoxide_{year}.nc", decode_times=False)
        nox_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/nitrogen-oxides_{year}.nc", decode_times=False)
        so2_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/sulphur-dioxide_{year}.nc", decode_times=False)

    emis_var_list = ["nh3_emis", "co_emis", "nox_emis", "so2_emis"]
    for var in emis_var_list:
        locations[var] = 0.0
        scales[var] = 1.0

    start = time.time()
    for _ in range(steps):
        start_step=time.time()
        pred = model.forward(batch)

        yield pred

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

        if nudge_time:
            ## update TIME static fields
            hour_t2 = np.int_(pred.metadata.time[1].hour / 6)
            day_t2 = pred.metadata.time[1].date()
            day_idx = pred.metadata.time[1].timetuple().tm_yday - 1
            print(day_idx)

            batch.static_vars["tod_cos"] = torch.from_numpy(tod_cos[hour_t2])
            batch.static_vars["tod_sin"] = torch.from_numpy(tod_sin[hour_t2])
            batch.static_vars["dow_cos"] = torch.from_numpy(dow_cos[(day_t2.weekday() + 1) % 7])
            batch.static_vars["dow_sin"] = torch.from_numpy(dow_sin[(day_t2.weekday() + 1) % 7])
            batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx + 1) % 365])
            batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx + 1) % 365])

        if nudge_emis:
            # update EMISSIONS static fields
            month = day_t2.month - 1
            year = day_t2.year

            batch.static_vars["nh3_emis"] = torch.from_numpy(nh3_emis_dict[year]["total_anthro"][month].data / nh3_emis_dict[year]["total_anthro"][month].data.max())
            batch.static_vars["co_emis"] = torch.from_numpy(co_emis_dict[year]["total_anthro"][month].data / co_emis_dict[year]["total_anthro"][month].data.max())
            batch.static_vars["nox_emis"] = torch.from_numpy(nox_emis_dict[year]["total_anthro"][month].data / nox_emis_dict[year]["total_anthro"][month].data.max())
            batch.static_vars["so2_emis"] = torch.from_numpy(so2_emis_dict[year]["total_anthro"][month].data / so2_emis_dict[year]["total_anthro"][month].data.max())

        print(f"STEP TIME: {np.round(time.time() - start_step, 2)}, {pred.metadata.time}", flush=True)

    print("ROLL STEP TIME: ", time.time() - start, flush=True)

def rollout_emissions(model: Aurora, 
                      batch: Batch, 
                      steps: int, 
                      test_year: int,
                      data_dir: str) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions. Updates with correct emissions fields.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    ## Load EMISSIONS fields
    nh3_emis = xr.open_dataset(f"{data_dir}/emissions/ammonia_{test_year}.nc", decode_times=False)
    co_emis = xr.open_dataset(f"{data_dir}/emissions/carbon-monoxide_{test_year}.nc", decode_times=False)
    nox_emis = xr.open_dataset(f"{data_dir}/emissions/nitrogen-oxides_{test_year}.nc", decode_times=False)
    so2_emis = xr.open_dataset(f"{data_dir}/emissions/sulphur-dioxide_{test_year}.nc", decode_times=False)

    emis_var_list = ["nh3_emis", "co_emis", "nox_emis", "so2_emis"]
    for var in emis_var_list:
        locations[var] = 0.0
        scales[var] = 1.0

    start = time.time()
    for _ in range(steps):
        pred = model.forward(batch)

        yield pred

        print(pred.metadata.time, flush=True)

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

        ## update TIME static fields
        hour_t2 = np.int_(pred.metadata.time[1].hour / 6)
        day_t2 = pred.metadata.time[1].date()
        day_idx = pred.metadata.time[1].timetuple().tm_yday - 1

        # update EMISSIONS static fields
        month = day_t2.month - 1
        year = day_t2.year

        batch.static_vars["nh3_emis"] = torch.from_numpy(nh3_emis["total_anthro"][month].data / nh3_emis["total_anthro"][month].data.max())
        batch.static_vars["co_emis"] = torch.from_numpy(co_emis["total_anthro"][month].data / co_emis["total_anthro"][month].data.max())
        batch.static_vars["nox_emis"] = torch.from_numpy(nox_emis["total_anthro"][month].data / nox_emis["total_anthro"][month].data.max())
        batch.static_vars["so2_emis"] = torch.from_numpy(so2_emis["total_anthro"][month].data / so2_emis["total_anthro"][month].data.max())

    print("ROLL STEP TIME: ", time.time() - start, flush=True)

def rollout_time(model: Aurora, batch: Batch, steps: int, test_year: int) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions. Updates with correct time fields.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)
    
    ## Construct TIME fields
    tod = np.array([0, 6, 12, 18])
    tod_norm = tod / 24
    tod_mat = np.array([np.tile(t, [45,72]) for t in tod_norm])
    tod_cos = np.cos(2*np.pi*tod_mat)
    tod_sin = np.sin(2*np.pi*tod_mat)

    dow = np.arange(7)
    dow_norm = dow / 7
    dow_mat = np.array([np.tile(d, [45,72]) for d in dow_norm])
    dow_cos = np.cos(2*np.pi*dow_mat)
    dow_sin = np.sin(2*np.pi*dow_mat)

    doy = np.arange(365)
    doy_norm = doy / 365
    doy_mat = np.array([np.tile(d, [45,72]) for d in doy_norm])
    doy_cos = np.cos(2*np.pi*doy_mat)
    doy_sin = np.sin(2*np.pi*doy_mat)

    time_var_list = ["tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin"]
    for var in time_var_list:
        locations[var] = 0.0
        scales[var] = 1.0

    start = time.time()
    for _ in range(steps):
        pred = model.forward(batch)

        yield pred

        print(pred.metadata.time, flush=True)

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

        ## update TIME static fields
        hour_t2 = np.int_(pred.metadata.time[1].hour / 6)
        day_t2 = pred.metadata.time[1].date()
        day_idx = pred.metadata.time[1].timetuple().tm_yday - 1

        batch.static_vars["tod_cos"] = torch.from_numpy(tod_cos[hour_t2])
        batch.static_vars["tod_sin"] = torch.from_numpy(tod_sin[hour_t2])
        batch.static_vars["dow_cos"] = torch.from_numpy(dow_cos[(day_t2.weekday() + 1) % 7])
        batch.static_vars["dow_sin"] = torch.from_numpy(dow_sin[(day_t2.weekday() + 1) % 7])
        batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx + 1) % 365])
        batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx + 1) % 365])

    print("ROLL STEP TIME: ", time.time() - start, flush=True)

def rollout_nudged(model: Aurora, 
                   batch: Batch, 
                   steps: int, 
                   test_years: list, 
                   roll_start,
                   roll_end,
                   static_data_test,
                   chem_data_test,
                   met_data_test,
                   sat_mask_test,
                   nudge_int: int, 
                   chem_nudge: bool,
                   met_nudge: bool,
                   sat_nudge: bool,
                   avg_kernel: bool,
                   chem_nudge_list,
                   met_nudge_list,
                   sat_nudge_list,
                   rollout_type: str,
                   gc_version: str, 
                   met_type: str,
                   device: int,
                   data_dir: str) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions. Updates with true fields every N steps.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    model.eval()
    # We will need to concatenate data, so ensure that everything is already of the right form.
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    ## TODO
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

    ## Set initial day, hour, and day index
    day_t1 = test_days_t1[0]
    day_t2 = test_days_t2[0]
    day_t3 = test_days_t3[0]
    hour_t1 = test_hours_t1[0]
    hour_t2 = test_hours_t2[0]
    hour_t3 = test_hours_t3[0]
    day_idx = test_idxs[0]

    met_vars = ("t", "q", "u", "v", "z")
    species_list = [var for var in batch.atmos_vars if var not in met_vars]
    
    ## Construct TIME fields
    tod = np.array([0, 6, 12, 18])
    tod_norm = tod / 24
    tod_mat = np.array([np.tile(t, [45,72]) for t in tod_norm])
    tod_cos = np.cos(2*np.pi*tod_mat)
    tod_sin = np.sin(2*np.pi*tod_mat)

    dow = np.arange(7)
    dow_norm = dow / 7
    dow_mat = np.array([np.tile(d, [45,72]) for d in dow_norm])
    dow_cos = np.cos(2*np.pi*dow_mat)
    dow_sin = np.sin(2*np.pi*dow_mat)

    doy = np.arange(365)
    doy_norm = doy / 365
    doy_mat = np.array([np.tile(d, [45,72]) for d in doy_norm])
    doy_cos = np.cos(2*np.pi*doy_mat)
    doy_sin = np.sin(2*np.pi*doy_mat)

    time_var_list = ["tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin"]
    for var in time_var_list:
        locations[var] = 0.0
        scales[var] = 1.0

    ## Load EMISSIONS fields
    nh3_emis_dict = {}
    co_emis_dict = {}
    nox_emis_dict = {}
    so2_emis_dict = {}
    for year in test_years:
        nh3_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/ammonia_{year}.nc", decode_times=False)
        co_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/carbon-monoxide_{year}.nc", decode_times=False)
        nox_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/nitrogen-oxides_{year}.nc", decode_times=False)
        so2_emis_dict[year] = xr.open_dataset(f"{data_dir}/emissions/sulphur-dioxide_{year}.nc", decode_times=False)
    
    emis_var_list = ["nh3_emis", "co_emis", "nox_emis", "so2_emis"]
    for var in emis_var_list:
        locations[var] = 0.0
        scales[var] = 1.0

    device_name = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    ## Compute Aurora layer thicknesses for the TROPOMI averaging kernel nudging
    aurora_levs = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    layer_dp = compute_layer_thickness(aurora_levs)  # shape (13,)
    dp_tensor = torch.from_numpy(layer_dp).to(device=device_name, dtype=dtype).view(1, 1, -1, 1, 1)

    chem_var_list = ["OH", "NO", "NO2", "Cl", "O3", "CO", "C14O"]
    met_var_list = ["t", "q", "u", "v", "z"]

    ## TODO: matching doy_idx to training doy_idx - this is INCORRECT!  
    year_start = datetime(roll_start.year, 1, 1, 0, 0)
    day_idx_start = int((roll_start - year_start).total_seconds() // (6 * 3600))    

    start = time.time()
    step_count = 0
    for _ in range(steps):

        start_step = time.time()
        step_count+=1
        
        with torch.inference_mode():
            pred = model.forward(batch)

        yield pred

        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },  
        )

        ## update TIME static fields
        day_t1 = pred.metadata.time[0].date()
        day_t2 = pred.metadata.time[1].date()

        hour_t1 = np.int_(pred.metadata.time[0].hour / 6)
        hour_t2 = np.int_(pred.metadata.time[1].hour / 6)
        
        ## TODO: should day_idx be the prediction field's day or the last input field's day?
        day_idx = pred.metadata.time[1].timetuple().tm_yday - 1

        ## TODO: matching doy_idx to training doy_idx - this is INCORRECT!  
        doy_idx = (day_idx_start + 1 + step_count) % 365

        print(f"DOY IDX: {doy_idx}", flush=True)

        expected = int((pred.metadata.time[1] - year_start).total_seconds() // (6 * 3600))
        assert expected == day_idx_start + 1 + step_count, \
            f"doy counter drift: metadata says {expected}, counter says {day_idx_start + 1 + step_count}"
        
        test_start = datetime(test_years[0], 1, 1, 0, 0)
        # six_hour_idx = int((pred.metadata.time[0] - test_start).total_seconds() // (6*3600))
        six_hour_idx = int((pred.metadata.time[0] - roll_start).total_seconds() // (6*3600))
        
        # six_hour_idx = (pred.metadata.time[0].timetuple().tm_yday - 1)*4 + hour_t1
        print("DAY IDX CHECK:", day_idx, hour_t1, six_hour_idx, flush=True)

        batch.static_vars["tod_cos"] = torch.from_numpy(tod_cos[hour_t2]).to(device=device_name, dtype=dtype)
        batch.static_vars["tod_sin"] = torch.from_numpy(tod_sin[hour_t2]).to(device=device_name, dtype=dtype)
        batch.static_vars["dow_cos"] = torch.from_numpy(dow_cos[(day_t2.weekday() + 1) % 7]).to(device=device_name, dtype=dtype)
        batch.static_vars["dow_sin"] = torch.from_numpy(dow_sin[(day_t2.weekday() + 1) % 7]).to(device=device_name, dtype=dtype)
        # batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx + 1) % 365]).to(device=device_name, dtype=dtype)
        # batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx + 1) % 365]).to(device=device_name, dtype=dtype)
        batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx) % 365]).to(device=device_name, dtype=dtype)
        batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx) % 365]).to(device=device_name, dtype=dtype)

        # update EMISSIONS static fields
        month = day_t2.month - 1
        year = day_t2.year

        batch.static_vars["nh3_emis"] = torch.from_numpy(nh3_emis_dict[year]["total_anthro"][month].data / nh3_emis_dict[year]["total_anthro"][month].data.max()).to(device=device_name, dtype=dtype)
        batch.static_vars["co_emis"] = torch.from_numpy(co_emis_dict[year]["total_anthro"][month].data / co_emis_dict[year]["total_anthro"][month].data.max()).to(device=device_name, dtype=dtype)
        batch.static_vars["nox_emis"] = torch.from_numpy(nox_emis_dict[year]["total_anthro"][month].data / nox_emis_dict[year]["total_anthro"][month].data.max()).to(device=device_name, dtype=dtype)
        batch.static_vars["so2_emis"] = torch.from_numpy(so2_emis_dict[year]["total_anthro"][month].data / so2_emis_dict[year]["total_anthro"][month].data.max()).to(device=device_name, dtype=dtype)

        if (step_count % nudge_int) == 0:

            ## update SAT fields
            if sat_nudge:

                print("SAT NUDGE CHECK", flush=True)

                mask_t1 = torch.from_numpy(sat_mask_test[six_hour_idx]).to(device=device_name)      # (lat, lon) bool
                mask_t2 = torch.from_numpy(sat_mask_test[six_hour_idx + 1]).to(device=device_name)  # (lat, lon) bool
                mask = torch.stack([mask_t1, mask_t2])[:, None, None, :, :]  # (2, 1, 1, lat, lon) — broadcasts over batch, level

                ## Open surface and atmos ERA5 datasets for t0
                download_path = Path(f"{data_dir}/era5_interp/{year}")

                surf_vars_ds_t1 = xr.open_dataset(download_path / f"{str(day_t1)}-surface-level_interp_uniform.nc", engine="netcdf4")
                atmos_vars_ds_t1 = xr.open_dataset(download_path / f"{str(day_t1)}-atmospheric_interp_uniform.nc", engine="netcdf4")

                ## Open surface and atmos ERA5 datasets for t1
                if day_t1 != day_t2:
                    if day_t2.year != year:
                        year = day_t2.year
                        download_path = Path(f"{data_dir}/era5_interp/{year}")
                    surf_vars_ds_t2 = xr.open_dataset(download_path / f"{str(day_t2)}-surface-level_interp_uniform.nc", engine="netcdf4")
                    atmos_vars_ds_t2 = xr.open_dataset(download_path / f"{str(day_t2)}-atmospheric_interp_uniform.nc", engine="netcdf4")
                else:
                    surf_vars_ds_t2 = surf_vars_ds_t1
                    atmos_vars_ds_t2 = atmos_vars_ds_t1

                if avg_kernel:
                    for var in sat_nudge_list:  # e.g. ["NO2"]
                        if var in chem_var_list:

                            print(f"VAR CHECK: {var}", flush=True)

                            true_vals = chem_data_test[var][:, six_hour_idx:six_hour_idx+2].permute(1, 0, 2, 3, 4).to(device=device_name, dtype=dtype)
                            aurora_vals = batch.atmos_vars[var]

                            mean_ak_achem_levels = np.array([0.16799302, 0.24112969, 0.572299, 0.76004654, 0.84239937, 0.92573033, 
                                                            1.01223258, 1.09242875, 1.12552418, 1.14823593, 1.17317673, 1.19390706, 1.15762777])

                            ak_tensor = torch.from_numpy(mean_ak_achem_levels).to(device=device_name, dtype=dtype).view(1, 1, -1, 1, 1)

                            true_vals_column_realistic = (true_vals * ak_tensor * dp_tensor).sum(dim=2, keepdim=True) / 9.81      # (2, 1, 1, lat, lon)
                            aurora_vals_column_realistic = (aurora_vals * ak_tensor * dp_tensor).sum(dim=2, keepdim=True) / 9.81   # (2, 1, 1, lat, lon)
                            scaling_factor_realistic = true_vals_column_realistic / aurora_vals_column_realistic

                            true_vals_column_uniform = (true_vals * dp_tensor).sum(dim=2, keepdim=True) / 9.81      # (2, 1, 1, lat, lon)
                            aurora_vals_column_uniform = (aurora_vals * dp_tensor).sum(dim=2, keepdim=True) / 9.81   # (2, 1, 1, lat, lon)
                            scaling_factor_uniform = true_vals_column_uniform / aurora_vals_column_uniform

                            print(f"SCALING FACTOR CHECK: {scaling_factor_realistic.shape}", flush=True)
                            # print(scaling_factor_realistic, flush=True)
                            # print(scaling_factor_uniform, flush=True)

                            # if step_count == nudge_int:  # first nudge only
                            #     np.savez(f"/home/disk/hermes06/ejkim1/AURORA/Output/nudge_{var}_{six_hour_idx}_FIX_LON.npz",
                            #             aurora=aurora_vals.detach().cpu().numpy(),
                            #             truth=true_vals.detach().cpu().numpy(),
                            #             mask=mask.detach().cpu().numpy(),
                            #             lat=batch.metadata.lat.cpu().numpy(),
                            #             lon=batch.metadata.lon.cpu().numpy())

                            if var == "NO2":
                                batch.atmos_vars[var] = torch.where(mask, aurora_vals * scaling_factor_realistic, aurora_vals)
                            elif var == "CO":
                                batch.atmos_vars[var] = torch.where(mask, aurora_vals * scaling_factor_uniform, aurora_vals)

                        elif var in met_var_list:

                            print(f"VAR CHECK: {var}", flush=True)

                            true_vals = torch.from_numpy(np.stack((atmos_vars_ds_t1[var].values[hour_t1], atmos_vars_ds_t2[var].values[hour_t2]))[:, None, :, :]).to(device=device_name, dtype=dtype)
                            aurora_vals = batch.atmos_vars[var]

                            true_vals_column_uniform = (true_vals * dp_tensor).sum(dim=2, keepdim=True) / 9.81      # (2, 1, 1, lat, lon)
                            aurora_vals_column_uniform = (aurora_vals * dp_tensor).sum(dim=2, keepdim=True) / 9.81   # (2, 1, 1, lat, lon)
                            scaling_factor_uniform = true_vals_column_uniform / aurora_vals_column_uniform

                            if var == "q":
                                batch.atmos_vars[var] = torch.where(mask, aurora_vals * scaling_factor_uniform, aurora_vals)
                else:
                    for var in sat_nudge_list:  # e.g. ["NO2"]
                        if var in chem_var_list:
                            true_vals = chem_data_test[var][:, six_hour_idx:six_hour_idx+2].permute(1, 0, 2, 3, 4).to(device=device_name, dtype=dtype)
                            batch.atmos_vars[var] = torch.where(mask, true_vals, batch.atmos_vars[var])
        
            ## update MET fields
            if met_nudge:

                if met_type == "era5":

                    ## Open surface and atmos ERA5 datasets for t0
                    download_path = Path(f"{data_dir}/era5_interp/{year}")

                    surf_vars_ds_t1 = xr.open_dataset(download_path / f"{str(day_t1)}-surface-level_interp_uniform.nc", engine="netcdf4")
                    atmos_vars_ds_t1 = xr.open_dataset(download_path / f"{str(day_t1)}-atmospheric_interp_uniform.nc", engine="netcdf4")

                    ## Open surface and atmos ERA5 datasets for t1
                    if day_t1 != day_t2:
                        if day_t2.year != year:
                            year = day_t2.year
                            download_path = Path(f"{data_dir}/era5_interp/{year}")
                        surf_vars_ds_t2 = xr.open_dataset(download_path / f"{str(day_t2)}-surface-level_interp_uniform.nc", engine="netcdf4")
                        atmos_vars_ds_t2 = xr.open_dataset(download_path / f"{str(day_t2)}-atmospheric_interp_uniform.nc", engine="netcdf4")
                    else:
                        surf_vars_ds_t2 = surf_vars_ds_t1
                        atmos_vars_ds_t2 = atmos_vars_ds_t1

                    ## Update designated met fields with true fields
                    if rollout_type == "nudged_partial":
                        for met_var in met_nudge_list:
                            batch.atmos_vars[met_var] = torch.from_numpy(np.stack((atmos_vars_ds_t1[met_var].values[hour_t1], atmos_vars_ds_t2[met_var].values[hour_t2]))[:, None, :, :]).to(device=device_name, dtype=dtype)
                    else:
                        batch.surf_vars["2t"] = torch.from_numpy(np.stack((surf_vars_ds_t1["t2m"].values[hour_t1], surf_vars_ds_t2["t2m"].values[hour_t2]), axis=0)[:, None, :, :]).to(device=device_name, dtype=dtype)
                        batch.surf_vars["10u"] = torch.from_numpy(np.stack((surf_vars_ds_t1["u10"].values[hour_t1], surf_vars_ds_t2["u10"].values[hour_t2]), axis=0)[:, None, :, :]).to(device=device_name, dtype=dtype)
                        batch.surf_vars["10v"] = torch.from_numpy(np.stack((surf_vars_ds_t1["v10"].values[hour_t1], surf_vars_ds_t2["v10"].values[hour_t2]), axis=0)[:, None, :, :]).to(device=device_name, dtype=dtype)
                        batch.surf_vars["msl"] = torch.from_numpy(np.stack((surf_vars_ds_t1["msl"].values[hour_t1], surf_vars_ds_t2["msl"].values[hour_t2]), axis=0)[:, None, :, :]).to(device=device_name, dtype=dtype)
                        for met_var in met_vars:
                            batch.atmos_vars[met_var] = torch.from_numpy(np.stack((atmos_vars_ds_t1[met_var].values[hour_t1], atmos_vars_ds_t2[met_var].values[hour_t2]))[:, None, :, :]).to(device=device_name, dtype=dtype)

                    atmos_vars_ds_t1.close()
                    surf_vars_ds_t1.close()
                    atmos_vars_ds_t2.close()
                    surf_vars_ds_t2.close()

                elif met_type == "merra2":

                    print("MERRA2 CHECK")

                    if rollout_type == "nudged_partial":
                        for var in met_nudge_list:
                            batch.atmos_vars[var] = met_data_test[var][:, six_hour_idx:(six_hour_idx+2)].permute(1, 0, 2, 3, 4).to(device=device_name, dtype=dtype)
                    else:
                        for var in met_vars: 
                            batch.atmos_vars[var] = met_data_test[var][:, six_hour_idx:(six_hour_idx+2)].permute(1, 0, 2, 3, 4).to(device=device_name, dtype=dtype)

            ## update CHEM fields
            if chem_nudge:
                if rollout_type == "nudged_partial":
                    for species in chem_nudge_list:
                        batch.atmos_vars[species] = chem_data_test[species][:, six_hour_idx:(six_hour_idx+2)].permute(1, 0, 2, 3, 4).to(device=device_name, dtype=dtype)
                else:
                    for species in species_list: 
                        batch.atmos_vars[species] = chem_data_test[species][:, six_hour_idx:(six_hour_idx+2)].permute(1, 0, 2, 3, 4).to(device=device_name, dtype=dtype)
            
                print("CHEM CHECK: ", batch.atmos_vars["O3"].shape, flush=True)

        print(f"STEP TIME: {np.round(time.time() - start_step, 2)}, {pred.metadata.time}", flush=True)
        
    print("ROLL STEP TIME: ", time.time() - start, flush=True)

def rollout(model: Aurora, batch: Batch, steps: int) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    batch = model.batch_transform_hook(batch)  # This might modify the available variables.
    # Use an arbitary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    counter = 0

    for _ in range(steps):

        start = time.time()

        pred = model.forward(batch)

        yield pred

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
    
        counter+=1
        print(f"Time: {time.time() - start}s, #{counter}")

