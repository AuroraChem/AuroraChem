import numpy as np
import xarray as xr

import torch

from aurora import Batch, Metadata
from aurora.normalisation import locations, scales

from pathlib import Path

from datetime import datetime
from datetime import time as time_dt

def make_batch_era5(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                    day_idx, chem_data, static_data, interp_scheme, new_static, data_dir):
    
    download_year = day_t1.year
    download_path = Path(f"{data_dir}/era5_interp/{download_year}")
    static_vars_ds = static_data[download_year]

    surf_vars_ds_t1 = xr.open_dataset(download_path / f"{str(day_t1)}-surface-level_interp{interp_scheme}.nc", engine="netcdf4")
    atmos_vars_ds_t1 = xr.open_dataset(download_path / f"{str(day_t1)}-atmospheric_interp{interp_scheme}.nc", engine="netcdf4")

    # Check if batch crosses over two days
    if day_t1 != day_t2:
        if day_t2.year != download_year:
            download_year = day_t2.year
            download_path = Path(f"{data_dir}/era5_interp/{download_year}")
            static_vars_ds = static_data[download_year]
        surf_vars_ds_t2 = xr.open_dataset(download_path / f"{str(day_t2)}-surface-level_interp{interp_scheme}.nc", engine="netcdf4")
        atmos_vars_ds_t2 = xr.open_dataset(download_path / f"{str(day_t2)}-atmospheric_interp{interp_scheme}.nc", engine="netcdf4")
    else:
        surf_vars_ds_t2 = surf_vars_ds_t1
        atmos_vars_ds_t2 = atmos_vars_ds_t1

    # gc_data["pressure_level"] = atmos_vars_ds_t1.pressure_level
    # gc_day_idx = day_idx * 2

    # print(f"IDX CHECK: {day_t1}, {day_t2}, {day_t3}, {hour_t1}, {hour_t2}, {hour_t3}, {day_idx}")

    input_batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(np.stack((surf_vars_ds_t1["t2m"].values[hour_t1], surf_vars_ds_t2["t2m"].values[hour_t2]))[None]),
            "10u": torch.from_numpy(np.stack((surf_vars_ds_t1["u10"].values[hour_t1], surf_vars_ds_t2["u10"].values[hour_t2]))[None]),
            "10v": torch.from_numpy(np.stack((surf_vars_ds_t1["v10"].values[hour_t1], surf_vars_ds_t2["v10"].values[hour_t2]))[None]),
            "msl": torch.from_numpy(np.stack((surf_vars_ds_t1["msl"].values[hour_t1], surf_vars_ds_t2["msl"].values[hour_t2]))[None]),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(np.stack((atmos_vars_ds_t1["t"].values[hour_t1], atmos_vars_ds_t2["t"].values[hour_t2]))[None]),
            "u": torch.from_numpy(np.stack((atmos_vars_ds_t1["u"].values[hour_t1], atmos_vars_ds_t2["u"].values[hour_t2]))[None]),
            "v": torch.from_numpy(np.stack((atmos_vars_ds_t1["v"].values[hour_t1], atmos_vars_ds_t2["v"].values[hour_t2]))[None]),
            "q": torch.from_numpy(np.stack((atmos_vars_ds_t1["q"].values[hour_t1], atmos_vars_ds_t2["q"].values[hour_t2]))[None]),
            "z": torch.from_numpy(np.stack((atmos_vars_ds_t1["z"].values[hour_t1], atmos_vars_ds_t2["z"].values[hour_t2]))[None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds_t1.lat.values.copy()),
            lon=torch.from_numpy(surf_vars_ds_t1.lon.values),
            time=(surf_vars_ds_t1.valid_time.values.astype("datetime64[s]").tolist()[hour_t1],
                  surf_vars_ds_t2.valid_time.values.astype("datetime64[s]").tolist()[hour_t2],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds_t1.pressure_level.values),
        ),
    )

    for chem in chem_data:
        gc_data = chem_data[chem]
        # print("GC DATA TIME", gc_data[day_idx:(day_idx+2)].time)
        # exit()
        input_batch.atmos_vars[chem] = gc_data[:, day_idx:(day_idx+2)]

    # Check if batch crosses over two days
    if day_t2 != day_t3:
        if day_t3.year != download_year:
            download_year = day_t3.year
            download_path = Path(f"{data_dir}/era5_interp/{download_year}")
            # static_vars_ds = xr.open_dataset(download_path / f"static_interp{interp_scheme}.nc", engine="netcdf4")
            static_vars_ds = static_data[download_year]
        surf_vars_ds_t3 = xr.open_dataset(download_path / f"{str(day_t3)}-surface-level_interp{interp_scheme}.nc", engine="netcdf4")
        atmos_vars_ds_t3 = xr.open_dataset(download_path / f"{str(day_t3)}-atmospheric_interp{interp_scheme}.nc", engine="netcdf4")
    else:
        surf_vars_ds_t3 = surf_vars_ds_t2
        atmos_vars_ds_t3 = atmos_vars_ds_t2

    true_batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_vars_ds_t3["t2m"].values[hour_t3][None]),
            "10u": torch.from_numpy(surf_vars_ds_t3["u10"].values[hour_t3][None]),
            "10v": torch.from_numpy(surf_vars_ds_t3["v10"].values[hour_t3][None]),
            "msl": torch.from_numpy(surf_vars_ds_t3["msl"].values[hour_t3][None]),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds_t3["t"].values[hour_t3][None]),
            "u": torch.from_numpy(atmos_vars_ds_t3["u"].values[hour_t3][None]),
            "v": torch.from_numpy(atmos_vars_ds_t3["v"].values[hour_t3][None]),
            "q": torch.from_numpy(atmos_vars_ds_t3["q"].values[hour_t3][None]),
            "z": torch.from_numpy(atmos_vars_ds_t3["z"].values[hour_t3][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds_t3.lat.values.copy()),
            lon=torch.from_numpy(surf_vars_ds_t3.lon.values),
            time=(surf_vars_ds_t3.valid_time.values.astype("datetime64[s]").tolist()[hour_t3],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds_t3.pressure_level.values),
        ),
    )

    for chem in chem_data:
        gc_data = chem_data[chem]
        true_batch.atmos_vars[chem] = gc_data[:, day_idx+2]

    if new_static:
        
        ## Add time of day variables
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
        
        input_batch.static_vars["tod_cos"] = torch.from_numpy(tod_cos[hour_t2])
        input_batch.static_vars["tod_sin"] = torch.from_numpy(tod_sin[hour_t2])
        input_batch.static_vars["dow_cos"] = torch.from_numpy(dow_cos[(day_t2.weekday() + 1) % 7])
        input_batch.static_vars["dow_sin"] = torch.from_numpy(dow_sin[(day_t2.weekday() + 1) % 7])
        
        doy_idx = (day_t2.timetuple().tm_yday - 1) % 365
        input_batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[doy_idx])
        input_batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[doy_idx])
        
        # input_batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx + 1) % 365])
        # input_batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx + 1) % 365])

        true_batch.static_vars["tod_cos"] = torch.from_numpy(tod_cos[hour_t3])
        true_batch.static_vars["tod_sin"] = torch.from_numpy(tod_sin[hour_t3])
        true_batch.static_vars["dow_cos"] = torch.from_numpy(dow_cos[(day_t3.weekday() + 1) % 7])
        true_batch.static_vars["dow_sin"] = torch.from_numpy(dow_sin[(day_t3.weekday() + 1) % 7])

        doy_idx = (day_t3.timetuple().tm_yday - 1) % 365
        true_batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[doy_idx])
        true_batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[doy_idx])

        # true_batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx + 2) % 365])
        # true_batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx + 2) % 365])

        time_var_list = ["tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin"]

        for var in time_var_list:
            locations[var] = 0.0
            scales[var] = 1.0

        ## Add emissions variables
        
        month = day_t2.month - 1
        year = day_t2.year

        nh3_emis = xr.open_dataset(f"{data_dir}/emissions/ammonia_{year}.nc", decode_times=False)
        co_emis = xr.open_dataset(f"{data_dir}/emissions/carbon-monoxide_{year}.nc", decode_times=False)
        nox_emis = xr.open_dataset(f"{data_dir}/emissions/nitrogen-oxides_{year}.nc", decode_times=False)
        so2_emis = xr.open_dataset(f"{data_dir}/emissions/sulphur-dioxide_{year}.nc", decode_times=False)

        input_batch.static_vars["nh3_emis"] = torch.from_numpy(nh3_emis["total_anthro"][month].data / nh3_emis["total_anthro"][month].data.max())
        input_batch.static_vars["co_emis"] = torch.from_numpy(co_emis["total_anthro"][month].data / co_emis["total_anthro"][month].data.max())
        input_batch.static_vars["nox_emis"] = torch.from_numpy(nox_emis["total_anthro"][month].data / nox_emis["total_anthro"][month].data.max())
        input_batch.static_vars["so2_emis"] = torch.from_numpy(so2_emis["total_anthro"][month].data / so2_emis["total_anthro"][month].data.max())

        month = day_t3.month - 1
        true_batch.static_vars["nh3_emis"] = torch.from_numpy(nh3_emis["total_anthro"][month].data / nh3_emis["total_anthro"][month].data.max())
        true_batch.static_vars["co_emis"] = torch.from_numpy(co_emis["total_anthro"][month].data / co_emis["total_anthro"][month].data.max())
        true_batch.static_vars["nox_emis"] = torch.from_numpy(nox_emis["total_anthro"][month].data / nox_emis["total_anthro"][month].data.max())
        true_batch.static_vars["so2_emis"] = torch.from_numpy(so2_emis["total_anthro"][month].data / so2_emis["total_anthro"][month].data.max())

        emis_var_list = ["nh3_emis", "co_emis", "nox_emis", "so2_emis"]

        for var in emis_var_list:
            locations[var] = 0.0
            scales[var] = 1.0


    surf_vars_ds_t1.close()
    atmos_vars_ds_t1.close()
    surf_vars_ds_t2.close()
    atmos_vars_ds_t2.close()
    surf_vars_ds_t3.close()
    atmos_vars_ds_t3.close()
    
    return input_batch, true_batch

def make_batch_merra2(day_t1, day_t2, day_t3, hour_t1, hour_t2, hour_t3, 
                      day_idx, chem_data, merra2_data, 
                      static_data, interp_scheme, new_static, data_dir):

    batch_met_var_dict = {
        "T":"t",
        "SPHU":"q",
        "U":"u",
        "V":"v",
        "Z":"z",
        "TS":"2t",
        "U10M":"10u",
        "V10M":"10v",
        "PSC2WET":"msl",
    }

    merra2_atmos_vars = ["T", "SPHU", "U", "V", "Z"]
    
    download_year = day_t1.year

    ## Make input batch
    ## ERA5 Static Data
    static_vars_ds = static_data[download_year]

    # Check if batch crosses over two days
    if day_t1 != day_t2:
        if day_t2.year != download_year:
            download_year = day_t2.year
            static_vars_ds = static_data[download_year]

    input_batch = Batch(
        surf_vars={},
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={},
        metadata=Metadata(
            lat=torch.from_numpy(static_vars_ds.lat.values.copy()),
            lon=torch.from_numpy(static_vars_ds.lon.values),
            time=(datetime.combine(day_t1, time_dt(hour_t1*6, 0)),
                  datetime.combine(day_t2, time_dt(hour_t2*6, 0)),),
            atmos_levels=(1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.),
        ),
    )

    for met in merra2_data:
        met_data = merra2_data[met]
        batch_met_var = batch_met_var_dict[met]
        if met in merra2_atmos_vars:
            input_batch.atmos_vars[batch_met_var] = met_data[:, day_idx:(day_idx+2)]
        else:
            input_batch.surf_vars[batch_met_var] = met_data[:, day_idx:(day_idx+2)]
    for chem in chem_data:
        gc_data = chem_data[chem]
        input_batch.atmos_vars[chem] = gc_data[:, day_idx:(day_idx+2)]

    ## Make true batch
    # Check if batch crosses over two days
    if day_t2 != day_t3:
        if day_t3.year != download_year:
            download_year = day_t3.year
            static_vars_ds = static_data[download_year]

    true_batch = Batch(
        surf_vars={},
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={},
        metadata=Metadata(
            lat=torch.from_numpy(static_vars_ds.lat.values.copy()),
            lon=torch.from_numpy(static_vars_ds.lon.values),
            time=(datetime.combine(day_t3, time_dt(hour_t3*6, 0)),),
            atmos_levels=(1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.),
        ),
    )
    
    for met in merra2_data:
        met_data = merra2_data[met]
        batch_met_var = batch_met_var_dict[met]
        if met in merra2_atmos_vars:
            true_batch.atmos_vars[batch_met_var] = met_data[:, day_idx+2]
        else:
            true_batch.surf_vars[batch_met_var] = met_data[:, day_idx+2]
    for chem in chem_data:
        gc_data = chem_data[chem]
        true_batch.atmos_vars[chem] = gc_data[:, day_idx+2]

    if new_static:
        
        ## Add time of day variables
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
        
        input_batch.static_vars["tod_cos"] = torch.from_numpy(tod_cos[hour_t2])
        input_batch.static_vars["tod_sin"] = torch.from_numpy(tod_sin[hour_t2])
        input_batch.static_vars["dow_cos"] = torch.from_numpy(dow_cos[(day_t2.weekday() + 1) % 7])
        input_batch.static_vars["dow_sin"] = torch.from_numpy(dow_sin[(day_t2.weekday() + 1) % 7])
        input_batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx + 1) % 365])
        input_batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx + 1) % 365])

        true_batch.static_vars["tod_cos"] = torch.from_numpy(tod_cos[hour_t3])
        true_batch.static_vars["tod_sin"] = torch.from_numpy(tod_sin[hour_t3])
        true_batch.static_vars["dow_cos"] = torch.from_numpy(dow_cos[(day_t3.weekday() + 1) % 7])
        true_batch.static_vars["dow_sin"] = torch.from_numpy(dow_sin[(day_t3.weekday() + 1) % 7])
        true_batch.static_vars["doy_cos"] = torch.from_numpy(doy_cos[(day_idx + 2) % 365])
        true_batch.static_vars["doy_sin"] = torch.from_numpy(doy_sin[(day_idx + 2) % 365])

        time_var_list = ["tod_cos", "tod_sin", "dow_cos", "dow_sin", "doy_cos", "doy_sin"]

        for var in time_var_list:
            locations[var] = 0.0
            scales[var] = 1.0

        ## Add emissions variables
        
        month = day_t2.month - 1
        year = day_t2.year

        nh3_emis = xr.open_dataset(f"{data_dir}/emissions/ammonia_{year}.nc", decode_times=False)
        co_emis = xr.open_dataset(f"{data_dir}/emissions/carbon-monoxide_{year}.nc", decode_times=False)
        nox_emis = xr.open_dataset(f"{data_dir}/emissions/nitrogen-oxides_{year}.nc", decode_times=False)
        so2_emis = xr.open_dataset(f"{data_dir}/emissions/sulphur-dioxide_{year}.nc", decode_times=False)

        input_batch.static_vars["nh3_emis"] = torch.from_numpy(nh3_emis["total_anthro"][month].data / nh3_emis["total_anthro"][month].data.max())
        input_batch.static_vars["co_emis"] = torch.from_numpy(co_emis["total_anthro"][month].data / co_emis["total_anthro"][month].data.max())
        input_batch.static_vars["nox_emis"] = torch.from_numpy(nox_emis["total_anthro"][month].data / nox_emis["total_anthro"][month].data.max())
        input_batch.static_vars["so2_emis"] = torch.from_numpy(so2_emis["total_anthro"][month].data / so2_emis["total_anthro"][month].data.max())

        month = day_t3.month - 1
        true_batch.static_vars["nh3_emis"] = torch.from_numpy(nh3_emis["total_anthro"][month].data / nh3_emis["total_anthro"][month].data.max())
        true_batch.static_vars["co_emis"] = torch.from_numpy(co_emis["total_anthro"][month].data / co_emis["total_anthro"][month].data.max())
        true_batch.static_vars["nox_emis"] = torch.from_numpy(nox_emis["total_anthro"][month].data / nox_emis["total_anthro"][month].data.max())
        true_batch.static_vars["so2_emis"] = torch.from_numpy(so2_emis["total_anthro"][month].data / so2_emis["total_anthro"][month].data.max())

        emis_var_list = ["nh3_emis", "co_emis", "nox_emis", "so2_emis"]

        for var in emis_var_list:
            locations[var] = 0.0
            scales[var] = 1.0
    
    return input_batch, true_batch

def trim_batch_lat(batch, batch_type):
    '''
    Trim extra latitude band and permute batch shapes from [batch, time, ...] to [time, batch, ...]
    '''
    
    if batch_type == "input":
        for var in batch.surf_vars:
            if batch.surf_vars[var].shape[-2] > 45:
                batch.surf_vars[var] = batch.surf_vars[var][..., :-1, :]
            batch.surf_vars[var] = batch.surf_vars[var].permute(1, 0, 2, 3)
        for var in batch.static_vars:
            if batch.static_vars[var].shape[-2] > 45:
                batch.static_vars[var] = batch.static_vars[var][..., :-1, :]
        for var in batch.atmos_vars:
            if batch.atmos_vars[var].shape[-2] > 45:
                batch.atmos_vars[var] = batch.atmos_vars[var][..., :-1, :]
            batch.atmos_vars[var] = batch.atmos_vars[var].permute(1, 0, 2, 3, 4)
        if batch.metadata.lat.shape[0] > 45:
            batch.metadata.lat = batch.metadata.lat[:-1]
        return batch
    elif batch_type == "true":
        for var in batch.surf_vars:
            if batch.surf_vars[var].shape[-2] > 45:
                batch.surf_vars[var] = batch.surf_vars[var][..., :-1, :]
        for var in batch.static_vars:
            if batch.static_vars[var].shape[-2] > 45:
                batch.static_vars[var] = batch.static_vars[var][..., :-1, :]
        for var in batch.atmos_vars:
            if batch.atmos_vars[var].shape[-2] > 45:
                batch.atmos_vars[var] = batch.atmos_vars[var][..., :-1, :]
        if batch.metadata.lat.shape[0] > 45:
            batch.metadata.lat = batch.metadata.lat[:-1]
        return batch
    else:
        print("NO BATCH TYPE")
        return -1