aur_levs = [1000, 925, 850, 700, 600, 500, 
                   400, 300, 250, 200, 150, 100, 50]

chem_vars = ["gc_o3", "gc_co", "O3", "OH", "CO", "NO", 
             "NO2", "ISOP", "CH2O", "Cl", "C12H4", 
             "C13H4", "CH4", "C14O"]

aur_var_dict = {
    "t": "Temperature", "q": "Specific Humidity",
    "u": "u Wind Vector", "v": "v Wind Vector",
    "z": "Geopotential Height", "O3": "Ozone",
    "OH": "OH", "CO": "CO", "NO": "NO", "NO2": "NO2",
    "CH4": "CH4", "Cl": "Cl", "C14O": "$^{14}$CO",
    "C12H4": "C12H4", "C13H4": "C13H4",
}

aur_unit_dict = {
    "t": "K", "q": "g / g", "u": "m / s",
    "v": "m / s", "z": "m",
    **{v: "ppb" for v in chem_vars},
    "OH": r"$10^6$ molec / cm$^3$",  # overrides the ppb entry from chem_vars
}

chem_colors = {
    "OH": "Greens", "ISOP": "Greens", "CH2O": "Greens",
    "NO": "Oranges", "NO2": "Oranges", "CO": "Oranges",
    "Cl": "YlGnBu", "O3": "YlGnBu",
    "CH4": "Purples", "C14O": "Purples",
    "C12H4": "Purples", "C13H4": "Purples",
}

phys_colors = {k: "cividis" for k in ["t", "q", "u", "v", "z"]}

chem_vmax = {
    1000: {"OH": 25e-5, "NO": 1.2, "NO2": 12, "Cl": 1e-7,
           "ISOP": 7, "CH2O": 7, "O3": 75, "CO": 500,
           "CH4": 1925, "C14O": 75e-11, "C12H4": 550, "C13H4": 6},
    500:  {"OH": 30e-5, "NO": 0.06, "NO2": 0.15, "Cl": 1e-7,
           "ISOP": 0.08, "CH2O": 0.4, "O3": 75, "CO": 200,
           "CH4": 1800, "C14O": 75e-11, "C12H4": 550, "C13H4": 6},
}

met_vmin = {
    1000: {"t": 235, "q": 0, "u": -15, "v": -16, "z": -10},
    500:  {"t": 225, "q": 0, "u": -20, "v": -35, "z": 0},
}

met_vmax = {
    1000: {"t": 300, "q": 2e-2, "u": 20, "v": 16, "z": 10},
    500:  {"t": 260, "q": 6e-3, "u": 50, "v": 35, "z": 0},
}

file_suffix_dict = {
    "free": "ROLLOUTS",
    "nudged": "ROLLOUTS_20_NUDGED",
    "nudged_met": "ROLLOUTS_20_NUDGED_METONLY",
    "nudged_chem": "ROLLOUTS_20_NUDGED_CHEMONLY",
    "nudged_tropomi_no2": "ROLLOUTS_TROPOMI_NO2",
    "nudged_tropomi_no2_ak_uniform": "ROLLOUTS_TROPOMI_NO2_AK_UNIFORM",
    "nudged_tropomi_no2_ak_realistic": "ROLLOUTS_TROPOMI_NO2_AK_REALISTIC"
}

fig_prefix_dict = {
    "free": "Free Rollout",
    "nudged": "Nudged Rollout",
    "nudged_met": "Met-Only Nudged",
    "nudged_chem": "Chem-Only Nudged",
}

spatial_subsets_dict = {
    "tropics": (slice(0, 15), slice(None)),
    "nh":      (slice(15, 30), slice(None)),
    "sh":      (slice(30, 45), slice(None)),
}

curtain_chem_vmax = {
    1000:{
        "OH": 15e-5,
        "NO": 0.2,
        "NO2": 2,
        "Cl": 1e-14,
        "O3": 75,
        "CO": 250,
        "CH4": 1925,
        "C14O": 75e-11,
        "C12H4": 550,
        "C13H4": 6,
    },
    500:{
        "OH": 30e-5,
        "NO": 0.02,
        "NO2": 0.1,
        "Cl": 1e-14,
        "O3": 75,
        "CO": 200,
        "CH4": 1800,
        "C14O": 75e-11,
        "C12H4": 550,
        "C13H4": 6,
    }
}

curtain_met_vmin = {
    1000:{
        "t": 235,
        "q": 0,
        "u": -15,
        "v": -10,
        "z": 0,
    },
    500:{
        "t": 225,
        "q": 0,
        "u": -20,
        "v": -35,
        "z": 0,
    }
}

curtain_met_vmax = {
    1000:{
        "t": 300,
        "q": 2e-2,
        "u": 20,
        "v": 10,
        "z": 0,
    },
    500:{
        "t": 260,
        "q": 6e-3,
        "u": 50,
        "v": 35,
        "z": 0,
    }
}