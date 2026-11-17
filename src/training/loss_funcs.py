import torch
import torch.nn as nn

from aurora.normalisation import normalise_atmos_var, normalise_surf_var

class AuroraL1Loss_NoWeight(nn.Module):
    """Loss function"""
    def __init__(self, reduction='mean', scale=1.0):
        super(AuroraL1Loss_NoWeight, self).__init__()
        self.reduction = reduction
        self.scale = scale

    def forward(self, preds, trues, gc_weights):

        # Constants
        V_S = 4
        V_A = 6
        alpha = 0.25
        beta = 1.0
    
        surf_term = 0
        for var in preds.surf_vars:
            norm_var_pred = normalise_surf_var(preds.surf_vars[var], var)
            norm_var_true = normalise_surf_var(trues.surf_vars[var], var)
            surf_term += torch.sum(torch.abs(norm_var_pred - norm_var_true[:, :-1]))
    
        surf_term *= alpha
        surf_term *= (1 / (44*72))
    
        atmos_term = 0
        for var in preds.atmos_vars:
            norm_var_pred = normalise_atmos_var(preds.atmos_vars[var], var, preds.metadata.atmos_levels)
            norm_var_true = normalise_atmos_var(trues.atmos_vars[var], var, trues.metadata.atmos_levels)
            atmos_term += torch.sum(torch.abs(norm_var_pred - norm_var_true[:, :, :-1]))
    
        atmos_term *= (1 / (44*72*13))

        loss = surf_term + atmos_term
        loss *= (1 / (V_S + V_A))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction

class AuroraL1Loss(nn.Module):
    """Loss function"""
    def __init__(self, reduction='mean', scale=1.0):
        super(AuroraL1Loss, self).__init__()
        self.reduction = reduction
        self.scale = scale

    def forward(self, preds, trues, gc_weights, patch_size):

        extra_dim_flag = (preds.surf_vars["2t"].shape[2] % 2 != 0)
        norm_val = preds.surf_vars["2t"].shape[2]

        # Constants
        V_S = 4
        V_A = 6
        alpha = 0.25
        beta = 1.0
    
        surf_term = 0
        for var in preds.surf_vars:
            norm_var_pred = normalise_surf_var(preds.surf_vars[var], var)
            norm_var_true = normalise_surf_var(trues.surf_vars[var], var)
            if extra_dim_flag:
                surf_term += torch.sum(torch.abs(norm_var_pred - norm_var_true))
            else:
                surf_term += torch.sum(torch.abs(norm_var_pred - norm_var_true[:, :-1]))
    
        surf_term *= alpha
        surf_term *= (1 / (norm_val*72))
    
        atmos_term = 0
        for var in preds.atmos_vars:
            norm_var_pred = normalise_atmos_var(preds.atmos_vars[var], var, preds.metadata.atmos_levels)
            norm_var_true = normalise_atmos_var(trues.atmos_vars[var], var, trues.metadata.atmos_levels)
            if var in gc_weights.keys():
                for idx in range(norm_var_pred.shape[2]):
                    if extra_dim_flag:
                        atmos_term += (gc_weights[var][idx] * torch.sum(torch.abs(norm_var_pred[:, :, idx] - norm_var_true[:, idx])))
                    else:
                        atmos_term += (gc_weights[var][idx] * torch.sum(torch.abs(norm_var_pred[:, :, idx] - norm_var_true[:, idx, :-1])))
            else:
                if extra_dim_flag:
              
                    atmos_term += torch.sum(torch.abs(norm_var_pred - norm_var_true))
                else:
                    atmos_term += torch.sum(torch.abs(norm_var_pred - norm_var_true[:, :, :-1]))
    
        atmos_term *= (1 / (norm_val*72*13))

        loss = surf_term + atmos_term
        loss *= (1 / (V_S + V_A))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction


class AuroraL1Loss_PolarMask(nn.Module):
    """Loss function"""
    def __init__(self, reduction='mean', scale=1.0):
        super(AuroraL1Loss_PolarMask, self).__init__()
        self.reduction = reduction
        self.scale = scale

    def forward(self, preds, trues, gc_weights, patch_size):

        extra_dim_flag = (preds.surf_vars["2t"].shape[2] % 2 != 0)
        norm_val = preds.surf_vars["2t"].shape[2]

        atmos_var_list = [var for var in preds.atmos_vars]

        # Constants
        V_S = 4
        V_A = int(len(atmos_var_list))
        alpha = 0.25
        beta = 1.0
    
        surf_term = 0.0
        for var in preds.surf_vars:
            norm_var_pred = normalise_surf_var(preds.surf_vars[var], var)
            norm_var_true = normalise_surf_var(trues.surf_vars[var], var)

            # Polar Mask
        
            # number of latitudes
            nlat = norm_var_pred.shape[-2]   # assuming (..., lat, lon)
            
            lat_mask = torch.ones(nlat, device=norm_var_pred.device)
            lat_mask[0] = 0.0
            lat_mask[-1] = 0.0
            
            # reshape for broadcasting over lon
            lat_mask = lat_mask.view(1, nlat, 1)   # or (nlat, 1) if no time dim
            
            if extra_dim_flag:
                surf_diff = torch.abs(norm_var_pred[-1] - norm_var_true)
                # surf_term += torch.sum(torch.abs(norm_var_pred[-1] - norm_var_true))
            else:
                surf_diff = torch.abs(norm_var_pred[-1] - norm_var_true[:, :-1])
                # surf_term += torch.sum(torch.abs(norm_var_pred[-1] - norm_var_true[:, :-1]))

            surf_diff = surf_diff * lat_mask
            surf_term += torch.sum(surf_diff)
        
        surf_term *= alpha
        surf_term *= (1 / (norm_val*72))
    
        atmos_term = 0.0
        for var in preds.atmos_vars:
            norm_var_pred = normalise_atmos_var(preds.atmos_vars[var], var, preds.metadata.atmos_levels)
            norm_var_true = normalise_atmos_var(trues.atmos_vars[var], var, trues.metadata.atmos_levels)

            # Polar Mask
        
            # number of latitudes
            nlat = norm_var_pred.shape[-2]   # assuming (..., lat, lon)
            
            lat_mask = torch.ones(nlat, device=norm_var_pred.device)
            lat_mask[0] = 0.0
            lat_mask[-1] = 0.0
            
            # reshape for broadcasting over lvl and lon
            lat_mask = lat_mask.view(1, 1, nlat, 1)
            
            if var in gc_weights.keys():

                # Loop over levels
                for idx in range(norm_var_pred.shape[2]):
                    if extra_dim_flag:
                        atmos_diff = torch.abs(norm_var_pred[-1, :, idx] - norm_var_true[:, idx])
                        # atmos_term += (gc_weights[var][idx] * torch.sum(torch.abs(norm_var_pred[-1, :, idx] - norm_var_true[:, idx])))
                    else:
                        atmos_diff = torch.abs(norm_var_pred[-1, :, idx] - norm_var_true[:, idx, :-1])
                        # atmos_term += (gc_weights[var][idx] * torch.sum(torch.abs(norm_var_pred[-1, :, idx] - norm_var_true[:, idx, :-1])))

                    atmos_diff = atmos_diff * lat_mask
                    atmos_term += gc_weights[var][idx] * torch.sum(atmos_diff)
            else:
                if extra_dim_flag:
                    atmos_diff = torch.abs(norm_var_pred[-1] - norm_var_true)
                    # atmos_term += torch.sum(torch.abs(norm_var_pred[-1] - norm_var_true))
                else:
                    atmos_diff = torch.abs(norm_var_pred[-1] - norm_var_true[:, :, :-1])
                    # atmos_term += torch.sum(torch.abs(norm_var_pred[-1] - norm_var_true[:, :, :-1]))

                atmos_diff = atmos_diff * lat_mask
                atmos_term += torch.sum(atmos_diff)
    
        atmos_term *= (1 / (norm_val*72*13))

        loss = surf_term + atmos_term
        loss *= (1 / (V_S + V_A))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction