import torch

def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'