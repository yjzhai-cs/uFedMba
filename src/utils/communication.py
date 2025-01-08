
from typing import Tuple

def get_bandwidth_rate(bandwidth_type:str="5G") -> Tuple[int, int]:
    """
    return upload/download bandwidth rate whose units is Mbps
    e.g., 10 Mbps = 10 Megabits per second = 10,000,000 bit per second
    """
    if bandwidth_type == "5G":
        return 20, 200
    elif bandwidth_type == "4G_LTE-Advanced":
        return 10, 42
    elif bandwidth_type == "3G_HSPA+":
        return 3, 6
    else:
        raise RuntimeError(f"{bandwidth_type} is not defined")

def get_upload_communication_time(model_size: int, bandwidth_type:str="5G") -> float:
    """
    model_size: The number of parameters of the model
    retrun the communication time(s) of uploading model
    """
    
    upload_rate, _ = get_bandwidth_rate(bandwidth_type=bandwidth_type)

    return (model_size * 4 * 8) / (upload_rate * 1000000)

def get_download_communication_time(model_size: int, bandwidth_type:str="5G") -> float:
    """
    model_size: The number of parameters of the model
    return the communication time(s) of downloading model
    """

    _, download_rate = get_bandwidth_rate(bandwidth_type=bandwidth_type)

    return (model_size * 4 * 8) / (download_rate * 1000000) # using Tensor.float32 store model parameters