import os
import numpy as np
import cv2
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import re
import pandas as pd

import funciones_imagenes.apply_mask as msk
import funciones_modelos.unet_funct as u_net
import funciones_modelos.unet_doble_loss as u_loss
import funciones_imagenes.extra_functions as ex


unet = keras.models.load_model('/home/mr1142/Documents/Data/models/unet_final_renacimiento_validation_6.h5', 
                                     custom_objects={"loss_mask": u_loss.loss_mask, 
                                                     "dice_coef_loss": ex.dice_coef_loss,
                                                     "dice_coef": ex.dice_coef})

uloss = keras.models.load_model('/home/mr1142/Documents/Data/models/uloss_final_renacimiento_validation_8.h5', 
                                     custom_objects={"MyLoss": u_loss.MyLoss, 
                                                     "loss_mask": u_loss.loss_mask, 
                                                     "dice_coef_loss": ex.dice_coef_loss,
                                                     "dice_coef": ex.dice_coef})

