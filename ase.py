
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
import funciones_imagenes.extra_functions as ex
import re
import tensorflow as tf
import funciones_modelos.unet_doble_loss as u_loss
import funciones_modelos.unet_funct as u_net
import funciones_modelos.evaluation as ev


# BORRAR RESULTADOS DEL CSV
path = '/home/mr1142/Documents/Data/models/mascaras/validation_results'
csvs = ex.list_files(path)

for csv in csvs:
    df = pd.read_csv(os.path.join(path, csv))
    index = [i for i in df.index if bool(re.search('fine_tuning', df['name'][i]))]
    df = df.drop(index)
    df.to_csv(os.path.join(path, csv), index = False)



# EVALUAR MÚLTIPLES MODELOS
path = '/home/mr1142/Documents/Data/models/mascaras'
names = ex.list_files(path)
names = [name for name in names if bool(re.search('uloss_', name))]
metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]



# EVALUAR UN MODELO
model = 'uloss_final_renacimiento_validation_6.h5'
metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

path = os.path.join('/home/mr1142/Documents/Data/models/mascaras', model)

unet_model = tf.keras.models.load_model(path, 
                custom_objects={"MyLoss": u_loss.MyLoss,
                            "loss_mask": u_loss.loss_mask, 
                            "dice_coef_loss": ex.dice_coef_loss,
                            "dice_coef": ex.dice_coef}) 


for model in names:
    if bool(re.search('uloss', model)):
        path = os.path.join('/home/mr1142/Documents/Data/models/mascaras', model)
        unet_model = tf.keras.models.load_model(path, 
                                     custom_objects={"MyLoss": u_loss.MyLoss,
                                                    "loss_mask": u_loss.loss_mask, 
                                                    "dice_coef_loss": ex.dice_coef_loss,
                                                    "dice_coef": ex.dice_coef})        
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics =metrics)
        ev.all_evaluations('uloss', 'patologic_' + model[6:-3], unet_model, '/home/mr1142/Documents/Data/patologic')
    else:
        path = os.path.join('/home/mr1142/Documents/Data/models/mascaras', model)
        unet_model = tf.keras.models.load_model(path, 
                                     custom_objects={"MyLoss": u_loss.MyLoss,
                                                    "loss_mask": u_loss.loss_mask, 
                                                     "dice_coef_loss": ex.dice_coef_loss,
                                                     "dice_coef": ex.dice_coef})
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        ev.all_evaluations('unet', 'patologic_' + model[5:-3], unet_model, '/home/mr1142/Documents/Data/patologic')


# '/home/mr1142/Documents/Data/patologic'

tf.keras.utils.plot_model(unet_model, to_file='images/ueff.png', show_layer_names = False, dpi=500)
tf.keras.utils.model_to_dot(unet_model, to_file='images/unet.png')


import numpy as np
path = '/home/mr1142/Documents/Data/models/validation_results/validation_results' + '.csv'
df = pd.read_csv(path)

evaluations = list(df.columns[2:9])

for ev in evaluations:
    print(ev)
    print('unet')
    np.mean(df[ev][df.type == 'unet'])
    print('uloss') 
    np.mean(df[ev][df.type == 'uloss'])
    print('-----')




import os
import tensorflow as tf
import funciones_modelos.evaluation as ev
path = '/home/mr1142/Documents/Data/segmentation/splited/train'
pixels = 256
import neural_net.neural_net.image_funct_bad as im
import gestion_imagenes.extra_functions as ex
# DATOS
masks_name = ex.list_files(os.path.join(path, 'mascara'))

images = im.create_tensor(path, 'images', masks_name, im.normalize)
masks = im.create_tensor(path, 'mascara', masks_name, im.binarize)
images.min()

# Aumento
images, masks = im.augment_tensor(images,masks,'new',2)


ima = os.path.join(path, 'images')

a = '/home/mr1142/Documents/Data/segmentation/images'