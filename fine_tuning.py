import os
import re
import tensorflow.keras as keras
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)                             
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='unet_final_renacimiento_validation_6.h5',
                        help="type of model") 
    parser.add_argument('-f',
                        '--fine_tuning',
                        type=int,
                        default=0,
                        help="layers not trainable")
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='',
                        help="name to add to the model") 

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    modelo = args.model
    fine_tune_at = args.fine_tuning
    name = args.name
    path = '/home/mr1142/Documents/Data/patologic'
    batch = 8
    epoch = 200

    import funciones_modelos.unet_doble_loss as u_loss
    import funciones_imagenes.image_funct as im
    import funciones_imagenes.extra_functions as ex
    import funciones_modelos.logs as logs
    import funciones_modelos.evaluation as ev

    p = os.path.join('/home/mr1142/Documents/Data/models', modelo)
    model = keras.models.load_model(p, 
                                custom_objects={"MyLoss": u_loss.MyLoss, 
                                                "loss_mask": u_loss.loss_mask, 
                                                "dice_coef_loss": ex.dice_coef_loss,
                                                "dice_coef": ex.dice_coef})

    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                    keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()]

    type = re.split('_', modelo)[0]
    if type == 'unet':
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                                loss=ex.dice_coef_loss,
                                metrics=metrics)
    else:
        model.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics =metrics)


    masks_name = ex.list_files(os.path.join(path, 'mascara'))
    images = im.create_tensor(path, 'images', masks_name, im.normalize)

    masks = im.create_tensor(path, 'mascara', masks_name, im.binarize)
    images = im.create_tensor(path, 'images', masks_name, im.normalize)

    images, masks = im.augment_tensor(images,masks)

    modelo = re.split('.h5', modelo)[0]
    callb = [logs.tensorboard(modelo + '_' + 'fine_tuning'), logs.early_stop(10)]

    history = model.fit(images,masks,
                            batch_size = 8,
                            epochs = 100,
                            callbacks= callb,
                            shuffle = True,
                            validation_split = 0.2) 

    model.save('/home/mr1142/Documents/Data/models/' + modelo + '_' + 'fine_tuning_' + name + '.h5')
    ev.all_evaluations(type, modelo + '_' + 'fine_tuning_' + name, model)
