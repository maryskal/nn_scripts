import argparse
import os
import funciones_modelos.logs as logs
import tensorflow as tf
import logging_function as log
import funciones_modelos.evaluation as ev


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-c',
                        '--callbacks',
                        type=bool,
                        default=False,
                        help="Callbacks")
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")                               
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="unet",
                        help="type of model") 

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    callbacks = args.callbacks
    name = args.name
    model = args.model
    path = '/home/mr1142/Documents/Data/segmentation'
    batch = 8
    epoch = 200
    pixels = 256

    #----------------------------------------------------
    import gestion_imagenes.image_funct as im
    import funciones_modelos.unet_doble_loss as u_loss
    import funciones_modelos.unet_funct as u_net
    import gestion_imagenes.extra_functions as ex

    metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

    def unet():
        unet_model = u_net.build_unet_model(256,1)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
        return unet_model

    def uloss():
        unet_model = u_loss.build_unet_model(256)
        unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 1e-4),
                            loss=u_loss.MyLoss,
                            metrics =metrics)
        return unet_model
    #----------------------------------------------------

    masks_name = ex.list_files(os.path.join(path, 'mascara'))
    images = im.create_tensor(path, 'images', masks_name, im.normalize, pixels)

    masks = im.create_tensor(path, 'mascara', masks_name, im.binarize, pixels)
    images = im.create_tensor(path, 'images', masks_name, im.normalize, pixels)

    print('\n')
    print(images.shape)

    images, masks = im.augment_tensor(images,masks)

    print(images.min())
    print(images.max())
    print(images.shape)

    if callbacks:
        callb = [logs.tensorboard(model + '_' + name), logs.early_stop(10)]
    else:
        callb = []


    if model == 'unet':
        unet_model = unet()
    elif model == 'uloss':
        unet_model = uloss()
    else:
        unet_model = None
        print('\n INCORRECT MODEL \n')

    history = unet_model.fit(images,masks,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks= callb,
                            shuffle = True,
                            validation_split = 0.2)        

    unet_model.save('/home/mr1142/Documents/Data/models/' + model + '_' + name + '.h5')
    
    # EVALUACIÓN
    ev.all_evaluations(model, name, unet_model)