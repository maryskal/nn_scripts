import argparse
import os
import funciones_modelos.logs as logs
import tensorflow as tf
import funciones_modelos.evaluation as ev

def modelo(fine_tune_at = 19):
    model_path = '/home/mr1142/Documents/Data/models/unsupervised.h5'
    backbone = tf.keras.models.load_model(model_path)
    inputs = backbone.input
    x1 = backbone.layers[38].output
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation = "sigmoid", name = 'conv_salida')(x1)
    model = tf.keras.Model(inputs, outputs, name="U-Net")

    backbone.trainable = True
    print('\ntrainable variables: {}'.format(len(backbone.trainable_variables)))

    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-n',
                        '--name',
                        type=str,
                        default='new',
                        help="name of the model")                               

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    name = args.name
    path = '/home/mr1142/Documents/Data/old_segmentation/splited/train'
    batch = 8
    epoch = 200
    pixels = 256

    #----------------------------------------------------
    import funciones_imagenes.image_funct as im
    import funciones_imagenes.extra_functions as ex
    import funciones_modelos.unet_doble_loss as u_loss

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

    metrics = [ex.dice_coef_loss, u_loss.loss_mask, 'accuracy', 'AUC',
                tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

    unet_model = modelo(10)
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                        loss=ex.dice_coef_loss,
                        metrics=metrics)
    
    callb = [logs.tensorboard(name), logs.early_stop(7)]

    history = unet_model.fit(images,masks,
                            batch_size = batch,
                            epochs = epoch,
                            callbacks= callb,
                            shuffle = True,
                            validation_split = 0.2)        

    unet_model.save('/home/mr1142/Documents/Data/models/mascaras/' + name + '.h5')
    
    # EVALUACIÓN
    ev.all_evaluations('fine', name, unet_model)
    ev.all_evaluations('patologic_' + name, unet_model, '/home/mr1142/Documents/Data/patologic')
