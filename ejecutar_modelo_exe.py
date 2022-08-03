import os
import re
import argparse
import cv2
import tensorflow.keras as keras


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--device',
                        help="GPU device",
                        type=str,
                        default=2)
    parser.add_argument('-m',
                        '--modelo',
                        type=str,
                        default='unet_final_renacimiento_validation_6.h5',
                        help="nombre del modelo (incluyendo extension)")
    parser.add_argument('-vp',
                        '--validation_path',
                        type=str,
                        default='/home/mr1142/Documents/Data/validacion_medica/NIH',
                        help="path with the validation set")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    modelo = args.modelo
    path = args.validation_path
    import funciones_imagenes.mask_funct as msk
    import funciones_imagenes.extra_functions as ex
    import funciones_modelos.unet_doble_loss as u_loss

    model = os.path.join('./modelos', modelo)
    model = keras.models.load_model(model, 
                                        custom_objects={"MyLoss": u_loss.MyLoss, 
                                                        "loss_mask": u_loss.loss_mask, 
                                                        "dice_coef_loss": ex.dice_coef_loss,
                                                        "dice_coef": ex.dice_coef})

    if not os.path.exists(os.path.join(path, 'mascara')):
        os.makedirs(os.path.join(path, 'mascara'))

    images = [f for f in os.listdir(path) if bool(re.search('png', f))]
    for image in images:
        img = cv2.imread(os.path.join(path, image))
        segmented = msk.des_normalize(msk.apply_mask(img, model))    
        cv2.imwrite(os.path.join(path,'mascara', image), segmented)