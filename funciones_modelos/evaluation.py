import os
import funciones_imagenes.image_funct as im
import funciones_imagenes.extra_functions as ex
import pandas as pd
import re
import numpy as np


def evaluate(model, path):
    file_names = ex.list_files(os.path.join(path, 'mascara'))
    masks = im.create_tensor(path, 'mascara', file_names, im.binarize)
    images = im.create_tensor(path, 'images', file_names, im.normalize)
    results = model.evaluate(images, masks, batch_size=8)
    print(results)
    return results


def save_eval(type, name, results, group=''):
    path = '/home/mr1142/Documents/Data/models/mascaras/validation_results/validation_results' + group + '.csv'
    df = pd.read_csv(path)
    save = [type, name] + results
    df.loc[len(df.index)] = save
    df.to_csv(path, index = False)


def all_evaluations(type, name, model, path = '/home/mr1142/Documents/Data/segmentation/splited/validation'):
    save_eval(type, name, evaluate(model, path))
    df = pd.read_csv('/home/mr1142/Documents/Data/segmentation/splited/validation/validation_data.csv')
    labels = [re.split('[|]', df['Finding Labels'][i]) for i in df.index]
    labels = [x for xs in labels for x in xs]
    labels = list(np.unique(labels))
    for lab in labels:
        index_contain = [i for i in df.index if bool(re.search(lab, df['Finding Labels'][i]))]
        images = list(df['Image Index'].iloc[index_contain])
        try:
            save_eval(type, name, evaluate(model,images), lab)
        except:
            print(f'{type}_{name} not save in label {lab}')
