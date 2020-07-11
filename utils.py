import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

_CLASSES_PATH = "/content/repo/model_data/classes.txt"


def _list_dir(path, extensions):
  '''
  params:
  command: str with the path to list
  '''
  files = os.listdir(path)
  return [os.path.join(path, file_path) for file_path in files if file_path.split('.')[-1] in extensions]


def generate_csv(df, filename, splits):
  train, val = splits
  new_names = []
  for name in df['image_name']:
    if name in train:
      split = 'train'
    else:
      split = 'val'
    new_names.append(f'./{split}/{name}')
  copy = df.copy()
  copy['image_name'] = new_names
  copy.sort_values('image_name', inplace=True)
  copy.to_csv(filename, index=False)

def generate_annotations_df(annotations):
    cols = ['image_name', 'x', 'y', 'width', 'height']
    df_data = []
    for file_name, boxes in annotations.items():
        for annotation in boxes:
            df_data.append([file_name.split('/')[-1]] + list(resize_annotation(file_name, *annotation)))

    return pd.DataFrame(df_data, columns=cols)

def resize_annotation(img_file, x, y, w, h):
    img = cv2.imread(img_file)
    width, height, _ = img.shape
    new_x = int(x * width)
    new_y = int(y * height)
    new_w = int(w * width)
    new_h = int(h * height)
    return new_x, new_y, new_w, new_h

def load_annotations():
  annotation_files = _list_dir('/content/repo/dataset/', {'txt'})
  annotation_files = list(sorted(annotation_files))
  image_files = _list_dir('/content/repo/dataset/', {'jpg', 'JPG'})
  image_files = list(sorted(image_files))
  result = {}
  total_images, total_boxes = 0, 0 
  for f, img_file in zip(annotation_files, image_files):
      result[img_file] = []
      total_images += 1
      with open(f, 'r') as f:
          annotations = f.readlines()
          for annotation in annotations:
              _, x, y, w, h = map(float, annotation.strip().split(' '))
              result[img_file].append((x,y,w,h))
              total_boxes += 1
  print(f'Loaded {total_images} images and {total_boxes} annotations.')
  return result

def load_negatives():
  negatives_files = _list_dir('/content/repo/negative_dataset/', {'jpeg', 'png', 'jpg'})
  return list(sorted(negatives_files))

def load_images_sizes(image_files):
  data = []
  for img_file in image_files:
    img = cv2.imread(img_file)
    width, height, _ = img.shape
    data.append([img_file, width, height])
  return pd.DataFrame(data, columns=['file', 'width', 'height'])

def split_data(data, val_split, use_negatives=False):
  if use_negatives:
    data = data + load_negatives()
  np.random.seed(10101)
  np.random.shuffle(data)
  np.random.seed(None)
  data_size = len(data)
  num_val = int(data_size * val_split)
  num_train = data_size - num_val
  return data[:num_train], data[num_train:]

def gen_dataset_file(outfile, data):
  with open(outfile, 'w') as f:
    for value in data:
      f.write(f'{value}\n')

def split_dataset(data, val_split, use_negatives=False):
  train, val = split_data(data, val_split, use_negatives)
  gen_dataset_file('/content/repo/model_data/train.txt', train)
  gen_dataset_file('/content/repo/model_data/val.txt', val)
  return train, val

def imShow(path):
  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()