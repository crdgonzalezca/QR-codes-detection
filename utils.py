import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

def convert_x_y_to_top_left(img_file, x, y, w, h):
  left = (x - (w // 2))
  top = (y - (h // 2))
  return top, left

def load_annotations(use_negatives = False):
  annotation_files = _list_dir("/content/repo/dataset/", set(["txt"]))
  annotation_files = list(sorted(annotation_files))

  image_files = _list_dir("/content/repo/dataset/", set(["jpg", "JPG"]))
  image_files = list(sorted(image_files))

  total_negatives, total_images, total_boxes = 0, len(image_files), 0
  if(use_negatives):
    negative_annotation_files = _list_dir("/content/repo/negative_dataset/", set(["txt"]))
    annotation_files += list(sorted(negative_annotation_files))

    negative_image_files = load_negatives()
    total_negatives += len(negative_image_files)
    image_files += negative_image_files
    

  result = {}
  for f, img_file in zip(annotation_files, image_files):
    result[img_file] = []
    with open(f, 'r') as f:
      annotations = f.readlines()
      for annotation in annotations:
          _, x, y, w, h = map(float, annotation.strip().split(' '))
          total_boxes += 1
          result[img_file].append((x,y,w,h))
  print(f'Loaded {total_negatives} negative images, {total_images} images and {total_boxes} annotations.')
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
  return pd.DataFrame(data, columns=['image', 'width', 'height'])

def split_data(data, val_split):
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

def split_dataset(data, val_split):
  train, val = split_data(data, val_split)
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


def plot_cluster_predictions(clustering, X, y, n_clusters = None, cmap = plt.cm.plasma,
                             plot_data=True, plot_centers=True, show_metric=False,
                             title_str=""):

  figure = plt.figure(figsize=(10, 10))
  figure.savefig('anchors_kmeans.png', dpi=200)
  if plot_data:        
    plt.scatter(X[:,0], X[:,1], color=cmap((y*255./(n_clusters-1)).astype(int)), alpha=.5)
  if plot_centers:
    plt.scatter(clustering.cluster_centers_[:,0], clustering.cluster_centers_[:,1], s=150,  lw=3,
                  facecolor=cmap((np.arange(n_clusters)*255./(n_clusters-1)).astype(int)),
                  edgecolor="black")   

  if show_metric:
    if hasattr(clustering, 'inertia_'):
      inertia = clustering.inertia_
    else:
      inertia = 0
    sc = silhouette_score(X, y) if len(np.unique(y))>1 else 0
    plt.title("n_clusters %d, inertia=%.0f sc=%.3f"%(n_clusters, inertia, sc)+title_str)
  else:
    plt.title(title_str)
  plt.xlabel('Width')
  plt.ylabel('Height')

def scale_anchors(anchors, width, height):
  result = anchors * [width, height]
  return result.astype(int)
