import json
import re
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


_CLASSES_PATH = "/content/repo/model_data/classes.txt"
_SHARED_DRIVE_PATH = '/content/shared_drive/Inteligentes'
_LOGS_FOLDER = os.path.join(_SHARED_DRIVE_PATH, 'darknet_logs')
_DATASET_PATH = "/content/repo/dataset/"
_NEGATIVE_DATASET_PATH = "/content/repo/negative_dataset/"
_AUGMENTATION_DATASET_PATH = "/content/repo/augmentation_dataset/"
_TRAIN_FILE_PATH = '/content/repo/model_data/train.txt'
_VAL_FILE_PATH = '/content/repo/model_data/val.txt'
_TEST_FILE_PATH = '/content/repo/model_data/test.txt'

def _list_dir(path, extensions=None):
  '''
  params:
  command: str with the path to list
  '''
  files = os.listdir(path)
  if not extensions:
    return [os.path.join(path, file_path) for file_path in files]
  return [os.path.join(path, file_path) for file_path in files if file_path.split('.')[-1] in extensions]


def generate_csv(df, filename, splits):
  train, val, test = splits
  new_names = []
  for name in df['image_name']:
    if name in train:
      split = 'train'
    elif name in val:
      split = 'val'
    else:
      split = 'test'
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

def load_annotations(use_negatives=False, use_augmentation=False):
  annotation_files = _list_dir(_DATASET_PATH, set(["txt"]))
  annotation_files = list(sorted(annotation_files))

  image_files = _list_dir(_DATASET_PATH, {"jpg", "JPG"})
  image_files = list(sorted(image_files))

  total_negatives, total_images, total_boxes, total_augmentation = 0, len(image_files), 0, 0
  if(use_negatives):
    negative_annotation_files = _list_dir(_NEGATIVE_DATASET_PATH, set(["txt"]))
    annotation_files += list(sorted(negative_annotation_files))

    negative_image_files = load_negatives()
    total_negatives += len(negative_image_files)
    image_files += negative_image_files
    
  if(use_augmentation):
    aug_annotation_files = _list_dir(_AUGMENTATION_DATASET_PATH, {"txt"})
    annotation_files += list(sorted(aug_annotation_files))

    aug_image_files = _list_dir(_AUGMENTATION_DATASET_PATH, {'jpeg', 'PNG', 'png', 'jpg'})
    total_augmentation = len(aug_image_files)
    image_files += aug_image_files
  
  result = {}
  for f, img_file in zip(annotation_files, image_files):
    result[img_file] = []
    with open(f, 'r') as f:
      annotations = f.readlines()
      for annotation in annotations:
          _, x, y, w, h = map(float, annotation.strip().split(' '))
          total_boxes += 1
          result[img_file].append((x,y,w,h))
  print(f'Loaded {total_negatives} negative images, {total_augmentation} augmentation images, {total_images} images and {total_boxes} annotations.')
  return result

def load_negatives():
  negatives_files = _list_dir(_NEGATIVE_DATASET_PATH, {'jpeg', 'png', 'jpg'})
  return list(sorted(negatives_files))

def load_images_sizes(image_files):
  data = []
  for img_file in image_files:
    img = cv2.imread(img_file)
    width, height, _ = img.shape
    data.append([img_file, width, height])
  return pd.DataFrame(data, columns=['image', 'width', 'height'])

def split_data(data, val_split, test_split):
  np.random.seed(10101)
  np.random.shuffle(data)
  np.random.seed(None)
  data_size = len(data)
  num_val = int(data_size * val_split)
  num_test = int(data_size * test_split)
  num_train = data_size - num_val - num_test
  return data[:num_train], data[num_train: num_train + num_val], data[num_train + num_val:]

def gen_dataset_file(outfile, data):
  with open(outfile, 'w') as f:
    for value in data:
      f.write(f'{value}\n')

def split_dataset(data, val_split, test_split):
  train, val, test = split_data(data, val_split, test_split)
  gen_dataset_file(_TRAIN_FILE_PATH, train)
  gen_dataset_file(_VAL_FILE_PATH, val)
  gen_dataset_file(_TEST_FILE_PATH, test)
  return train, val, test

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


def get_loss_from_logs(text):
  lines = text.split('\n')
  loss_regex = r'([0-9]+?): [-+]?[0-9]*\.?[0-9]+, ([-+]?[0-9]*\.?[0-9]+?) avg loss.*'
  losses = []
  expected_epoch = 1
  for line in lines:
    line = line.strip()
    if re.match(loss_regex, line):
      search = re.search(loss_regex, line)
      epoch = int(search.group(1))
      loss = float(search.group(2))
      if expected_epoch != epoch: # Identify left epochs in logs
        print(f'\tExpected {expected_epoch} but got {epoch}')
        for e in range(expected_epoch ,epoch):
          losses.append(losses[-1])
        expected_epoch = epoch
      losses.append(loss)
      expected_epoch += 1
  return losses

def get_map_from_logs(text):
  lines = text.split('\n')
  map_regex = 'mean_average_precision \(mAP@0\.5\) = ([0-9]*\.?[0-9]+ ?)'
  loss_regex = r'([0-9]+?): [-+]?[0-9]*\.?[0-9]+, ([-+]?[0-9]*\.?[0-9]+?) avg loss.*'
  epoch = 0
  maps = []
  epochs = []
  for line in lines:
    line = line.strip()
    if re.match(loss_regex, line):
      epoch = int(re.search(loss_regex, line).group(1))
    if re.match(map_regex, line):
      ap = float(re.search(map_regex, line).group(1)) * 100
      maps.append(ap)
      epochs.append(epoch)
  return maps, epochs

def parse_logs(ignore_logs_from=[]):
  experiments = map(lambda x: x.split('/')[-1], _list_dir(_LOGS_FOLDER))
  losses = {}
  maps = {}
  maps_epochs = {}
  for experiment in experiments:
    if experiment in ignore_logs_from:
      continue
    logs_name = 'loss_tiny.txt'
    logs_path = os.path.join(_LOGS_FOLDER, experiment, logs_name)
    text = ''
    with open(logs_path, 'r') as f:
      text = f.read()
    print(f'Loading logs from {experiment}.')
    exp_losses = get_loss_from_logs(text)
    exp_maps, exp_maps_epochs = get_map_from_logs(text)
    maps[experiment] = exp_maps
    maps_epochs[experiment] = exp_maps_epochs
    losses[experiment] = exp_losses
  return losses, [maps, maps_epochs]

def get_metrics_from_results(text):
  lines = text.split('\n')
  recall_precision_regex = r'\[(.)*\]'
  ap_regex = 'AP: ([-+]?[0-9]*\.?[0-9]+?)%'
  recall_precision = []
  precision = []
  recall = []
  avg_precision = 0.0

  for line in lines:
    line = line.strip()
    if re.match(ap_regex, line):
      avg_precision = float(re.search(ap_regex, line).group(1))
    for match in re.finditer(recall_precision_regex, line):
      s = match.start()
      e = match.end()
      recall_precision.append(line[s:e])

  for i, x in enumerate(recall_precision):
    for z in x[1:len(x) - 1].split(","):
      val = float(z.split("'")[1])
      if (i == 0):  precision.append(val)
      else: recall.append(val)
  return precision, recall, avg_precision

def get_experiments_metrics(dataset_name, metrics_path):
  experiments_path = _list_dir(metrics_path, {'txt'})
  recall_x_precision = {}
  maps = {}
  precisions = {}
  recalls = {}
  avg_precisions = {}
  
  for experiment_path in experiments_path:
    name_experiment = "_".join(experiment_path.split("/")[-1].split(".")[0].split("_")[1:])
    if  name_experiment.split('_')[0] != dataset_name:
      continue
    with open(experiment_path, 'r') as f:
      text = f.read()
    precision, recall, avg_precision = get_metrics_from_results(text)
    precisions[name_experiment] = precision
    recalls[name_experiment] = recall
    avg_precisions[name_experiment] = avg_precision
  return precisions, recalls, avg_precisions

def read_json(path):
  with open(path, 'r') as f:
    text = f.read()
    return json.loads(text)