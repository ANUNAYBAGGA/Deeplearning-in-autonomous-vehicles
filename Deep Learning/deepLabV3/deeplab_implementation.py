'''
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
'''

import collections
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import io
import sys
import tarfile
from grabscreen import grab_screen
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
# import skvideo.io
import tensorflow as tf
from skimage import measure


class DeepLabModel(object):
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    self.graph = tf.Graph()
    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break
    tar_file.close()
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')
    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.compat.v1.Session(graph=self.graph)
            
  def run(self, image):
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

def create_pascal_label_colormap():
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)
  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3
  return colormap

def label_to_color_image(label):
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')
  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')
  return colormap[label]

def add_text(image,string,pos):
    cv2.putText(image,string,(pos),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


#_TARBALL_NAME = 'deeplabv3_pascal_train_aug_2018_01_04.tar.gz'
#_TARBALL_NAME = 'deeplab_cityscapes_xception71.tar.gz'
_TARBALL_NAME = 'deeplabv3_mobilenet.tar.gz'
_FROZEN_GRAPH_NAME = 'frozen_inference_graph'

model = DeepLabModel(_TARBALL_NAME)
cap = cv2.VideoCapture(0)
colormap = create_pascal_label_colormap()
final = np.zeros((1, 384, 1026, 3))
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


while True:
    screenshot = grab_screen(region=(0,40,800,640))
    
    # From cv2 to PIL
    pil_im = Image.fromarray(screenshot)
    
    # Run model
    resized_im, seg_map = model.run(pil_im)
    
    # Adjust color of mask
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    '''
    #BBOX
    map_labeled = measure.label(seg_map, connectivity=1)
    for region in measure.regionprops(map_labeled):
      if region.area > 500:
          box = region.bbox
          p1 = (box[1], box[0])
          p2 = (box[3], box[2])
          screenshot = np.array(screenshot) 
          cv2.rectangle(screenshot, p1, p2, (77,255,9), 2)
          add_text(image,LABEL_NAMES[seg_map[tuple(region.coords[0])]],(p1[0],p1[1]-10))
    cv2.imshow('segmentation',image)
    '''
    # Convert PIL image back to cv2 and resize
    frame = np.array(pil_im)
    r = seg_image.shape[1] / frame.shape[1]
    dim = (int(frame.shape[0] * r), seg_image.shape[1])[::-1]
    #resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    #resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)   
    # Stack horizontally color frame and mask
    #color_and_mask = np.hstack((resized, seg_image))
    cv2.imshow('frame', seg_image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
