#!/usr/bin/env python
"""
Created on 26/01/21
Author: Davide Rigoni
Emails: davide.rigoni.2@phd.unipd.it - drigoni@fbk.eu
Description: this file contains code from: https://github.com/peteanderson80/bottom-up-attention needed
in order to give some examples of the object detector using the flickr30k images.
"""

import matplotlib
matplotlib.use('Agg')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from model_code.utils import Timer
import matplotlib.pyplot as plt
import numpy as np
import caffe, os, cv2
import argparse

CLASSES = ['__background__']
with open(os.path.join(cfg.DATA_DIR, 'vg/objects_vocab.txt')) as f:
  for object in f.readlines():
    CLASSES.append(object.lower().strip())
    
ATTRS = []
with open(os.path.join(cfg.DATA_DIR, 'vg/attributes_vocab.txt')) as f:
  for attr in f.readlines():
    ATTRS.append(attr.lower().strip())
    
RELATIONS = []
with open(os.path.join(cfg.DATA_DIR, 'vg/relations_vocab.txt')) as f:
  for rel in f.readlines():
    RELATIONS.append(rel.lower().strip())    

NETS = ['VGG', 'ResNet-101']

MODELS = [
  'faster_rcnn_end2end',
  'faster_rcnn_end2end_attr',
  'faster_rcnn_end2end_attr_rel',
  'faster_rcnn_end2end_attr_rel_softmax_primed',
  'faster_rcnn_end2end_attr_softmax_primed',
  'faster_rcnn_end2end_final',
]  


def vis_detections(ax, class_name, dets, attributes, rel_argmax, rel_score, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, 4]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
            
        if attributes is not None:
            att = np.argmax(attributes[i])
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f} ({:s})'.format(class_name, score, ATTRS[att]),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
        else:
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
                    
        #print class_name
        #print 'Outgoing relation: %s' % RELATIONS[np.argmax(rel_score[i])]

    ax.set_title(('detections with '
                  'p(object | box) >= {:.1f}').format(thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo_tuples(net, image_name):
    """Detect objects, attributes and relations in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)
    if attr_scores is not None:
        print 'Found attribute scores'
    if rel_scores is not None:
        print 'Found relation scores'
        rel_scores = rel_scores[:,1:] # drop no relation
        rel_argmax = np.argmax(rel_scores, axis=1).reshape((boxes.shape[0],boxes.shape[0]))
        rel_score = np.max(rel_scores, axis=1).reshape((boxes.shape[0],boxes.shape[0]))
        
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])    

    # Visualize detections for each class
    CONF_THRESH = 0.2
    NMS_THRESH = 0.2
    ATTR_THRESH = 0.2
    
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im)
    
    # Detections
    det_indices = []
    det_scores = []
    det_objects = []
    det_bboxes = []
    det_attrs = []
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, NMS_THRESH))
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        
        if len(inds) > 0:
            keep = keep[inds]
            for k in keep:
                det_indices.append(k)
                det_bboxes.append(cls_boxes[k])
                det_scores.append(cls_scores[k])
                det_objects.append(cls)
                if attr_scores is not None:
                    attr_inds = np.where(attr_scores[k][1:] >= ATTR_THRESH)[0]
                    det_attrs.append([ATTRS[ix] for ix in attr_inds])
                else:
                    det_attrs.append([])
        
    for i,(idx,score,obj,bbox,attr) in enumerate(zip(det_indices,det_scores,det_objects,det_bboxes,det_attrs)):
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        box_text = '{:s} {:.3f}'.format(obj, score)
        if len(attr) > 0:
            box_text += "(" + ",".join(attr) + ")"
        ax.text(bbox[0], bbox[1] - 2,
                box_text,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        if rel_scores is not None:
            rel_score = rel_score[det_indices].T[det_indices].T
            rel_argmax = rel_argmax[det_indices].T[det_indices].T
            # Outgoing
            score = np.max(rel_score[i])
            ix = np.argmax(rel_score[i])
            subject = det_objects[ix]
            relation = RELATIONS[rel_argmax[i][ix]]
            print 'Relation: %.2f %s -> %s -> %s' % (score, obj, relation, subject)
            # Incoming
            score = np.max(rel_score.T[i])
            ix = np.argmax(rel_score.T[i])
            subject = det_objects[ix]
            relation = RELATIONS[rel_argmax[ix][i]]
            print 'Relation: %.2f %s -> %s -> %s' % (score, subject, relation, obj)        

    ax.set_title(('detections with '
                  'p(object|box) >= {:.1f}').format(CONF_THRESH),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()    
    plt.savefig('data/demo/'+im_file.split('/')[-1].replace(".jpg", "_demo.jpg"))    
    
def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)
    #print 'relations'
    #print rel_scores.shape
    #rel_argmax = np.argsort(rel_scores, axis=1).reshape((boxes.shape[0],boxes.shape[0]))
    #rel_score = np.max(rel_scores, axis=1).reshape((boxes.shape[0],boxes.shape[0]))
    #print rel_argmax.shape
    #print rel_score.shape
    #print np.min(rel_score)
    #print np.max(rel_score)
    #np.savetxt('rel_score.csv', rel_score, delimiter=',')
    #np.savetxt('rel_argmax.csv', rel_argmax, delimiter=',')
    #print fail
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.4
    NMS_THRESH = 0.3
    
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if attr_scores is not None:
            attributes = attr_scores[keep]
        else: 
            attributes = None
        if rel_scores is not None:
            rel_argmax_c = rel_argmax[keep]
            rel_score_c = rel_score[keep]
        else:
            rel_argmax_c = None
            rel_score_c = None
        vis_detections(ax, cls, dets, attributes, rel_argmax_c, rel_score_c, thresh=CONF_THRESH)
    plt.savefig('data/demo/'+im_file.split('/')[-1].replace(".jpg", "_demo.jpg"))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='net', help='Network to use, e.g. VGG16',
                        choices=NETS, default='VGG16')
    parser.add_argument('--model_code', dest='model_code', help='Model to use, e.g. faster_rcnn_end2end',
                        choices=MODELS, default='faster_rcnn_end2end_attr_rel_softmax_primed')

    args = parser.parse_args()

    return args    
    
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        
    args = parse_args()
    
    prototxt = os.path.join(cfg.ROOT_DIR, 'models/vg', args.net, args.model, 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', 'resnet101_faster_rcnn_final.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _, _, _= im_detect(net, im)

    im_names = ['demo/000456.jpg', 
                'demo/000542.jpg', 
                'demo/001150.jpg',
                'demo/001763.jpg', 
                'demo/004545.jpg',
                'demo/65567.jpg',
                'demo/81641.jpg',
              ]
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo_tuples(net, im_name)

    plt.show()
