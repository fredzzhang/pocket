"""
Visualization for HICODet dataset

Written by Frederic Zhang
Australian National University

Last updated in Jun. 2019
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from .visualizer import ImageVisualizer

class HICODetVisualizer(ImageVisualizer):
    """
    Visualization for HicoDet dataset

    Arguments:
        imdir(str): Directory of source images
        db(DataDict): A data dict with required fields
            db.image_id (ndarray) (N,)
            db.box_h (ndarray) (N,4)
            db.box_o (ndarray) (N,4)
            db.scores (ndarray) (N,81)
            db.joints (ndarray) (N,17,[x,y])
            db.labels (scipy sparse matrix) (N,600)
            db.per_image (ndarray) (M,2)
        mode(str): A choice between 'IMAGE' and 'BOX'
            'IMAGE' mode manages data by images
            'BOX' mode manages data by box pairs
    """
    def __init__(self, imdir, db, hoi_list, mode='IMAGE'):
        super(HICODetVisualizer, self).__init__(imdir)
        assert mode in ['IMAGE', 'BOX'], \
                'Invalid mode \'{}\', choose between \'IMAGE\' and \'BOX\''.\
                format(mode)
        assert len(self._im_paths) == len(db.per_image), \
                'Number of images ({}) does not match the stats in database ({})'.\
                format(len(self._im_paths), len(db.per_image))
        self._mode = mode
        self._db = db
        self._hoi_list = hoi_list

        # Joint connections to form human skeleton
        self._skeleton = np.array([
            [15,13],[13,11],[16,14],[14,12],
            [11,12],[5,11],[6,12], [5,6],[5,7],
            [6,8],[7,9],[8,10],[1,2],[0,1],
            [0,2],[1,3],[2,4],[3,5],[4,6]
            ])

    def __len__(self):
        """Return the number of images or boxes based on selected mode"""
        if self._mode == 'IMAGE':
            return len(self._db.per_image)
        elif self._mode == 'BOX':
            return len(self._db.box_h)
        else:
            raise ValueError

    @property
    def mode(self):
        """Return the data management mode"""
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        """Change the data management mode"""
        self._mode = new_mode

    def show(self, i):
        """Display bounding boxes based on selected mode"""
        if self._mode == 'IMAGE':
            self._show_by_image(i)
        elif self._mode == 'BOX':
            self._show_by_box(i)
        else:
            raise ValueError('Unsupported mode {}'.format(self._mode))

    def _show_by_image(self, i):
        """Display bouning boxes within an image interactively"""
        im = cv2.imread(self._im_paths[i])
        start_i = self._db.per_image[i, 0]
        num_pairs = self._db.per_image[i, 1]
        # loop to ask for the box pair to be displayed
        while(1):
            print('\nTHERE ARE {} BOX PAIRS IN IMAGE [{}]\
                    \nCHOOSE ONE USING ZERO-BASED INDEX: '.\
                    format(num_pairs, i))
            box_id = int(input())
            # return upon invalid index
            if box_id not in range(num_pairs):
                print('\nBREAK LOOP DUE TO INVALID INDEX\n')
                return
            
            box_h = self._db.box_h[start_i + box_id, :].astype(np.int32)
            box_o = self._db.box_o[start_i + box_id, :].astype(np.int32)
            labels = self._db.labels[start_i + box_id, :].toarray()
            scores = self._db.scores[start_i + box_id, :]
            joints = self._db.joints[start_i + box_id, :, :]
            hois = self._hoi_list[np.where(labels)[1], :]

            self._display(im.copy(), box_h, box_o, scores, joints, hois)
            
    def _show_by_box(self, i):
        """Display the selceted box pair in the corresponding image"""
        image_id = self._db.image_id[i]
        im = cv2.imread(self._im_paths[image_id])
        box_h = self._db.box_h[i, :].astype(np.int32)
        box_o = self._db.box_o[i, :].astype(np.int32)
        labels = self._db.labels[i, :].toarray()
        scores = self._db.scores[i, :]
        joints = self._db.joints[i, :, :]
        hois = self._hoi_list[np.where(labels)[1], :]

        self._display(im, box_h, box_o, scores, joints, hois)
    
    def _display(self, im, box_h, box_o, scores, joints, hois):
        """Draw the box pairs onto the image and print labels"""
        # draw boxes
        cv2.rectangle(im, (box_h[0], box_h[1]), (box_h[2], box_h[3]),
                (255, 0, 0))
        cv2.rectangle(im, (box_o[0], box_o[1]), (box_o[2], box_o[3]),
                (0, 255, 0))
        # draw human skeleton
        for i in range(self._skeleton.shape[0]):
            cv2.line(im, 
                        (int(joints[self._skeleton[i, 0], 0]), int(joints[self._skeleton[i, 0], 1])), 
                        (int(joints[self._skeleton[i, 1], 0]), int(joints[self._skeleton[i, 1], 1])),
                        (0, 0, 255), 1)
        # print object confidence scores
        # clip the y coord. to make sure the text is visible
        cv2.putText(im, '{:.2f}'.format(scores[0]), (box_h[0], np.max([box_h[1], 20])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(im, '{:.2f}'.format(np.max(scores[1:])), (box_o[0], np.max([box_o[1], 20])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # show the image
        cv2.imshow('Box pair', im)
        cv2.waitKey(1)

        # print labels
        print('\nLABELS FOR CURRENT BOX PAIR:\n')
        for j in range(len(hois)):
            print('{} {}'.format(hois[j, 0].item(), hois[j, 1].item()))

