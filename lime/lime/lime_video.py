"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from tqdm.auto import tqdm
import cv2

from . import lime_base


class VideoExplanation(object):
    def __init__(self, video, resized_video, segments, resized_segments, segmentation_mode):
        """Init function.

        Args:
            video: 4d numpy array, used for displaying results
            resized_video: resized 4d numpy array, used for faster computation
            segments: 3d numpy array representing different regions for the video
            resized_segments: 3d numpy array representing different regions for the resized video
            segmentation_mode: type of segmentation to be used, either "space", "time-blackened", "time-clipped",
                "time-freeze" or "space-time"
        """
        self.video = video
        self.resized_video = resized_video
        self.segments = segments
        self.resized_segments = resized_segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}
        self.segmentation_mode = segmentation_mode

    def get_score_map(self, label, th=None, top_k=None):
        """ Computes a score map representing the importance of each region from the local explanations.

        Args:
            label: class to be explained
            th: minimum threshold to preserve the score of a region
            top_k: preserve only top k regions. None to preserve all of them.

        Returns:
            3d numpy array representing the importance of each region
        """
        score_map = np.zeros(self.video.shape[:3], dtype='float16')
        for id_seg, score in self.local_exp[label]:
            if self.segmentation_mode == "space":
                score_map[:, self.segments == id_seg] = score
            elif self.segmentation_mode.startswith("time"):
                score_map[self.segments == id_seg, :, :] = score
            elif self.segmentation_mode == "space-time":
                score_map[self.segments == id_seg] = score

        if th is not None:
            score_map[np.abs(score_map) < th] = 0
        if top_k is not None:
            th_value = np.unique(np.abs(score_map))[-top_k]
            score_map[np.abs(score_map) < th_value] = 0

        return score_map

    def get_score_map_rgb(self, score_map, hist_stretch=True, invert=True):
        """ Given a video and explanations score map, it merges them into one sole video.

        Args:
            score_map: explanations, with same shape as video, except for channels dimension
            hist_stretch: whether to perform histogram stretching to see better low intensity scores
            invert: True to use white as minimum score or False to use black

        Returns:
            Masked video, containing both input video and explanations
        """
        return LimeVideoExplainer.score_map_rgb(self.video, score_map, hist_stretch, invert)

    def get_masked_video(self, score_map_rgb, alpha=0.5):
        """ Put the explanation mask to the video, with a given transparency.

        Args:
            score_map_rgb: input mask
            alpha: transparency of mask

        Returns:
            Masked video
        """
        return LimeVideoExplainer.mask_video(self.video, score_map_rgb, alpha)

    def get_video_and_mask(self, label, positive_only=False, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (video, mask), where video is a 4d numpy array and mask is a 3d
            numpy array that can be used with skimage.segmentation.mark_boundaries
        """

        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")

        segments = self.segments
        video = self.video
        exp = self.local_exp[label]
        mask = np.zeros(video.shape[:3], segments.dtype)

        if hide_rest:
            temp = np.zeros(video.shape)
        else:
            temp = video.copy()

        if positive_only or negative_only:
            if positive_only:
                fs = [f for f, w in exp if w >= min_weight][:num_features]
            else:
                fs = [f for f, w in exp if w <= -min_weight][:num_features]

            mask_f = np.isin(segments, fs)

            if self.segmentation_mode == "space":
                temp[:, mask_f, :] = video[:, mask_f, :].copy()
                mask[:, mask_f] = 1
            elif self.segmentation_mode.startswith("time"):
                temp[mask_f, :, :, :] = video[mask_f, :, :, :].copy()
                mask[mask_f, :, :] = 1
            elif self.segmentation_mode == "space-time":
                temp[mask_f, :] = video[mask_f, :].copy()
                mask[mask_f] = 1

            return temp, mask
        else:
            # Positive mask
            fs_pos = [f for f, w in exp if w >= min_weight][:num_features]
            mask_pos = np.isin(segments, fs_pos)

            # Negative mask
            fs_neg = [f for f, w in exp if w <= -min_weight][:num_features]
            mask_neg = np.isin(segments, fs_neg)

            temp = video.copy()

            # Set mask
            if self.segmentation_mode == "space":
                mask[:, mask_pos] = 1
                mask[:, mask_neg] = -1
                temp[:, mask_pos, 1] = np.max(video)
                temp[:, mask_neg, 0] = np.max(video)
            elif self.segmentation_mode.startswith("time"):
                mask[mask_pos, :, :] = 1
                mask[mask_neg, :, :] = -1
                temp[mask_pos, :, :, 1] = np.max(video)
                temp[mask_neg, :, :, 0] = np.max(video)
            elif self.segmentation_mode == "space-time":
                mask[mask_pos] = 1
                mask[mask_neg] = -1
                temp[mask_pos, 1] = np.max(video)
                temp[mask_neg, 0] = np.max(video)

            return temp, mask


class LimeVideoExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, in_video, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5,
                         num_samples=1000,
                         batch_size=10,
                         distance_metric='cosine',
                         model_regressor=None,
                         progress_bar=True,
                         resize_scale=None,
                         segmentation_mode="space",
                         seg_width_slices = 10,
                         seg_height_slices= 10,
                         seg_time_slices = 10):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            in_video: input video to be explained
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: If not None, will hide superpixels with this color.
                Otherwise, use the mean pixel color of the image.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_samples: size of the neighborhood to learn the linear model
            batch_size: batch size for model predictions
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            resize_scale: scale to which rescale input video to be explained, for faster computation. None for no
                resizing
            progress_bar: if True, show tqdm progress bar.
            segmentation_mode: type of segmentation to perform. Available options are: "space", "time-blackened",
                "time-clipped" and "time-freeze"
            seg_width_slices: number of desired regions along X space dimension
            seg_height_slices: number of desired regions along Y space dimension
            seg_time_slices: number of desired regions along time dimension

        Returns:
            A VideoExplanation object with the corresponding explanations.
        """
        segments = LimeVideoExplainer.segmentation(in_video.shape, segmentation_mode, seg_time_slices,
                                                   seg_height_slices, seg_width_slices)
        if resize_scale is not None:
            resized_video = LimeVideoExplainer.resize_video(in_video, resize_scale, resize_scale)
            resized_segments = LimeVideoExplainer.segmentation(resized_video.shape, segmentation_mode, seg_time_slices,
                                                               seg_height_slices, seg_width_slices)
        else:
            resized_video = in_video
            resized_segments = segments

        num_features = np.unique(resized_segments).shape[0]

        fudged_video = resized_video.copy()
        if hide_color is None:
            for x in np.unique(resized_segments):
                fudged_video[resized_segments == x] = (
                    np.mean(resized_video[resized_segments == x][:, 0]),
                    np.mean(resized_video[resized_segments == x][:, 1]),
                    np.mean(resized_video[resized_segments == x][:, 2]))
        else:
            fudged_video[:] = hide_color

        top = labels

        data, labels = self.data_labels(resized_video, fudged_video, resized_segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar,
                                        mode=segmentation_mode)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = VideoExplanation(in_video, resized_video, segments, resized_segments, segmentation_mode)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            if segmentation_mode != "time-clipped" and segmentation_mode != "time-freeze":
                (ret_exp.intercept[label],
                 ret_exp.local_exp[label],
                 ret_exp.score[label],
                 ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                    data, labels, distances, label, num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)
            else:
                ret_exp.intercept[label] = None
                ret_exp.score[label] = None
                ret_exp.local_pred[label] = labels[0, ret_exp.top_labels]
                scores = (labels[:, ret_exp.top_labels]) * 2 - 1
                ret_exp.local_exp[label] = [(i, score[0]) for i, score in zip(range(scores.shape[0]), scores)]
        return ret_exp

    def data_labels(self,
                    in_video,
                    fudged_video,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True,
                    mode="space"):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            in_video: 4d numpy array, the video
            fudged_video: 4d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the video
            classifier_fn: function that takes a list of videos and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.
            mode: dimension along which perform segmentation. Available options
            are "space", "time" and "space-time". Default "space".

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """

        n_features = np.unique(segments).shape[0]
        if mode == "time-clipped" or mode == "time-freeze":
            data = np.zeros(shape=(n_features+1, n_features), dtype=int)
            for i in range(data.shape[1]):
                data[i+1, i] = 1
        else:
            data = np.random.randint(0, 2, num_samples * n_features) \
                .reshape((num_samples, n_features))
        data[0, :] = 1
        videos = []
        labels = []
        rows = tqdm(data) if progress_bar else data

        for row in rows:
            temp = copy.deepcopy(in_video)
            zeros = np.where(row == 0)[0]
            mask = np.isin(segments, zeros)

            if type(fudged_video) == int:
                if mode == "space":
                    temp[:, mask, :] = fudged_video
                elif mode == "time-blackened":
                    temp[mask, :, :, :] = fudged_video
                elif mode == "time-clipped":
                    temp = temp[np.logical_not(mask), :, :, :]
                elif mode == "time-freeze":
                    temp = np.repeat(temp[np.logical_not(mask), :, :, :][:1, :, :, :], in_video.shape[0], axis=0)
                else:  # space-time
                    temp[mask, :] = fudged_video
            else:
                if mode == "space":
                    temp[:, mask, :] = fudged_video[:, mask, :]
                elif mode == "time-blackened":
                    temp[mask, :, :, :] = fudged_video[mask, :, :, :]
                elif mode == "time-clipped":
                    temp = temp[np.logical_not(mask), :, :, :]
                elif mode == "time-freeze":
                    temp = np.repeat(temp[np.logical_not(mask), :, :, :][:1, :, :, :], in_video.shape[0], axis=0)
                else:  # space-time
                    temp[mask, :] = fudged_video[mask, :]

            videos.append(temp)

            if len(videos) == batch_size:
                preds = classifier_fn(videos)
                labels.extend(preds)
                videos = []

        if len(videos) > 0:
            preds = classifier_fn(videos)
            labels.extend(preds)

        return data, np.array(labels)

    @staticmethod
    def segmentation(video_shape, mode, n_slices, h_n, w_n):
        """ Segmentate the video along different chosen dimensions, according to mode.

        Args:
            video_shape: shape of the input video
            mode: type of segmentation. Either "space", "time-blackened", "time-clipped", "time-freeze" or "space-time"
            n_slices: Number of regions along time dimension
            h_n: Number of regions along Y dimension
            w_n: Number of regions along X dimension

        Returns:
            numpy array of video_shape with segmentated regions
        """
        if mode == "space":
            return LimeVideoExplainer.space_segmentation(video_shape, h_n, w_n)
        elif mode == "time-blackened" or mode == "time-clipped" or mode == "time-freeze":
            return LimeVideoExplainer.time_segmentation(video_shape, n_slices)
        elif mode == "space-time":
            return LimeVideoExplainer.space_time_segmentation(video_shape, n_slices, h_n, w_n)
        else:
            raise NotImplementedError

    @staticmethod
    def space_time_segmentation(video_shape, n_slices, h_n, w_n):
        """ Segmentate the video along space and time dimensions, according to mode.

        Args:
            video_shape: shape of the input video
            n_slices: Number of regions along time dimension
            h_n: Number of regions along Y dimension
            w_n: Number of regions along X dimension

        Returns:
            numpy array of video_shape with segmentated regions
        """
        seg = np.zeros(shape=video_shape[:3], dtype=int)
        space_seg = LimeVideoExplainer.space_segmentation(video_shape, h_n, w_n)
        slice_size = video_shape[0] / n_slices
        for t in range(n_slices):
            seg[int(t * slice_size):int((t + 1) * slice_size), :, :] = space_seg + h_n * w_n * t
        return seg

    @staticmethod
    def time_segmentation(video_shape, n_slices):
        """ Segmentate the video along time dimension.

        Args:
            video_shape: shape of the input video
            n_slices: Number of regions along time dimension

        Returns:
            numpy array of video_shape with segmentated regions
        """
        res = np.zeros(video_shape[0], dtype=int)
        slice_size = video_shape[0] / n_slices
        for i in range(n_slices):
            res[int(i * slice_size):int((i + 1) * slice_size)] = i
        return res

    @staticmethod
    def space_segmentation(video_shape, h_n, w_n):
        """ Segmentate the video along space dimensions.

        Args:
            video_shape: shape of the input video
            h_n: Number of regions along Y dimension
            w_n: Number of regions along X dimension

        Returns:
            numpy array of video_shape with segmentated regions
        """
        res = np.zeros(video_shape[1:3], dtype=int)
        h_size = video_shape[1] / h_n
        w_size = video_shape[2] / w_n
        seg_id = 0
        for y in range(1, h_n + 1):
            for x in range(1, w_n + 1):
                res[int((y - 1) * h_size):int(y * h_size), int((x - 1) * w_size):int(x * w_size)] = seg_id
                seg_id += 1
        return res

    @staticmethod
    def resize_video(video, scale_x, scale_y):
        """ Resize video to new height and width.

        Args:
            video: input video
            scale_x: new video width
            scale_y: new video height

        Returns:
            Resized video
        """
        new_w = int(video.shape[2] * scale_x)
        new_h = int(video.shape[1] * scale_y)
        video_out = np.zeros(shape=(video.shape[0], new_h, new_w, 3), dtype='uint8')

        for i in range(video.shape[0]):
            frame = video[i, :, :, :]
            video_out[i, :, :, :] = cv2.resize(frame, dsize=(new_w, new_h))

        return video_out

    @staticmethod
    def painter(img1, img2, alpha2):
        """ Merges two images into one, according to alpha factor.

        Args:
            img1: 1st image
            img2: 2nd image
            alpha2: Importance of 1st image. 1 maximum, 0 minimum.

        Returns:
            Merged images.
        """
        return (img1.astype('float') * (1 - alpha2) + img2.astype('float') * alpha2).astype('uint8')

    @staticmethod
    def mask_video(video, mask, alpha=0.5):
        """ Put a mask to a video, with a given transparency.

        Args:
            video: input video
            mask: input mask
            alpha: transparency of mask

        Returns:
            Masked video
        """
        out_video = np.zeros(video.shape, dtype=video.dtype)
        for n_frame in range(video.shape[0]):
            frame = cv2.cvtColor(video[n_frame, :, :, :], cv2.COLOR_BGR2RGB)
            mask_frame = cv2.cvtColor(mask[n_frame, :, :, :], cv2.COLOR_BGR2RGB)
            out_video[n_frame, :, :, :] = LimeVideoExplainer.painter(frame, mask_frame, alpha)
        return out_video

    @staticmethod
    def score_map_rgb(video, score_map, hist_stretch=True, invert=True):
        """ Given a video and explanations score map, it merges them into one sole video.

        Args:
            video: input video
            score_map: explanations, with same shape as video, except for channels dimension
            hist_stretch: whether to perform histogram stretching to see better low intensity scores
            invert: True to use white as minimum score or False to use black

        Returns:
            Masked video, containing both input video and explanations
        """
        rgb_score_map = np.zeros(video.shape, dtype='float16')

        rgb_score_map[:, :, :, 0] = np.abs(score_map)
        rgb_score_map[:, :, :, 1] = np.abs(score_map)
        rgb_score_map[:, :, :, 2] = np.abs(score_map)

        if hist_stretch:
            max_value = np.max(rgb_score_map[:, :, :, 2])
            if max_value > 0:
                rgb_score_map = np.clip(rgb_score_map * (1 / max_value), 0, 1)

        if invert:
            rgb_score_map = 1 - rgb_score_map

        rgb_score_map[score_map < 0, 0] = 1
        rgb_score_map[score_map > 0, 1] = 1

        return (rgb_score_map * 255).astype('uint8')

    @staticmethod
    def join_expl(video, expl_list, alpha_list, method='arithmetic', th=None, top_k=None):
        """ Merges two or more score maps into one, using chosen mean and a weight for each dimension

        Args:
            video: input video
            expl_list: list of different explanations on the same video
            alpha_list: weight of each explanation score map
            method: "arithmetic", "geometric" or "harmonic", referring to the type of average used
            th: minimum score of each explanation to take it into account
            top_k: preserve only k most important regions. None for preserving all of them.

        Returns:
            Score map containing merged explanation score maps.
        """

        if method == 'geometric':
            pos_scores = np.ones(video.shape[:3], dtype='float16')
            neg_scores = np.ones(video.shape[:3], dtype='float16')
        else:
            pos_scores = np.zeros(video.shape[:3], dtype='float16')
            neg_scores = np.zeros(video.shape[:3], dtype='float16')

        neg_mask = np.ones(video.shape[:3])
        pos_mask = np.ones(video.shape[:3])

        for expl, alpha in zip(expl_list, alpha_list):
            values = expl.get_score_map(expl.top_labels[0], th=th)
            pos_values = values.copy()
            pos_values[pos_values < 0] = 0
            neg_values = -values.copy()
            neg_values[neg_values < 0] = 0
            if method == 'arithmetic':
                pos_scores += pos_values * alpha
                neg_scores += neg_values * alpha
            elif method == 'geometric':
                pos_scores *= pos_values ** alpha
                neg_scores *= neg_values ** alpha
            elif method == 'harmonic':
                pos_scores[pos_values > 0] += alpha / pos_values[pos_values > 0]
                pos_mask[pos_values <= 0] = 0
                neg_scores[neg_values > 0] += alpha / neg_values[neg_values > 0]
                neg_mask[neg_values <= 0] = 0

        if method == 'arithmetic':
            res = (pos_scores - neg_scores) / sum(alpha_list)
        elif method == 'geometric':
            res = pos_scores ** sum(alpha_list) - neg_scores ** sum(alpha_list)
        elif method == 'harmonic':
            pos_scores[pos_scores > 0] = sum(alpha_list) / pos_scores[pos_scores > 0]
            pos_scores *= pos_mask
            neg_scores[neg_scores > 0] = sum(alpha_list) / neg_scores[neg_scores > 0]
            neg_scores *= neg_mask
            res = pos_scores - neg_scores

        if top_k is not None:
            th_value = np.unique(np.abs(res))[-top_k]
            res[np.abs(res) < th_value] = 0

        return res