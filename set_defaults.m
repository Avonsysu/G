
% In this script you can set the data paths, as well as some options and
% default parameters used in most experiments.
%
% Joao F. Henriques, 2013


%path where cache files will be saved, and where each dataset is located
paths = [];
paths.cache = './data/detector_cache/';  %cache path
paths.ps = '/home/share/yafang/dataset/';  %ps path


%whether to use PARFOR or not
parallel = true;

%whether to show template visualization and precision curves by default
show_plots = false;

%whether to show detection bounding boxes, in an interactive figure
show_detections = false;

%whether to save the trained template weights
save_weights = true;

%whether to save the resulting plots
save_plots = false;

%feature parameters (more can be added easily in "get_features")
features = [];
features.hog = false;
features.hog_orientations = 9;  %number of orientation bins
features.LOMO = false;
features.color = true;

%sampling parameters
sampling = [];
sampling.flip_positives = true;  %train with horizontally-flipped virtual pos. samples by default
sampling.reject_aspect_ratio = 1.5;  %reject pos. samples with an aspect ratio that differs by this much from the mean


%test-time detection parameters
detection = [];
detection.max_scale = 1;  %maximum scale, relative to image size
detection.min_scale = 0.1;  %min. scale
detection.scale_step = 1.1;  %relative scale step

detection.max_peaks = 80;  %maximum number of detections per scale
detection.suppressed_scale = 0.3;  %response suppression around peaks, relative to object size

detection.max_overlap = 0.6;  %maximum relative area overlap between two detections

%training parameters, set for each experiment differently
training = [];


%ensure the cache path ends with the directory separator
%(note that the '/' separator works on all systems, including Windows).
if all(paths.cache(end) ~= '/\'), paths.cache(end+1) = '/'; end

