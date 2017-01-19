%set data paths and other parameters
set_defaults;

%cell size for HOG or any other dense feature
cell_size = 8;

%size of object in pixels (width or height, whichever is largest)
object_size = 16 * cell_size;
%pad samples with extra cells around the object, to minimize wrap-around effects
padding_cells = 5;

%crop these cells from the final template. this may be needed because
%large templates (too much padding) can't detect near image borders.
cropping_cells = 4;  %should be an even number

%negative sampling parameters
sampling.neg_stride = 2/3;  %stride of grid sampling
sampling.neg_image_size = [600 800];  %max. size of sampled images [rows, columns]

%standard deviation and magnitude of regression targets
target_sigma = 0.2;
target_magnitude = 1;

%train a complex SVR (liblinear) with no bias term
training.type = 'svr';
training.regularization = 1e-2;  %SVR-C, obtained by cross-validation on a log. scale
training.epsilon = 5e-3;
training.complex = true;
training.bias_term = 0;


%make sure the needed functions are in the path
addpath('./detection', './evaluation', './training', './utilities', './libraries')

run('/home/share/yafang/vlfeat-0.9.20/toolbox/vl_setup');

%main script for training and evaluation
%run_circulant;

load_samples;
