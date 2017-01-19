%path where cache files will be saved, and where each dataset is located
clear samples;  %avoid stressing memory use

neg_cache_file = [paths.cache 'neg_samples_PS.mat'];
pos_cache_file = [paths.cache 'pos_samples_PS.mat'];

%current parameters, to compare against the cache file
new_parameters = struct('sampling',sampling, 'features',features, 'cell_size',cell_size, ...
	'padding_cells',padding_cells, 'object_size',object_size);

if exist(neg_cache_file, 'file'),
	load(neg_cache_file)  %load data and parameters
	
	%if the parameters are the same, we're done
	if isequal(parameters, new_parameters),
		disp('Reloaded samples from cache.')
        
        load(pos_cache_file)
		
		%compute padding, relative to the object size
		padding = (patch_sz - object_sz) ./ object_sz;
		
		return
	end
end

load([paths.ps 'annotation/test/train_test/Train.mat']);
load([paths.ps 'annotation/test/train_test/TestG50.mat']);

%otherwise, start from scratch. first, load the positive samples
load_pos_samples;

num_pos_samples = size(pos_samples,4);
sample_sz = size(pos_samples);

%now the negatives
disp('Loading negative samples...')

%compute time
tic;

%list training image files for all classes
%images = dataset_list(dataset, 'train', class, false);  %skip images with this class
images = {};
for n = 1 : size(Train,1)
   for k = 1 : size(Train{n,1}.scene,2)
       image = Train{n,1}.scene(k).imname;
       images = [images;image]; 
   end
end

n = numel(images);

%dense sampling

%stride size, in cells (vertical and horizontal directions)
stride_sz = floor(sampling.neg_stride * sample_sz(1:2));
%stride_sz = floor(sampling.neg_stride * sample_sz(3));

%compute max. number of samples given the image size and stride
%(NOTE: this is probably a pessimistic estimate!)
num_neg_samples = numel(images) * prod(floor(sampling.neg_image_size / cell_size ./ stride_sz));
%num_neg_samples = numel(images) * floor(sample_sz(3) / stride_sz);

%initialize data structure for all samples, starting with positives
samples = zeros([sample_sz(1:3), num_neg_samples], 'single');

progress();

idx = 1;  %index of next sample

for f = 1:n,
    %load image and bounding box info
    %[boxes, im] = dataset_image(dataset, class, images{f});
    im = imread([paths.ps 'Image/SSM/' images{f}]);
    if size(im,3) == 1,
        im = repmat(im, [1, 1, 3]);  %from grayscale to RGB
    end  

    %ensure maximum size
    if size(im,1) > sampling.neg_image_size(1),
        im = imresize(im, [sampling.neg_image_size(1), NaN], 'bilinear');
    end
    if size(im,2) > sampling.neg_image_size(2),
        im = imresize(im, [NaN, sampling.neg_image_size(2)], 'bilinear');
    end
    
    %extract features 
    x = get_features(im, features, cell_size);
  
    %extract subwindows, given the specified stride
    for r = 1 : stride_sz(1) : size(x,1) - sample_sz(1) + 1,
        for c = 1 : stride_sz(2) : size(x,2) - sample_sz(2) + 1
            %store the sample
            samples(:,:,:,idx) = x(r : r+sample_sz(1)-1, c : c+sample_sz(2)-1, :);
            idx = idx + 1;
        end
    end

    progress(f, n);
end

assert(idx > 1, 'No valid negative samples to load.')

%trim any uninitialized samples at the end
if idx - 1 < size(samples,4),
	samples(:,:,:, idx : end) = [];
end

disp(['Loaded ' int2str(size(samples,4)) ' negtive samples.  cost time' num2str(toc)]);

% %save the results
% if ~exist(paths.cache, 'dir'),
% 	mkdir(paths.cache)
% end

parameters = new_parameters;

try  %use 7.3 format, otherwise Matlab *won't* save matrices >2GB, silently
	save(neg_cache_file, 'samples', 'parameters', '-v7.3')
catch  %#ok<CTCH>  if it's not supported just use whatever is the default
	save(neg_cache_file, 'samples', 'parameters')
end
