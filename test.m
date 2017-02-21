%path where cache files will be saved, and where each dataset is located
clear samples;  %avoid stressing memory use
clear pos_samples;

neg_cache_file = [paths.cache 'neg_samples_PS_2.mat'];
pos_cache_file = [paths.cache 'pos_samples_PS_2.mat'];

%current parameters, to compare against the cache file
new_parameters = struct('sampling',sampling, 'features',features, 'cell_size',cell_size, ...
	'padding_cells',padding_cells, 'object_size',object_size);

%if exist(neg_cache_file, 'file'),
%	load(neg_cache_file)  %load data and parameters
	%samples = samples(:,:,:,1:10000);
	%if the parameters are the same, we're done
%	if isequal(parameters, new_parameters),
%		disp('Reloaded samples from cache.')        
%        load(pos_cache_file)		
		%compute padding, relative to the object size
%		padding = (patch_sz - object_sz) ./ object_sz;		
%		return
%	end
%end

load([paths.ps 'annotation/test/train_test/Train.mat']);
load([paths.ps 'annotation/test/train_test/TestG50.mat']);

%otherwise, start from scratch. first, load the positive samples
load_pos_samples;

sample_sz = size(pos_samples);

%now the negatives
disp('Loading negative samples...')

%compute time
tic;
num_of_train = 0;
for k = 1 : size(Train,1)
   num_of_train = num_of_train + size(Train{k,1}.scene,2);
end
n = num_of_train;

try
	rand('seed', 0);  %#ok<RAND>
catch  %#ok<CTCH>
	rng(0);  %new syntax
end
%compute max. number of samples given the image size and stride
%(NOTE: this is probably a pessimistic estimate!)
num_neg_samples = (sampling.neg_samples_per_image + 1) * n;

%initialize data structure for all samples, starting with positives
samples = zeros([sample_sz(1:3), num_neg_samples], 'single');

idx = 1;
for i = 1 : size(Train,1)
   for j = 1 : size(Train{i,1}.scene,2)
       image = Train{i,1}.scene(j).imname;
       boxes = Train{i,1}.scene(j).idlocate;
       im = imread([paths.ps 'Image/SSM/' image]);
       
       ratio = boxes(3) / boxes(4) / aspect_ratio;
       %center coordinates
       xc = boxes(1) + boxes(3) / 2;
       yc = boxes(2) + boxes(4) / 2;
       sz = object_sz / object_sz(1) * boxes(4);  %rescale to have same height
       %apply padding in all directions
       sz = (1 + padding) .* sz;
       %x and y coordinates to extract. remember all sizes ("sz") are
       %in Matlab's format (rows, columns)
       xs = floor(xc - sz(2) / 2) : floor(xc + sz(2) / 2);
       ys = floor(yc - sz(1) / 2) : floor(yc + sz(1) / 2);
       %avoid out-of-bounds coordinates (set them to the values at
       %the borders)
       bounded_xs = max(1, min(size(im,2), xs));
       bounded_ys = max(1, min(size(im,1), ys));
       patch = im(bounded_ys, bounded_xs, :);  %extract the patch
       %set out-of-bounds pixels to 0
       patch(ys < 1 | ys > size(im,1), xs < 1 | xs > size(im,2), :) = 0;
       %resize to the common size
       patch = imresize(patch, patch_sz, 'bilinear');
       %extract features (e.g., HOG)
       sample = get_features(patch, features, cell_size);
       
       samples(:,:,:,idx) = sample;
       idx = idx + 1;
       
        %extract features (e.g., HOG)
		x = get_features(im, features, cell_size);

		%if image isn't big enough for a full patch, skip it
		if size(x,1) < sample_sz(1) || size(x,2) < sample_sz(2),
			continue
        end
        
		for s = 1:sampling.neg_samples_per_image,
			%random position, fully inside image
			r = randi([1, size(x,1) - sample_sz(1)]);
			c = randi([1, size(x,2) - sample_sz(2)]);

			%extract sample and store it
			samples(:,:,:, idx) = x(r : r + sample_sz(1) - 1, c : c + sample_sz(2) - 1, :);
			idx = idx + 1;
		end
       
   end
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
