disp('Loading positive samples...')

%compute time
tic;

n = size(TestG50,2);

%first, count positive samples and figure out average aspect ratio
aspect_ratio_sum = 0;
num_pos_samples = n;
for k = 1:n,
    boxes = TestG50(1).Query.idlocate;
    
	aspect_ratio_sum = aspect_ratio_sum + sum(boxes(:,3) ./ boxes(:,4));
	%num_pos_samples = num_pos_samples + 1;
end
aspect_ratio = aspect_ratio_sum / num_pos_samples;  %average value

assert(num_pos_samples > 0, 'No valid positive samples to load.')

if ~isscalar(object_size),  %both height and width were specified
	object_sz = object_size;
	aspect_ratio = object_sz(2) / object_sz(1);
else
	%we are only given the size of the largest dimension as a reference
	%(width or height), the other must be deduced from the data.
	if aspect_ratio >= 1,  %width >= height, width is fixed to object_size
		object_sz = [floor(object_size / aspect_ratio), object_size];
	else  %width < height, height is fixed to object_size
		object_sz = [object_size, floor(object_size * aspect_ratio)];
	end
end

%enlarge the patch size with padding
patch_sz = object_sz + padding_cells * cell_size;

%make sure the patch size is a multiple of the cell size
patch_sz = ceil(patch_sz / cell_size) * cell_size;

%length of the diagonal
diagonal = sqrt(sum(object_sz.^2));

%total padding, relative to the object size
padding = (patch_sz - object_sz) ./ object_sz;


%extract features (e.g., HOG) of a dummy sample to figure out the size
sample = get_features(zeros(patch_sz), features, cell_size);

%allocate results array
if ~sampling.flip_positives,
	pos_samples = zeros([size(sample), num_pos_samples], 'single');
    pos_ids = cell(1,num_pos_samples);
else  %allocate twice the samples, for flipped versions
	pos_samples = zeros([size(sample), 2 * num_pos_samples], 'single');
    pos_ids = cell(1,2 * num_pos_samples);
end
idx = 1;

%if debug_pos_samples, figure, end

progress();

for k = 1:n,
	%load image and ground truth bounding boxes (x,y,w,h)
	%[boxes, im] = dataset_image(dataset, class, images{k});
    image = TestG50(n).Query.imname;
    boxes = TestG50(n).Query.idlocate;
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

    %store the sample
    pos_samples(:,:,:,idx) = sample;
    pos_ids{1,idx} = TestG50(n).Query.idname;
    idx = idx + 1;
    
    if sampling.flip_positives,
        %store a horizontally flipped version too
        sample = get_features(patch(:, end:-1:1, :), features, cell_size);
        pos_samples(:,:,:,idx) = sample;
        pos_ids{1,idx} = TestG50(n).Query.idname;
        idx = idx + 1;
    end
    
	progress(k, n);
end

%trim any uninitialized samples at the end
num_rejected = size(pos_samples,4) - idx + 1;
if num_rejected > 0,
    pos_samples(:,:,:,idx : end) = [];
end

%print some debug info
disp(['Loaded ' int2str(size(pos_samples,4)) ' positive samples. Rejected '...
	int2str(num_rejected) ' (wrong aspect ratio).']);

disp(['cost time:',num2str(toc)]);

%save the results
if ~exist(paths.cache, 'dir'),
	mkdir(paths.cache)
end

try  %use 7.3 format, otherwise Matlab *won't* save matrices >2GB, silently
	save(pos_cache_file, 'pos_ids', 'pos_samples', 'num_pos_samples','patch_sz', 'object_sz', '-v7.3')
catch  %#ok<CTCH>  if it's not supported just use whatever is the default
	save(pos_cache_file, 'pos_ids', 'pos_samples', 'num_pos_samples','patch_sz', 'object_sz')
end
