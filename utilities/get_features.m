function x = get_features(im, features, cell_size)
%GET_FEATURES
%   Extracts dense features from image.
%
%   X = GET_FEATURES(IM, FEATURES, CELL_SIZE)
%   Extracts features specified in struct FEATURES, from image IM. The
%   features should be densely sampled, in cells or intervals of CELL_SIZE.
%   The output has size [height in cells, width in cells, features].
%
%   To specify HOG features, set field 'hog' to true, and
%   'hog_orientations' to the number of bins.
%
%   To experiment with other features simply add them to this function
%   and include any needed parameters in the FEATURES struct.
%
%   Joao F. Henriques, 2013

	%right now, only HOG features (from VLFeat)
	if features.hog,
		x = vl_hog(single(im), cell_size, ...
			'NumOrientations', features.hog_orientations, 'BilinearOrientations');
    elseif features.LOMO,
        if size(im,3) == 1,
			im = repmat(im, [1, 1, 3]);  %from grayscale to RGB
        end
        options.numScales = 1;%3
        options.blockSize = 10;
        options.blockStep = 10;%5
        options.hsvBins = [8,8,8];
        options.tau = 0.3;
        options.R = [3, 5];
        options.numPoints = 4;
        d = LOMO(im,options);
        x = permute(d,[3 2 1 4]);
       % x = dd;
    elseif features.color,
        if size(im,3) == 1,
			im = repmat(im, [1, 1, 3]);  %from grayscale to RGB
        end
        blockSize = cell_size;
 %      overlapSize = 0;
 %      I = imresize(im,[128,48])/255; 
        I = im;
        [M N L] = size(I);
        a = floor(M/blockSize);
        b = floor(N/blockSize);
        x = [];
        parfor i=1:a
          for j=1:b
              patch = I( (i-1)*blockSize+1 : i*blockSize , (j-1)*blockSize+1 : j * blockSize,:);
              color_feature = color(double(patch));
              hog_feature = double(hog(patch));
              x(i,j,:) = [color_feature;hog_feature];
          end
        end 
	end
end

