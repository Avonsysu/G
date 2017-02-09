function [precision, recall, ap] = evaluate_detector( pos_id , weights, bias, object_sz, cell_size, features, detection, ...
	paths, save_file_name, save_plots, show_plots, show_detections, parallel,TestG)
%EVALUATE_DETECTOR
%   Evaluates a detector on the test set, computing precision/recall
%   curves and average precision (according to the Pascal VOC criterion).
%   Optionally, it can save the curve data and show it, as well as show
%   the resulting detections in an interactive figure.
%
%   Joao F. Henriques, 2013


	%list all image files from the test set
	%images = dataset_list(dataset, 'test');
	%n = numel(images);
    n = size(TestG(pos_id).Gallery,2);
    
    %n = 3
    
	%store results in cell arrays because we don't know a priori their size
	scores = cell(n,1);
	labels = cell(n,1);
    max_scores = zeros(n,1);
    max_labels = zeros(n,1);
	total_gt_rects = 0;  %number of ground truth detections

	if show_detections,
		detections = cell(n,1);
	end


	%debug info: show number of detected scales, to help set those parameters
% 	compute_scales = @(effective_min_scale) ...
% 		ceil(log(detection.max_scale / effective_min_scale) / log(detection.scale_step));

% 	disp(['Maximum number of detection scales: ' ...
% 		int2str(compute_scales(detection.min_scale)) ...
% 		', for typical 640x480 images: ' ...
% 		int2str(compute_scales(max(detection.min_scale, max(object_sz) / 640))) '.'])


	%for most evaluation metrics, we don't have to threshold detections
	threshold = -inf;

	tic
	progress('Running evaluation')

	if ~parallel,
		for f = 1:n,
			%load image to run detector on, and ground truth to evaluate it
			%[gt_rects, im] = dataset_image(dataset, class, images{f});
            gt_rects = TestG(pos_id).Gallery(f).idlocate;
            image = TestG(pos_id).Gallery(f).imname;
            im = imread([paths.ps 'Image/SSM/' image]);

			%run detector
			rects = detect(im, weights, bias, object_sz, cell_size, features, detection, threshold);

			%compute evaluation info based on ground truth, and store it
			[labels{f}, scores{f}] = evaluate_image(gt_rects, rects);
            [max_scores(f),index] = max(scores{f});
            max_labels(f) = labels{f}(index);
            
			%store bounding boxes if necessary
			if show_detections, detections{f} = rects; end

			%count total number of positives
			total_gt_rects = total_gt_rects + size(gt_rects,1);

			progress(f, n)
		end
	else
		%exactly the same, but with "parfor"
		parfor f = 1:n,
			gt_rects = TestG(pos_id).Gallery(f).idlocate;
            image = TestG(pos_id).Gallery(f).imname;
            im = imread([paths.ps 'Image/SSM/' image]);
            
			rects = detect(im, weights, bias, object_sz, cell_size, features, detection, threshold);
			
            [labels{f}, scores{f}] = evaluate_image(gt_rects, rects);
			
            [max_scores(f),index] = max(scores{f});
            max_labels(f) = labels{f}(index);
            
            if show_detections, detections{f} = rects; end
			total_gt_rects = total_gt_rects + size(gt_rects,1);
			progress(f, n)
		end
	end
	toc

	%concatenate cell array contents into vectors
	%all_labels = cat(1, labels{:});
	%all_scores = cat(1, scores{:});

	%compute PR curve and AP; a figure will be shown if show_plots is true
	[precision, recall, ap, sorted_scores] = ...
		precision_recall(max_labels, max_scores, total_gt_rects, show_plots);
	
	disp([num2str(pos_id) '_AP: ' num2str(ap)])

	if show_plots,  %set figure name
		set(gcf, 'Number','off', 'Name',['PR for ' class ' in ' dataset.name ' (AP: ' num2str(ap) ')']);
	end

	if save_plots,  %save curve data to a MAT file
		if ~exist([paths.cache 'plots/'], 'dir'),
			mkdir([paths.cache 'plots/'])
		end
		save([paths.cache 'plots/' save_file_name], 'precision', 'recall', 'ap')
	end

	%show detections in an interactive figure
	if show_detections,
		%for visualization, choose a threshold at ~0.5 precision
		visualization_threshold = sorted_scores(find(precision <= 0.5, 1));
		if isempty(visualization_threshold),
			visualization_threshold = median(sorted_scores);
		end
		
		visualize_detections(dataset, class, images, detections, labels, visualization_threshold);
	end

end

