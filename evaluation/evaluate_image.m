function [labels, scores] = evaluate_image(gt_rects, rects)
%EVALUATE_IMAGE
%   Evaluates a single image (mainly called by "evaluate_detector").
%
%   [LABELS, SCORES] = EVALUATE_IMAGE(GT_RECTS, RECTS)
%   Matches ground-truth bounding boxes GT_RECTS to detections RECTS,
%   yielding a list of labels for each detection (whether it's a true
%   positive or false positive).
%
%   Joao F. Henriques, 2013

	scores = rects(:,5);
	
	if isempty(gt_rects) || isempty(rects),
		%no ground-truth rects or no detections, all detections (if any)
		%must be false positives
		labels = -ones(size(rects,1),1);
	else
		labels = assign_gt_bbox(gt_rects, rects(:,1:4), scores);
		labels(labels == 0) = -1;
	end

end

%INRIA benchmark code:
% assign detections to ground truth bboxes
function l = assign_gt_bbox(gt,dt,s)
    %compute pairwise intersection by union
    overlap = zeros(size(gt,1), size(dt,1));
    agt = gt(:,3).*gt(:,4);
    adt = dt(:,3).*dt(:,4);
    for i = 1:size(gt,1)
        for j = 1:size(dt,1)
            rr = rectint(gt(i,:),dt(j,:));
            overlap(i,j) = rr/(agt(i)+ adt(j) - rr);
        end
    end
    oT = 0.5; %PASCAL CRITERION
    notTaken = ones(1,length(s));
    l = zeros(length(s),1);
    for i = 1:size(gt,1)
        bbi = find(overlap(i,:) > oT & notTaken == 1);
        [mval,mi] = max(s(bbi));
        mindx = bbi(mi);
        notTaken(mindx) = 0;
        l(mindx) = 1;
    end
end

