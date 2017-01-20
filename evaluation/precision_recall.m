function [precision, recall, ap, sorted_scores] = ...
	precision_recall(labels, scores, num_positives, show_plots)
%PRECISION_RECALL
%   Computes precision-recall (PR) curve and average precision (AP).
%
%   [PRECISION, RECALL, AP, SORTED_SCORES] = PRECISION_RECALL(LABELS,
%     SCORES, NUM_POSITIVES, SHOW_PLOTS)
%   Computes a PR curve as vectors PRECISION and RECALL, as well as the AP,
%   given LABELS and SCORES as computed by the "evaluate_image" function.
%   The number of true positives NUM_POSITIVES is also required. If
%   SHOW_PLOTS is true, a figure will be created.
%
%   The corresponding scores SORTED_SCORES can also be returned, which may
%   be useful to obtain a threshold for a given precision or recall.
%
%   Joao F. Henriques

	%this implementation follows closely "vl_pr" and "vl_tpfp" in VLFeat
	%(mostly to make it self-contained)

	%trivial case
	if isempty(labels) || isempty(scores) || num_positives == 0,
		precision = [];
		recall = [];
		ap = 0;
		return
	end
	
	assert(all(isfinite(scores)))

	%sort labels by descending score order
	[sorted_scores, order] = sort(scores, 'descend');
	labels = labels(order);
	
	%accumulate TP and FP
	tp = [0; cumsum(labels > 0)];
	fp = [0; cumsum(labels < 0)];
	
	%compute precision and recall
	recall = tp / num_positives;
	precision = max(tp, eps) ./ max(tp + fp, eps);
	
	%compute AP on recalled positives
	sel = find(diff(recall)) + 1;
	ap = sum(precision(sel)) / num_positives;
	
	%plot results if needed
	if nargin > 3 && show_plots,
		figure;
		plot(recall, precision, 'b-', 'LineWidth',2); grid on; axis square
		xlabel('Recall'), ylabel('Precision')
	end

end

