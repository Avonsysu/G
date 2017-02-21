Gallery_sz = 50;
TestG = importdata([paths.ps 'annotation/test/train_test/TestG' num2str(Gallery_sz) '.mat']);
%load([paths.cache 'results/2686_G50_results.mat'])
precision = cell(100,1);
recall = cell(100,1);
ap = cell(100,1);
index = 1;
for i = 1901 : 2000,
    file_name = [num2str(i) '_G50'];
    
    load([paths.cache 'weights_2/' file_name '_weights.mat'])
    [precision{index}, recall{index}, ap{index}] = evaluate_detector(i, weights, bias, object_sz, cell_size, features, ...
	  detection, paths, file_name, save_plots, show_plots, show_detections, parallel,TestG);
    index = index+1;
end
  
save_file_name = [num2str(i) '_G' num2str(Gallery_sz)];
if ~exist([paths.cache 'results_2/'], 'dir'),
    mkdir([paths.cache 'results_2/'])
end
save([paths.cache 'results_2/' save_file_name '_results.mat'], 'precision', 'recall', 'ap')
 

