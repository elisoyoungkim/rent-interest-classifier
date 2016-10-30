% Import datasets for training set
filename = 'IMDb_data.csv';
fid = fopen(filename,'rt');
[md]=textscan(fid, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d',...
       'headerlines', 1,...
       'delimiter',',',...
       'TreatAsEmpty','NA',...
       'EmptyValue', NaN);                              
fclose(fid);
featuresFraud = [md{2} md{3} md{4} md{5} md{6} md{7} md{8} md{9} md{10} md{11} md{12} md{13} md{14} md{14}];
L = {'nbreviews','duration','directorfblikes','actor3fblikes','actor1fblikes','gross','nbvotedusers','casttotalfblikes','nbmanInPoster','nbUsrReviews','budget','actor2fblikes','score','moviefblikes'}
cov_mat = corr(featuresFraud);
imagesc(cov_mat); % plot the matrix
set(gca, 'XTick', 1:15); % center x-axis ticks on bins
set(gca, 'YTick', 1:15); % center y-axis ticks on bins
set(gca, 'XTickLabel', L); % set x-axis labels
set(gca, 'YTickLabel', L); % set y-axis labels
title('Correlation graph', 'FontSize', 14); % set title
colormap('jet'); % set the colorscheme
% colorbar on; % enable colorbar