
%% Working Example of Clustering Analysis 
% This example provides a working example of the core components for
% identifying robust clusters. In the example, data contains 
% 400 features x 200 samples (can be thought of as ROI x Subject). First,
% Pearson correlation is used to esimate the similarity across samples. The
% similarity matrix is then clustered using Modularity-Maximization. 
% The clustering is repeated 100x. Then Consensus-Based Partition is used 
% to identify stable clusters cross iterations. Lastly, SVM is used to 
% corroborate the seperation of the data into clusters.
%
% The code is sperated into to parts. In Section 1, we show each step in the
% process to give the user a detailed view of the analysis. This is done
% so that the user can be modify each of the steps to fit their 
% specific needs. Common changes might be: 1) using other metrics to 
% estimate the similarity; 2) clustering algorithms; 3) classfiers

% In Section 2, the core elements of are wrapped together in one function
% for ease.

%%
clear 
close all

%% Add path
addpath Utils/

%% Load Sample Data

load Demo_1.mat

%% Section 1
% Estimate the Similarity & Cluster
r = corr(X); % Estimate the similarity across Samples; Data structure Features x Sample 
r = r - diag(diag(r)); % For modularity-maximization set diagonal to zero

%% Modularity-Maximization Based Clustering
Nitr = 100; % number of time to repeat the clustering
gamma = 1; % resolution parameter in modularity-maximization
for i = 1:Nitr
    c(i,:) = community_louvain(r,gamma,[],'negative_sym'); % negative_sym takes into account the negative values in the correlation matrix
end

% Consensus-Based Partition
clustID = consensus_iterative(c); 
[~,cidx] = sort(clustID); % sort the clusters for visualization

%% SVM-Based Corroboration
X1 = X';
mdl = fitcecoc(X1,clustID,'KFold',10); 
predictedClustId = kfoldPredict(mdl); 


%% Section 2 
% Alteratively, the above steps can be are combined into the following function
gamma = 1;
Nitr = 100;

[clustID,SVM_Confusion_Mat,Similarity_Mat,ClustID_per_Itr] = clustData(X,gamma,Nitr);

[~,cidx] = sort(clustID); % sort the clusters for visualization

%% Basic Visualizations of the Data & Results
% Data
close all
figure('Units','centimeters','Position',[10 10 10 5])
imagesc(X')
caxis([-15  15])
colormap(redblue)
set(gca,'XTick',[],'YTick',[])
ylabel('Features')
xlabel('Sample')
title('Data')

f2s = 'Figures/Sample_Data.eps';
saveas(gcf,f2s,'epsc')

% Iterative Mod
figure('Units','centimeters','Position',[10 10 10 5])
imagesc(ClustID_per_Itr)
caxis([1 max(ClustID_per_Itr(:))])
set(gca,'XTick',[],'YTick',[])
xlabel('Sample')
ylabel('Interation')
title('Cluster Labels')
colorbar
colormap(parula(max(ClustID_per_Itr(:))))
f2s = 'Figures/Clust_Iterations.eps';
saveas(gcf,f2s,'epsc')

% Similarity Matrix
cbMin = -0.1;
cbMax = 0.1;
cmap = redblue(100);
nodeComm = clustID(cidx);
% CommColor = rand(max(clustID),3);
Cbar = 'on';
DispComm = 'on'; 
cbarlabel = 'Similarity (r)'; 

my_ConnMat_Figure(Similarity_Mat(cidx,cidx),cbMax,cbMin,cmap,nodeComm,CommColor,Cbar,DispComm,cbarlabel)
xlabel('Sample')
ylabel('Sample')

f2s = 'Figures/Similarity_Matrix.eps';
saveas(gcf,f2s,'epsc')

% Confusion Matrix
figure('Units','centimeters','Position',[10 10 7 7])
confusionchart(SVM_Confusion_Mat,'Normalization','row-normalized')
title('Confusion Matrix')
f2s = 'Figures/Confusion_Matrix.eps';
saveas(gcf,f2s,'epsc')

% Comparison of Cluster Empirical and Observed Clustered Data
[~,score] = pca(X');

figure('Units','centimeters','Position',[10 10 5 5])
gscatter(score(:,1), score(:,2), labels);
legend('off')
xlabel('PC 1'); 
ylabel('PC 2');
title('Empirical Clusters');
f2s = 'Figures/Empirical_Clusters.eps';
saveas(gcf,f2s,'epsc')

figure('Units','centimeters','Position',[10 10 5 5])
gscatter(score(:,1), score(:,2), clustID);
xlabel('PC 1'); ylabel('PC 2');
legend('off')
title('Recovered Clusters');
f2s = 'Figures/Recoved_Clusters.eps';
saveas(gcf,f2s,'epsc')


% Histogram of Similarity Matrix
figure('Units','centimeters','Position',[10 10 7 5])
histogram(Similarity_Mat(Similarity_Mat~=0),'Normalization','probability')
xlabel('Similarity (r)'); 
ylabel('Prop. of Samples');
f2s = 'Figures/Similarity_Histogram.eps';
saveas(gcf,f2s,'epsc')
