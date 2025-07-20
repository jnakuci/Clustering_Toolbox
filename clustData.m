function [ClustID,Confusion_Mat,Similarity_Mat,ClustID_per_Itr] = clustData(data,gamma,Nitr)

%%  Description
% Function for identifying robust clusters in data. 
% Data is first clustered using Modularity-Maximization. Followed by 
% Consensus-Based Partition to identify stable clusters across iterations. 
% Lastly, SVM is used to corroborate the seperation of the data. 
% 
% INPUTS 
% data: rows correspond to Features and the columns correspond to
% Samples (for instance ROI x Subject).
% gamma: resolution parameter for modularity-maximization 
% Nitr: number of iterations to perform the clustering 
%
% OUPUTS
% ClustID: Cluster labels
% Confusion_Mat: Confusion matrix reflecting the SVM-based corroboration
% of cluster lables


%% Estimate the Similarity
Similarity_Mat = corr(data); 
Similarity_Mat = Similarity_Mat - diag(diag(Similarity_Mat)); % For modularity-maximization set diagonal to zero

%% Modularity-Maximization Based Clustering

for i = 1:Nitr
    ClustID_per_Itr(i,:) = community_louvain(Similarity_Mat,gamma,[],'negative_sym'); % negative_sym takes into account the negative values in the correlation matrix
end

%% Consensus-Based Partition
ClustID = consensus_iterative(ClustID_per_Itr); 

%% SVM-Based Corroboration
X1 = data';
mdl = fitcecoc(X1,ClustID,'KFold',10); 
SVM_predictedClustID = kfoldPredict(mdl); 

% Confusion 
Confusion_Mat = confusionmat(ClustID,SVM_predictedClustID);

end





