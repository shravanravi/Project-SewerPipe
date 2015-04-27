%%
clc; clear all; close all;
tic;
% addpath C:\Users\Shravan\Google Drive\Shared with Dell\...
%         Team SewerPipe\Programs\Matlab Functions\libsvm-3.20\...
%         matlab\multiclass;
% addpath C:\Users\Shravan\Google Drive\Shared with Dell\...
%         Team SewerPipe\Programs\Main\src;
load ('ZZZ_imageSetsFinalFeatMatLabels.mat');

%%
% Load Training, Validation & Testing Data
[test_Y]       = YTest;
[test_feature] = XTest;
[test_x]       = double(test_feature);
Ntest          = size (test_x , 1);
[val_Y]         = Yval;
[val_feature]   = Xval;
[val_x]         = double(val_feature);
Nval            = size (val_x , 1);
[train_Y]       = Ytrain;
[train_feature] = Xtrain;
[train_x]       = double(train_feature);
Ntrain          = size (train_x , 1);

% Shuffle (This step is skipped since we used an already shuffled data set)
% [sortedTrainLabel, permIndex] = sortrows(rawTrainLabel);
% sortedTrainData = rawTrainData(permIndex,:);
% [sortedTestLabel, permIndex] = sortrows(rawTestLabel);
% sortedTestData = rawTestData(permIndex,:);
% (OR)
% [ train_X, train_Y, Target ] = shuffleFeatMatLabel( train_x, train_label );
% [ val_X, val_Y, Target ]   = shuffleFeatMatLabel( val_x, val_label );
% 

%% Feature Reduction
% FOR TRAINING
 Feature_Matrix = train_x;
% Remove inf and NAN
Feature_Matrix(isinf(Feature_Matrix) | isnan(Feature_Matrix)) = 0;

% Remove zero rows
Feature_Matrix( all(~Feature_Matrix,2), : ) = [];

% Remove zero columns
Feature_Matrix( :, all(~Feature_Matrix,1) ) = [];

%Remove duplicate columns 
feat_mat_trans = Feature_Matrix';
unq_feat_mat   = unique (feat_mat_trans , 'rows' , 'stable');
feat_matrix    = unq_feat_mat';

% Use pca to obtain the eigenvalues
[coeff , score , eigenvalues] = pca (feat_matrix); 
% plot (eigenvalues);
% By visual inspection, numFeatures to be reduced to : k = 30
k       = 30;
[coeff_1 , score_1 , eigenvalues_1] = pca (feat_matrix , 'NumComponents' , k);

train_X = score_1;

% FOR VALIDATION
Feature_Matrix_V = val_x;
% Remove inf and NAN
Feature_Matrix_V(isinf(Feature_Matrix_V) | isnan(Feature_Matrix_V)) = 0;

% Remove zero rows
Feature_Matrix_V( all(~Feature_Matrix_V,2), : ) = [];

% Remove zero columns
Feature_Matrix_V( :, all(~Feature_Matrix_V,1) ) = [];

%Remove duplicate columns 
feat_mat_trans_V = Feature_Matrix_V';
unq_feat_mat_V   = unique (feat_mat_trans_V , 'rows' , 'stable');
feat_matrix_V    = unq_feat_mat_V';

% Use pca to obtain the eigenvalues
[coeff_V , score_V , eigenvalues_V] = pca (feat_matrix_V); 
% plot (eigenvalues);
% By visual inspection, numFeatures to be reduced to : k = 30
k       = 30;
[coeff_2 , score_2 , eigenvalues_2] = pca (feat_matrix_V , 'NumComponents' , k);
val_X = score_2;

% FOR TESTING : 
Feature_Matrix_T = test_x;
% Remove inf and NAN
Feature_Matrix_T(isinf(Feature_Matrix_T) | isnan(Feature_Matrix_T)) = 0;

% Remove zero rows
Feature_Matrix_T( all(~Feature_Matrix_T,2), : ) = [];

% Remove zero columns
Feature_Matrix_T( :, all(~Feature_Matrix_T,1) ) = [];

%Remove duplicate columns 
feat_mat_trans_T = Feature_Matrix_T';
unq_feat_mat_T   = unique (feat_mat_trans_T , 'rows' , 'stable');
feat_matrix_T    = unq_feat_mat_T';

% Use pca to obtain the eigenvalues
[coeff_T , score_T , eigenvalues_T] = pca (feat_matrix_T); 
% plot (eigenvalues);
% By visual inspection, numFeatures to be reduced to : k = 30
k       = 30;
[coeff_3 , score_3 , eigenvalues_3] = pca (feat_matrix_T , 'NumComponents' , k);
test_X = score_3;

totalData = [train_X; test_X];
totalLabel = [train_Y; test_Y]; 
NClass = 9;
%%
% % #######################
% % Automatic Cross Validation 
% % Parameter selection using n-fold cross validation
% % #######################
% stepSize = 10;
% bestLog2c = 1;
% bestLog2g = -1;
% epsilon = 0.01;
% bestcv = 0;
% Ncv = 3; % Ncv-fold cross validation cross validation
% deltacv = 10^6;
% t = 0;
% 
% while abs(deltacv) > epsilon
%     bestcv_prev = bestcv;
%     prevStepSize = stepSize;
%     stepSize = prevStepSize/2;
%     log2c_list = bestLog2c-prevStepSize:stepSize/2:bestLog2c+prevStepSize;
%     log2g_list = bestLog2g-prevStepSize:stepSize/2:bestLog2g+prevStepSize;
%     
%     numLog2c = length(log2c_list);
%     numLog2g = length(log2g_list);
%     cvMatrix = zeros(numLog2c,numLog2g);
%     
%     for i = 1:numLog2c
%         log2c = log2c_list(i);
%         for j = 1:numLog2g
%             log2g = log2g_list(j);
%             cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
%             cv = get_cv_ac(train_Y, train_X, cmd, Ncv);
%             if (cv >= bestcv),
%                 bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%             end
%         end
%     end
%     deltacv = bestcv - bestcv_prev;
%     
% end
% disp(['The best parameters, yielding Accuracy=',num2str(bestcv*100),'%, are: C=',num2str(bestc),', gamma=',num2str(bestg)]);

%
% #######################
% Train the SVM in one-vs-rest (OVR) mode
% #######################
bestc = 2;
bestg = 16;
bestParam = ['-q -c ', num2str(bestc), ', -g ', num2str(bestg)];
% bestParam = ['-q -c 8 -g 0.0625'];
model = ovrtrainBot(train_Y, train_X, bestParam);

% FOR VVALIDATION : 
% #######################
% Classify samples using OVR model
% #######################
[predict_label_val, accuracy_val, decis_values_val] = ovrpredictBot(val_Y, val_X, model);
[decis_value_winner_val, label_out_val] = max(decis_values_val,[],2);
% 
% % % #######################
% % Make confusion matrix
% % #######################
% FOR VALIDATION : 
[confusionMatrix_val ,order_val] = confusionmat(val_Y , label_out_val);
% % Note: For confusionMatrix
% % column: predicted class label
% % row: ground-truth class label
% % But we need the conventional confusion matrix which has
% % column: actual
% % row: predicted
% figure; imagesc(confusionMatrix_val);
% title ('Confusion Matrix for Validation Set' , 'FontSize' , 18);
% xlabel('actual class label' , 'FontSize' , 18);
% ylabel('predicted class label' , 'FontSize' , 18);
totalAccuracy_val = trace(confusionMatrix_val)/Nval;
disp(['Total accuracy from the SVM: ',num2str(totalAccuracy_val*100),'%']);

% FOR TESTING : 
% #######################
% Classify samples using OVR model
% #######################
[predict_label_test, accuracy_test, decis_values_test] = ovrpredictBot(test_Y, test_X, model);
[decis_value_winner_test, label_out_test] = max(decis_values_test,[],2);

% % #######################
% % Make confusion matrix
% % ####################### 
[confusionMatrix_test ,order_test] = confusionmat(test_Y , label_out_test);
% figure; imagesc(confusionMatrix_val);
% title ('Confusion Matrix for Validation Set' , 'FontSize' , 18);
% xlabel('actual class label' , 'FontSize' , 18);
% ylabel('predicted class label' , 'FontSize' , 18);
totalAccuracy_val = trace(confusionMatrix_val)/Nval;
disp(['Total accuracy from the SVM: ',num2str(totalAccuracy_val*100),'%']);
%  
%%
% #######################
% Plot the results
% #######################
figure; 
% subplot(1,3,2); imagesc(predict_label_val); title('predicted labels_val'); xlabel('class k vs rest'); ylabel('observations'); colorbar;
subplot(1,2,1); imagesc(decis_values_val); title('decision values_val'); xlabel('class k vs rest'); ylabel('observations'); colorbar;
subplot(1,2,2); imagesc(label_out_val); title('output labels_val'); xlabel('class k vs rest'); ylabel('observations'); colorbar;

% plot the true label for the test set
patchSize = 20*exp(decis_value_winner_test);
% colorList = generateColorList(NClass);
% colorPlot = colorList(test_Y,:);
figure; 
scatter(test_X(:,1),test_X(:,2),patchSize , 'filled'); hold on;

% plot the predicted labels for the test set
patchSize = 10*exp(decis_value_winner_test);
% colorPlot = colorList(label_out_test,:);
scatter(test_X(:,1),test_X(:,2),patchSize ,'filled');

%%
% % #######################
% % Plot the decision boundary
% % #######################

% Generate data to cover the domain
minData = min(totalData,[],1);
maxData = max(totalData,[],1);
stepSizePlot = (maxData-minData)/50;
[xI yI] = meshgrid(minData(1):stepSizePlot(1):maxData(1),minData(2):stepSizePlot(2):maxData(2));
% % #######################
% % Classify samples using OVR model
% % #######################
[pdl, acc, dcsv] = ovrpredictBot(xI(:)*0, [xI(:) yI(:)], model);
% % Note: when the ground-truth labels of testData are unknown, simply put
% % any random number to the testLabel
[dcsv_winner, label_domain] = max(dcsv,[],2);
% 
% plot the result
patchSize = 20*exp(dcsv_winner);
% colorList = generateColorList(NClass);
% colorPlot = colorList(label_domain,:);
figure; 
scatter(xI(:),yI(:),patchSize ,'filled');

%%
Runtime = toc;
%% Cite:
% 1. libsvm for MATLAB?, Kittipat's Homepage
%    https://sites.google.com/site/kittipat/libsvm_matlab/complete_libsvm_example