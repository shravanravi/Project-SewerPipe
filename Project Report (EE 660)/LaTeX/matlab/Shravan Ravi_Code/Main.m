%//%************************************************************************%
%//%*                                                                      *%
%//%*                         Project Sewer Pipe						   *%
%//%*                                                                      *%
%//%*             Name: Shravan Ravi                    		           *%
%//%*                   Preetham Aghalaya Manjunatha                       *%
%//%*             USC ID Number: 9241187852		                           *%
%//%*             USC Email: shravanr@usc.edu                              *%
%//%*             Submission Date: 12/08/2014                              *%
%//%************************************************************************%
%//%*             Viterbi School of Engineering,                           *%
%//%*             Sonny Astani Dept. of Civil Engineering,                 *%
%//%*             Ming Hseih Dept. of Electrical Engineering,              *%
%//%*             University of Southern California,                       *%
%//%*             Los Angeles, California.                                 *%
%//%************************************************************************%

%% Start parameters
%--------------------------------------------------------------------------
clear all; close all; clc;
Start = tic;
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);

%% Inputs
MainInputs;

%% Extract feature matrix and class labels

if (largefeatmatlabel_inpstruct.flagswitch)
    featlab_traintestset = largefeaturematrixNlabels ...
        (hybrid_inpstruct, imgPartitionStruct);
end

% Bag-of-Features
bag_Tr_BoFOri   = bagOfFeatures(imgPartitionStruct.trainingSets);
featureMatrixTr_BoFOri = encode(bag_Tr_BoFOri, imgPartitionStruct.trainingSets);
labelArrayTr_BoFOri    = makeLabelArray(imgPartitionStruct.trainingSets);

bag_Vl_BoFOri   = bagOfFeatures(imgPartitionStruct.validationSets);
featureMatrixVl_BoFOri = encode(bag_Vl_BoFOri, imgPartitionStruct.validationSets);
labelArrayVl_BoFOri    = makeLabelArray(imgPartitionStruct.validationSets);

bag_Ts_BoFOri   = bagOfFeatures(imgPartitionStruct.trainingSets);
featureMatrixTs_BoFOri = encode(bag_Ts_BoFOri, imgPartitionStruct.testingSets);
labelArrayTs_BoFOri    = makeLabelArray(imgPartitionStruct.testingSets);

% Training, validation and testing
% Bag-of-Features
load ZZZ_imageSetsFull_BoF_Original.mat
categoryClassifier = trainImageCategoryClassifier(imgPartitionStruct.trainingSets, bag_Tr_BoFOri);
[confMatTr,knownLabelIdxTr,predictedLabelIdxTr,scoreTr] = evaluate(categoryClassifier, ...
                                                    imgPartitionStruct.trainingSets);
                                                
[confMatVl,knownLabelIdxVl,predictedLabelIdxVl,scoreVl] = evaluate(categoryClassifier, ...
                                                    imgPartitionStruct.validationSets);

[confMatTs,knownLabelIdxTs,predictedLabelIdxTs,scoreTs] = evaluate(categoryClassifier, ...
                                                    imgPartitionStruct.testingSets);

%% Pre-processing steps
%--------------------------------------------------------------------------
% Create image sets
%--------------------------------------------------------------------------
if (shuffleNpartfiles_inpstruct.flagswitch && ~ exist('ZZZ_imageSetsFull.mat','file'))
    imgPartitionStruct = makePartitions( shuffleNpartfiles_inpstruct );     %Subroutine: makePartitions 
else
    load ZZZ_imageSetsFull.mat
end
%--------------------------------------------------------------------------
% Image inpainting
%--------------------------------------------------------------------------
if (cropNinpaint_inpstruct.flagswitch)
    largesScaleImInpaint( imgPartitionStruct.imSetImgLocFull );             %Subroutine: largesScaleImInpaint
end

%% Processing step

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
% [sortedTrainLabel, permIndexTrain] = sortrows(rawTrainLabel);
% sortedTrainData = rawTrainData(permIndexTrain,:);
% [sortedTestLabel, permIndexTest] = sortrows(rawTestLabel);
% sortedTestData = rawTestData(permIndexTest,:);
% (OR)
% [ train_X, train_Y, Target ] = shuffleFeatMatLabel( train_x, train_label );
% [ val_X, val_Y, Target ]   = shuffleFeatMatLabel( val_x, val_label );

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

%% CLASSIFIER NO. 1 : SVM (Support Vevtor Machines)

%#######################
% Automatic Cross Validation 
% Parameter selection using n-fold cross validation
%#######################
stepSize = 10;
bestLog2c = 1;
bestLog2g = -1;
epsilon = 0.01;
bestcv = 0;
Ncv = 3;       % Ncv-fold cross validation
deltacv = 10^6;
t = 0;

while abs(deltacv) > epsilon
    bestcv_prev = bestcv;
    prevStepSize = stepSize;
    stepSize = prevStepSize/2;
    log2c_list = bestLog2c-prevStepSize:stepSize/2:bestLog2c+prevStepSize;
    log2g_list = bestLog2g-prevStepSize:stepSize/2:bestLog2g+prevStepSize;
    
    numLog2c = length(log2c_list);
    numLog2g = length(log2g_list);
    cvMatrix = zeros(numLog2c,numLog2g);
    
    for i = 1:numLog2c
        log2c = log2c_list(i);
        for j = 1:numLog2g
            log2g = log2g_list(j);
            cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
            cv = get_cv_ac(train_Y, train_X, cmd, Ncv);
            if (cv >= bestcv),
                bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
            end
        end
    end
    deltacv = bestcv - bestcv_prev;
    
end
disp(['The best parameters, yielding Accuracy=',num2str(bestcv*100),'%, are: C=',num2str(bestc),', gamma=',num2str(bestg)]);

% #######################
% Train the SVM in one-vs-rest (OVR) mode
% #######################
bestc = 2;
bestg = 16;
bestParam = ['-q -c ', num2str(bestc), ', -g ', num2str(bestg)];
% bestParam = ['-q -c 8 -g 0.0625'];
model = ovrtrainBot(train_Y, train_X, bestParam);

% FOR VALIDATION : 
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

%% Cite:
% 1. libsvm for MATLAB?, Kittipat's Homepage
%    https://sites.google.com/site/kittipat/libsvm_matlab/complete_libsvm_example

%% CLASSIFIER NO. 2 : NN (Neural Networks)

%% Notes:
% 70 - 15 - 15
% 1 HL = Min error = 0.0892 | no. of units (neurons) = 265         | Runtime = 358  secs.
% 2 HL = Min error = 0.0693 | no. of units (neurons) = 305,65      | Runtime = 4.89 hours
% 3 HL = Min error = 0.0812 | no. of units (neurons) = 290,215,395 | Runtime = 3.95 hours

% 50 - 5 - 45
% 1 HL = Min error = 0.1345 | no. of units (neurons) = 355          | Runtime = 280  secs.
% 2 HL = Min error = 0.1150 | no. of units (neurons) = 685,645      | Runtime = 4.04 hours
% 3 HL = Min error = 0.1239 | no. of units (neurons) = 525,615,675  | Runtime = 7.42 hours

%% Start parameters
%//%************************************************************************%
tic;
clear all; close all; clc;
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);

%% This script assumes these variables are defined:
%//%************************************************************************%
%   Feature_matrix - input data
%   Labels - target data

% Change of variables
% [ 1 2 3 4 5 6 6 7 7 8 8 8 ...... 12]
% [ 1 2 3 4 5 6 6 7 7 8 8 8 ...... 12]
% [ 1 2 3 4 5 6 6 7 7 8 8 8 ...... 12]
%  . . . . . . . . . . . . . . . . . .
% [ 1 2 3 4 5 6 6 7 7 8 8 8 ...... 12]
% [ 1 2 3 4 5 6 6 7 7 8 8 8 ...... 12]
% rows x columns [n x m]
% n - samples; m - features (requires transpose). Similarly, to
% labels/targets
% If not the above format, then no need to transpose

% Feature matrix (transposed) and % Labels/targets (transposed)
load ZZZ_imageSetsFinalFeatMatLabels.mat
x = [Xtrain; Xval; XTest]';
t = [Ttrain; Tval; TTest]';
% t = [Ytrain; Yval; YTest]';

% x = (compute_mapping(x', 'PCA', 50))';
% Training, validation and testing size (1 to 100%)
% Caution: Total shall make 100%
% train_size = 50; 
% valid_size = 5; 
% test_size  = 45; 

% Number of hidden layers [maximum 3]
hidden_layers = 1;

% Hidden layer size
% hiddenLayerSize_Vec = 5:5:1000;   % 1HL
% hiddenLayerSize_Vec   = 30:90:1000;   % 2HL
hiddenLayerSize_Vec = 20:100:800;   % 3 HL

% Window view / plotting options [on | off]
plotter      = 'no';
viewfinalnet = 'off';

%% Create hidden layers (>1) neuron units combinations
%//%************************************************************************%
switch hidden_layers    
    case 1
        pairs = hiddenLayerSize_Vec(:);
        
    case 2
        [p,q] = meshgrid(hiddenLayerSize_Vec, hiddenLayerSize_Vec);
        pairs = [p(:) q(:)];
        
    case 3
        [p,q,r] = meshgrid(hiddenLayerSize_Vec, hiddenLayerSize_Vec, hiddenLayerSize_Vec);
        pairs = [p(:) q(:) r(:)];     
end

%% Train the network
%//%************************************************************************%

% Waitbar handler
h = waitbar(0,'Initializing...','Name','Finding optimum number of neurons...!',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0)

% Main loop

for step = 1 : 1 %length(pairs)
    
    % Check for Cancel button press
    if getappdata(h,'canceling')
        break
    end
            
    % Create a Pattern Recognition Network
    net = patternnet(300);
    
    % Network options
    %//%*******************************************************************
    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
  
    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'divideind';  % Divide data randomly
    net.divideMode = 'none';  % Divide up every sample
    net.divideParam.trainInd = 1 : size(Xtrain,1);

    net.divideParam.valInd   = size(Xtrain,1) + 1 : ...
                               size(Xtrain,1) + size(Xval,1);

    net.divideParam.testInd  = size(Xtrain,1) + size(Xval,1) + 1 :...
                               size(Xtrain,1) + size(Xval,1) + ...
                               size(XTest,1);
                           
    % Goal
    net.trainParam.goal = 1e-3;
    
    % Change the transfer function for all hidden layers [output layer in deafult 'softmax']
    for i = 1:size(pairs,2)
        net.layers{i}.transferFcn = 'tansig';
    end

    % Turn on/off nntraintoll window
    net.trainParam.showWindow = 0;    

    % Callback of neural netwrok function
    % Train the Network
    [net,tr] = train(net,x,t,'useParallel','yes','useGPU','yes');

    % Test the Network
    y = net(x,'useParallel','yes','useGPU','yes');
    e = gsubtract(t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    performance = perform(net,t,y);

    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    valTargets = t  .* tr.valMask{1};
    testTargets = t  .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y);
    valPerformance = perform(net,valTargets,y);
    testPerformance = perform(net,testTargets,y);
    
%     [netOutput, tr, test_Errors, overallErrors] = funct_NNtrain_test (net, x, t);   
%     netOutputTotal(step).network = net;
%     trTotal(step).trainNN        = tr;
    
    % Store error outputs   
    NNerrDetails(step).testErrTotal = percentErrors;
    NNerrDetails(step).neuronUnits  = pairs(step,:);
    
    % Report current estimate in the waitbar's message field
    waitbar(step/length(pairs), h, sprintf('Hidden layer units itr. = %i | Err: %1.3f',...
            step, percentErrors))
end

% DELETE the waitbar; don't try to CLOSE it
delete(h)       

%% View network
if (strcmp(viewfinalnet, 'on'))
    view (finalnet)
end

% Close nntraintool window
nntraintool('close');

%% End parameters
%--------------------------------------------------------------------------
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
Runtime = toc(Start);
