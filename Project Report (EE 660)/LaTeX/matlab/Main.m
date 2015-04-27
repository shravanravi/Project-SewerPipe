%//%************************************************************************%
%//%*                              Ph.D                                    *%
%//%*                         Project Sewer Pipe						   *%
%//%*                                                                      *%
%//%*             Name: Preetham Aghalaya Manjunatha    		           *%
%//%*                   Shravan Ravi    		           *%
%//%*             USC ID Number: 7356627445		                           *%
%//%*             USC Email: aghalaya@usc.edu                              *%
%//%*             Submission Date: 12/08/2012                              *%
%//%************************************************************************%
%//%*             Viterbi School of Engineering,                           *%
%//%*             Sonny Astani Dept. of Civil Engineering,                 *%
%//%*             University of Southern california,                       *%
%//%*             Los Angeles, California.                                 *%
%//%************************************************************************%

%% Start parameters
%--------------------------------------------------------------------------
clear all; close all; clc; %#ok<CLSCR>
Start = tic;
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);

%% Inputs
%--------------------------------------------------------------------------
MainInputs;


%% Pre-processing steps
%--------------------------------------------------------------------------
% Create image sets
% if (shuffleNpartfiles_inpstruct.flagswitch && ~ exist('ZZZ_imageSetsFull.mat','file'))
%     imgPartitionStruct = makePartitions( shuffleNpartfiles_inpstruct );
% else
%     load ZZZ_imageSetsFull.mat
% end

% Image inpainting
% if (cropNinpaint_inpstruct.flagswitch)
%     largesScaleImInpaint( imgPartitionStruct.imSetImgLocFull );
% end

%% Processing step
%--------------------------------------------------------------------------
% Extract feature matrix and class labels
%--------------------------------------------------------------------------

% A hybrid algorithm (RoboCRACK)
% if (largefeatmatlabel_inpstruct.flagswitch)
%     featlab_traintestset = largefeaturematrixNlabels ...
%         (hybrid_inpstruct, imgPartitionStruct);
% end

% Bag-of-Features
% bag_Tr_BoFOri   = bagOfFeatures(imgPartitionStruct.trainingSets);
% featureMatrixTr_BoFOri = encode(bag_Tr_BoFOri, imgPartitionStruct.trainingSets);
% labelArrayTr_BoFOri    = makeLabelArray(imgPartitionStruct.trainingSets);
% 
% bag_Vl_BoFOri   = bagOfFeatures(imgPartitionStruct.validationSets);
% featureMatrixVl_BoFOri = encode(bag_Vl_BoFOri, imgPartitionStruct.validationSets);
% labelArrayVl_BoFOri    = makeLabelArray(imgPartitionStruct.validationSets);
% 
% bag_Ts_BoFOri   = bagOfFeatures(imgPartitionStruct.trainingSets);
% featureMatrixTs_BoFOri = encode(bag_Ts_BoFOri, imgPartitionStruct.testingSets);
% labelArrayTs_BoFOri    = makeLabelArray(imgPartitionStruct.testingSets);

%--------------------------------------------------------------------------
% Training, validation and testing
%--------------------------------------------------------------------------

% A hybrid algorithm (RoboCRACK)

% Bag-of-Features
load ZZZ_imageSetsFull_BoF_Original.mat
categoryClassifier = trainImageCategoryClassifier(imgPartitionStruct.trainingSets, bag_Tr_BoFOri);
[confMatTr,knownLabelIdxTr,predictedLabelIdxTr,scoreTr] = evaluate(categoryClassifier, ...
                                                    imgPartitionStruct.trainingSets);
                                                
[confMatVl,knownLabelIdxVl,predictedLabelIdxVl,scoreVl] = evaluate(categoryClassifier, ...
                                                    imgPartitionStruct.validationSets);

[confMatTs,knownLabelIdxTs,predictedLabelIdxTs,scoreTs] = evaluate(categoryClassifier, ...
                                                    imgPartitionStruct.testingSets);
                                                
%% Post-processing steps
%--------------------------------------------------------------------------

    
%% End parameters
%--------------------------------------------------------------------------
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);
Runtime = toc(Start);
