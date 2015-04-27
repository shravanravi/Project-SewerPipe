%//%************************************************************************%
%//%*                              Ph.D                                    *%
%//%*                           Crack Package						       *%
%//%*                                                                      *%
%//%*             Name: Preetham Aghalaya Manjunatha    		           *%
%//%*             USC ID Number: 7356627445		                           *%
%//%*             USC Email: aghalaya@usc.edu                              *%
%//%*             Submission Date: --/--/2012                              *%
%//%************************************************************************%
%//%*             Viterbi School of Engineering,                           *%
%//%*             Sonny Astani Dept. of Civil Engineering,                 *%
%//%*             University of Southern california,                       *%
%//%*             Los Angeles, California.                                 *%
%//%************************************************************************%
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

%% End parameters
% Close figures, waitbars and all
clcwaitbarz = findall(0,'type','figure','tag','TMWWaitbar');
delete(clcwaitbarz);

% Close nntraintool window
nntraintool('close');

% Runtime
Runtime = toc;
