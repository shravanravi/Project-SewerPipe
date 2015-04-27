function [ Feature_matrix, Labels, Target, indexMap ] = shuffleFeatMatLabel( featMat, labelArray )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

ix = randperm(numel(labelArray));

% Populate the matrices
Feature_matrix  = featMat(ix,:);
Labels          = labelArray(ix,:);    
Target = full(ind2vec(Labels'))';

% Shuffled index mapping
indexMap = [labelArray, ix'];
end

