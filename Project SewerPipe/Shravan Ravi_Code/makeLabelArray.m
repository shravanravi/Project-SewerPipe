function [ labelArray ] = makeLabelArray( imgSets )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

imSetTotalNum = sum(cat(1, imgSets.Count));
labelArray    = zeros (imSetTotalNum, 1);
count = 1;
for i = 1:length(imgSets)
    labelArray(count : count + imgSets(i).Count-1,:) = i;
    count = imgSets(i).Count + count;
end

end

