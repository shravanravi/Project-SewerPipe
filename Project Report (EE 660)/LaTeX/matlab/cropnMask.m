function [cropped_img , mask] = cropnMask (input_img)
% Description

I = imread (input_img);
cropped_img = imcrop (I , [0 60 352 240]);


end
