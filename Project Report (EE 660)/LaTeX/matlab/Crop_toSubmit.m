function [cropped_img , mask] = crop (input_img)

I = imread (input_img);
cropped_img = imcrop (I , [0 60 352 240]);

M = zeros (181,352);
M(145:170 , 51:101)   = 255;
M(145:170 , 132:215)   = 255;
M(145:170 , 246:318)   = 255;
mask = M;
end
