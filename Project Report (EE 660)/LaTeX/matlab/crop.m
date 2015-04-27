function [cropped_img , mask] = crop (input_img)
% Description

I = imread (input_img);
cropped_img = imcrop (I , [0 60 352 240]);

M = zeros (181,352);
M(145:170 , 51:101)   = 255;
M(145:170 , 132:215)   = 255;
M(145:170 , 246:318)   = 255;
mask = M;

% imwrite (input_img , input_img '.jpg' , '.jpg');

% M_r = cropped_img(: , : , 1);
% M_g = cropped_img(: , : , 2);
% M_b = cropped_img(: , : , 3);
% 
% M_r(145:170 , 51:101)   = 255;
% M_r(145:170 , 132:215)   = 255;
% M_r(145:170 , 246:318)   = 255;
% 
% M_g(145:170 , 51:101)   = 255;
% M_g(145:170 , 132:215)   = 255;
% M_g(145:170 , 246:318)   = 255;
% 
% M_b(145:170 , 51:101)   = 255;
% M_b(145:170 , 132:215)   = 255;
% M_b(145:170 , 246:318)   = 255;
% % M_b(~(145:170 , 51:101)) && M_b(~(145:170 , 132:215)) && M_b(~(145:170 , 246:318)) = 0;
% 
% mask_r = M_r;
% mask_g = M_g;
% mask_b = M_b;
end
