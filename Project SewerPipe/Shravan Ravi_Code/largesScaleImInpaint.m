function largesScaleImInpaint( imSetImgLocation )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

mask = zeros (181,352);
mask(145:170 , 51:101)    = 255;
mask(145:170 , 132:215)   = 255;
mask(145:170 , 246:318)   = 255;

h = waitbar(0,'1','Name','Image Inpainting',...
            'CreateCancelBtn',...
            'setappdata(gcbf,''canceling'',1)');
setappdata(h,'canceling',0)

for i = 1:length(imSetImgLocation)
    
    % Check for Cancel button press
    if getappdata(h,'canceling')
        break
    end
    % Report current estimate in the waitbar's message field
    waitbar(i/length(imSetImgLocation),h,sprintf('Image no.: %i',i))
    
    I = imread (cell2mat(imSetImgLocation(i)));
    cropped_img = imcrop (I , [0 60 352 240]);
    inpaintImage = inpaint ( cropped_img , mask );
    imwrite(uint8(inpaintImage),cell2mat(imSetImgLocation(i)))
end
delete(h)       % DELETE the waitbar; don't try to CLOSE it.


end

