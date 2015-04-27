% function plot3Dbargraph ()
close all; clc


    % Plot the 3D bar graph
%     load 'ZZZ_imageSetsFull.mat'
    imgFull =  [imgPartitionStruct.imgSets.Count];
    trnFull =  [imgPartitionStruct.trainingSets.Count];
    valFull =  [imgPartitionStruct.validationSets.Count];
    tstFull =  [imgPartitionStruct.testingSets.Count];

    Y = [valFull; tstFull; trnFull; imgFull]';
    
    % Plot 3D graph
    str = {'Val'; 'Test'; 'Train'; 'Full'};
    figure
    h = bar3(Y);
%     colorbar
%     for k = 1:length(h)
%         zdata = h(k).ZData;
%         h(k).CData = zdata;
%         h(k).FaceColor = 'interp';
%     end
    title('Class Distribution')    
    set(gca, 'PlotBoxAspectRatioMode','auto')
    set(gca, 'XTickLabel',str, 'XTick', 1:numel(str), ...
             'YTickLabel',(1:9), 'YTick', (1:9),...
             'FontSize', 18, 'LineWidth', 3)
    ylabel ('Class Labels', 'FontSize', 18)
    zlabel ('Number of Images', 'FontSize', 18)
    legend ('Val', 'Test', 'Train', 'Full')
%     axis tight; 


    % resize the figure window
%     set(hFig, 'units','normalized','outerposition',[0 0 1 1])

    % Export figure to some format
%     export_fig('fig_texture_counts_.pdf',...
%         '-pdf', '-transparent', gcf);
    
% end