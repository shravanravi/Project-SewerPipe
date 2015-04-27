function shuffleNcopy(input)

    % Get the ground truth file names
    if (isempty(input.folderpath_ground))
        Allfilenames = dir (input.folderpath_original);
    else
        Allfilenames = dir (input.folderpath_ground);
    end

    % Filtered file names
    names = setdiff({Allfilenames.name},{'.','..', 'desktop.ini', 'thumbs.db'});

    % Sort the files by name
    names_sorted = filenamesort(names);

    % Random permutation
    ix = randperm(length(names_sorted));
    Shuffled_filenames  = names_sorted(ix);

    % Training data
    % From Shuffled_filenames (1) to Shuffled_filenames (N * input.train_percent/100)
    train_maxIndex = ceil(length(Shuffled_filenames) * input.trainValTest_percent(1) / 100);


    % Condition to check between ground truth or original folder only
    if (isempty(input.folderpath_ground) | isempty(input.folderpath_gtrain) | isempty(input.folderpath_gtest))
        % Delete files from a folder(s)
        delete(fullfile(input.folderpath_train,  '*.*'));
        delete(fullfile(input.folderpath_test,   '*.*'));    

        % Copyfile in a loop
        for i = 1:length(Shuffled_filenames)    
            if (i <= train_maxIndex)

                % Train set images from original folder
                switch input.filenameRenaming
                    case 'actual'
                        copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_train,    Shuffled_filenames{i}),'f');
                    case 'seqnumb'
                        outputBaseFileName = sprintf('%3.3d.png', i);
                        copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_train,    outputBaseFileName),'f');
                end
            else

                % Test set images from original folder
                switch input.filenameRenaming
                    case 'actual'
                        copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_test,    Shuffled_filenames{i}),'f');
                    case 'seqnumb'
                        outputBaseFileName = sprintf('%3.3d.png', i);
                        copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_test,    outputBaseFileName),'f');
                end

            end
        end   

    else
        % Delete files from a folder(s)
        delete(fullfile(input.folderpath_train,  '*.*'));
        delete(fullfile(input.folderpath_gtrain, '*.*'));
        delete(fullfile(input.folderpath_test,   '*.*'));
        delete(fullfile(input.folderpath_gtest,  '*.*'));

        % Copyfile in a loop
        for i = 1:length(Shuffled_filenames)    
            if (i <= train_maxIndex)

                % Train set images from original folder
                switch input.filenameRenaming
                    case 'actual'
                    copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_train,    Shuffled_filenames{i}),'f');
                    case 'seqnumb'
                        outputBaseFileName = sprintf('%3.3d.png', i);
                        copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_train,    outputBaseFileName),'f');
                end


                % Train set images from ground-truth folder   
                switch input.filenameRenaming
                    case 'actual'
                    copyfile(fullfile(input.folderpath_ground, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_gtrain, Shuffled_filenames{i}),'f');
                    case 'seqnumb'
                        outputBaseFileName = sprintf('%3.3d.png', i);
                        copyfile(fullfile(input.folderpath_ground, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_gtrain,    outputBaseFileName),'f');
                end

            else

                % Test set images from original folder
                switch input.filenameRenaming
                    case 'actual'
                    copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_test, Shuffled_filenames{i}),'f');
                    case 'seqnumb'
                        outputBaseFileName = sprintf('%3.3d.png', i);
                        copyfile(fullfile(input.folderpath_original, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_test,    outputBaseFileName),'f');
                end


                % Test set images from ground-truth folder     
                switch input.filenameRenaming
                    case 'actual'
                    copyfile(fullfile(input.folderpath_ground, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_gtest, Shuffled_filenames{i}),'f');
                    case 'seqnumb'
                        outputBaseFileName = sprintf('%3.3d.png', i);
                        copyfile(fullfile(input.folderpath_ground, Shuffled_filenames{i}), ...
                             fullfile(input.folderpath_gtest,    outputBaseFileName),'f');
                end

            end
        end
    end   
end