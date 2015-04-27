%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Shuffle and partition params

%--------------------------------------------------------------------------
% Shuffle flag
%--------------------------------------------------------------------------
shuffleNpartfiles_inpstruct.flagswitch             = 1;   %[0-off | 1-on]

%--------------------------------------------------------------------------
% Training percentage
% [train val test] - need first two values
%--------------------------------------------------------------------------
shuffleNpartfiles_inpstruct.splitRatio   = [0.6 0.1];

%--------------------------------------------------------------------------
% File renaming
% seqnum | actual -- string datatype
%--------------------------------------------------------------------------
shuffleNpartfiles_inpstruct.splitShuffleType       = 'randomized';

%--------------------------------------------------------------------------
% Folder paths for ground-truth, original. Train, test corresponds to the 
% new repository folder to store
%--------------------------------------------------------------------------

% Ground-truth
shuffleNpartfiles_inpstruct.imgFolder              = 'E:\Google Drive\Team SewerPipe\Programs\Main\data';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Texture feature vector params

%--------------------------------------------------------------------------
% Shuffle flag
%--------------------------------------------------------------------------
texturefeature_inpstruct.flag_texturefeatures    = 0;   %[0-off | 1-on]

%--------------------------------------------------------------------------
% Training images (data samples) to be considered
%--------------------------------------------------------------------------
% full - uses all images in train folders (string)
% half - uses all images in train folders (string)
% n    - n number of images (integer)
texturefeature_inpstruct.imagenumbers            = 'full';

%--------------------------------------------------------------------------
% Texture filter
%--------------------------------------------------------------------------
% laws - Laws multi-channel filter (level, edge, spot, wave and ripple)
% sfta - Segmentation-based Fractal Texture Analysis
texturefeature_inpstruct.texturefilter_type      = 'laws';

%--------------------------------------------------------------------------
% GPU array
%--------------------------------------------------------------------------
% yes - creates GPU array (note: works for certain Matlab functions)
% no  - non GPU array
texturefeature_inpstruct.gpuarray                = 'no';

%--------------------------------------------------------------------------
% Norm type
%--------------------------------------------------------------------------
% L1        - L1 norm
% L2        - l2 norm
% infinity  - infinity norm
% frobenius - frobenius norm
texturefeature_inpstruct.normtype                = 'frobenius';

%--------------------------------------------------------------------------
% Window size (energy)
%--------------------------------------------------------------------------
% n    - odd integer value (such as 3, 5, 7, 11, 13, 15, 17, ...)
texturefeature_inpstruct.windowsize              = 15;

%--------------------------------------------------------------------------
% Training data folder path
%--------------------------------------------------------------------------
texturefeature_inpstruct.folderpath_texture      = [];

%--------------------------------------------------------------------------
% Provide names of original data folders (include name of the folder even its folder
% is empty)
%--------------------------------------------------------------------------
texturefeature_inpstruct.originaldata_folders    = {'', ''};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Crop and image inpaint params

%--------------------------------------------------------------------------
% On/off flag
%--------------------------------------------------------------------------
cropNinpaint_inpstruct.flagswitch      = 0;   %[0-off | 1-on]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training and testing feature matrix and labels parameters

%--------------------------------------------------------------------------
% Shuffle flag
%--------------------------------------------------------------------------
% For train and test data
hybrid_inpstruct.flagswitch              = 1;   %[0-off | 1-on]

% Save feature matrix and labels in image info structure (as a variable).
% Else store n samples in sequential .mat files
hybrid_inpstruct.savefeaturematrixNlabels ...
                                                    = 1;   %[0-off | 1-on]    
                                                
%--------------------------------------------------------------------------
% Training images (data samples) to be considered
%--------------------------------------------------------------------------
% full - uses all images in train folders (string)
% half - uses all images in train folders (string)
% n    - n number of images (integer)
hybrid_inpstruct.imagenumbers            = 'full';

% Turn on/off adaptive histogram
hybrid_inpstruct.adaphist                = 'off';

%--------------------------------------------------------------------------
% Training data folder path
%--------------------------------------------------------------------------
hybrid_inpstruct.folderpath              = 'E:\Google Drive\Team SewerPipe\Programs\Main\data';

%--------------------------------------------------------------------------
% Provide names of groundtruth folders (include pseudoname even if ground-truth doesn't 
% exist for a original folder, e.g. nocrack --> ground_nocrack. Also if the folder
% is empty)
%--------------------------------------------------------------------------
hybrid_inpstruct.groundtruth_folders     = {};

%--------------------------------------------------------------------------
% Provide names of original data folders (include name of the folder even its folder
% is empty)
%--------------------------------------------------------------------------
hybrid_inpstruct.originaldata_folders    = {};

%--------------------------------------------------------------------------
% GPU array
%--------------------------------------------------------------------------
% yes - creates GPU array (note: works for certain Matlab functions)
% no  - non GPU array
hybrid_inpstruct.gpuarray                = 'no';

%--------------------------------------------------------------------------
% Colorspace segmentation options 
%--------------------------------------------------------------------------
% Type of colorspace to segment RGB ground-truths
% HSV (recommended) or RGB
hybrid_inpstruct.colorspace = 'hsv';  %[hsv | rgb]

% RGB startindex
% n   - channel value (integer [0, 255])
hybrid_inpstruct.RGBstartindex           = 235;

%--------------------------------------------------------------------------
% Anisotropic diffusion parameters
%--------------------------------------------------------------------------
% Stage I and Stage II
hybrid_inpstruct.aniso.num_iter = [10, 10];
hybrid_inpstruct.aniso.delta_t  = [1/7, 1/7];
hybrid_inpstruct.aniso.kappa    = [5, 5];
hybrid_inpstruct.aniso.option   = [1, 1];

%--------------------------------------------------------------------------
% Hessian matrix parameters
%--------------------------------------------------------------------------
% Frangi filter options
hybrid_inpstruct.frangiopt.FrangiScaleRange = [1 15];
hybrid_inpstruct.frangiopt.FrangiScaleRatio = 1;
hybrid_inpstruct.frangiopt.FrangiBetaOne    = 0.5;
hybrid_inpstruct.frangiopt.FrangiBetaTwo    = 15;
hybrid_inpstruct.frangiopt.BlackWhite       = 1;
hybrid_inpstruct.frangiopt.verbose          = 0;

%--------------------------------------------------------------------------
% Blob removal parameters
%--------------------------------------------------------------------------
% Blob filter standard deviation scale
hybrid_inpstruct.blobfilter_sigma    = 0.3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%