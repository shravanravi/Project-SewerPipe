[ Xtrain, Ytrain, Ttrain, ImageShuffIndexMap_train ] ...
    = shuffleFeatMatLabel(featureMatrixTr_BoFOri ,  labelArrayTr_BoFOri);

[ Xval, Yval, Tval, ImageShuffIndexMap_val ] ...
    = shuffleFeatMatLabel(featureMatrixVl_BoFOri ,  labelArrayVl_BoFOri);


[ XTest, YTest, TTest, ImageShuffIndexMap_test ] ...
    = shuffleFeatMatLabel(featureMatrixTs_BoFOri ,  labelArrayTs_BoFOri);

