%cnn分析
%搭建網路
tic

%數據導入
clear;clc;
digitDatasePath = "C:\Users\joe\Downloads\論文需求\論文需求\檢測線\擴增"
imds = imageDatastore(digitDatasePath, 'IncludeSubfolders',true, 'LabelSource','foldernames');

% 調整圖像大小
imds.ReadFcn = @(filename)repmat(imresize(im2gray(imread(filename)), [227, 227]), 1, 1, 3);

%劃分訓練集與測試集
[imgTrain,imgTest] = splitEachLabel(imds,0.8,'randomize');


%預訓練參數
params = load("C:\Users\joe\Desktop\專題\params_2023_08_21__23_08_11.mat");

%搭建神經網路
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([7 7],64,"Name","conv1","Padding",[3 3 3 3],"Stride",[2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    batchNormalizationLayer("Name","bn_conv1","Epsilon",0.001,"Offset",params.bn_conv1.Offset,"Scale",params.bn_conv1.Scale,"TrainedMean",params.bn_conv1.TrainedMean,"TrainedVariance",params.bn_conv1.TrainedVariance)
    reluLayer("Name","activation_1_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res2a_branch1","BiasLearnRateFactor",0,"Bias",params.res2a_branch1.Bias,"Weights",params.res2a_branch1.Weights)
    batchNormalizationLayer("Name","bn2a_branch1","Epsilon",0.001,"Offset",params.bn2a_branch1.Offset,"Scale",params.bn2a_branch1.Scale,"TrainedMean",params.bn2a_branch1.TrainedMean,"TrainedVariance",params.bn2a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Bias",params.res2a_branch2a.Bias,"Weights",params.res2a_branch2a.Weights)
    batchNormalizationLayer("Name","bn2a_branch2a","Epsilon",0.001,"Offset",params.bn2a_branch2a.Offset,"Scale",params.bn2a_branch2a.Scale,"TrainedMean",params.bn2a_branch2a.TrainedMean,"TrainedVariance",params.bn2a_branch2a.TrainedVariance)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res2a_branch2b.Bias,"Weights",params.res2a_branch2b.Weights)
    batchNormalizationLayer("Name","bn2a_branch2b","Epsilon",0.001,"Offset",params.bn2a_branch2b.Offset,"Scale",params.bn2a_branch2b.Scale,"TrainedMean",params.bn2a_branch2b.TrainedMean,"TrainedVariance",params.bn2a_branch2b.TrainedVariance)
    reluLayer("Name","activation_3_relu")
    convolution2dLayer([1 1],256,"Name","res2a_branch2c","BiasLearnRateFactor",0,"Bias",params.res2a_branch2c.Bias,"Weights",params.res2a_branch2c.Weights)
    batchNormalizationLayer("Name","bn2a_branch2c","Epsilon",0.001,"Offset",params.bn2a_branch2c.Offset,"Scale",params.bn2a_branch2c.Scale,"TrainedMean",params.bn2a_branch2c.TrainedMean,"TrainedVariance",params.bn2a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Bias",params.res2b_branch2a.Bias,"Weights",params.res2b_branch2a.Weights)
    batchNormalizationLayer("Name","bn2b_branch2a","Epsilon",0.001,"Offset",params.bn2b_branch2a.Offset,"Scale",params.bn2b_branch2a.Scale,"TrainedMean",params.bn2b_branch2a.TrainedMean,"TrainedVariance",params.bn2b_branch2a.TrainedVariance)
    reluLayer("Name","activation_5_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res2b_branch2b.Bias,"Weights",params.res2b_branch2b.Weights)
    batchNormalizationLayer("Name","bn2b_branch2b","Epsilon",0.001,"Offset",params.bn2b_branch2b.Offset,"Scale",params.bn2b_branch2b.Scale,"TrainedMean",params.bn2b_branch2b.TrainedMean,"TrainedVariance",params.bn2b_branch2b.TrainedVariance)
    reluLayer("Name","activation_6_relu")
    convolution2dLayer([1 1],256,"Name","res2b_branch2c","BiasLearnRateFactor",0,"Bias",params.res2b_branch2c.Bias,"Weights",params.res2b_branch2c.Weights)
    batchNormalizationLayer("Name","bn2b_branch2c","Epsilon",0.001,"Offset",params.bn2b_branch2c.Offset,"Scale",params.bn2b_branch2c.Scale,"TrainedMean",params.bn2b_branch2c.TrainedMean,"TrainedVariance",params.bn2b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_2")
    reluLayer("Name","activation_7_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","res2c_branch2a","BiasLearnRateFactor",0,"Bias",params.res2c_branch2a.Bias,"Weights",params.res2c_branch2a.Weights)
    batchNormalizationLayer("Name","bn2c_branch2a","Epsilon",0.001,"Offset",params.bn2c_branch2a.Offset,"Scale",params.bn2c_branch2a.Scale,"TrainedMean",params.bn2c_branch2a.TrainedMean,"TrainedVariance",params.bn2c_branch2a.TrainedVariance)
    reluLayer("Name","activation_8_relu")
    convolution2dLayer([3 3],64,"Name","res2c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res2c_branch2b.Bias,"Weights",params.res2c_branch2b.Weights)
    batchNormalizationLayer("Name","bn2c_branch2b","Epsilon",0.001,"Offset",params.bn2c_branch2b.Offset,"Scale",params.bn2c_branch2b.Scale,"TrainedMean",params.bn2c_branch2b.TrainedMean,"TrainedVariance",params.bn2c_branch2b.TrainedVariance)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([1 1],256,"Name","res2c_branch2c","BiasLearnRateFactor",0,"Bias",params.res2c_branch2c.Bias,"Weights",params.res2c_branch2c.Weights)
    batchNormalizationLayer("Name","bn2c_branch2c","Epsilon",0.001,"Offset",params.bn2c_branch2c.Offset,"Scale",params.bn2c_branch2c.Scale,"TrainedMean",params.bn2c_branch2c.TrainedMean,"TrainedVariance",params.bn2c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_3")
    reluLayer("Name","activation_10_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res3a_branch2a.Bias,"Weights",params.res3a_branch2a.Weights)
    batchNormalizationLayer("Name","bn3a_branch2a","Epsilon",0.001,"Offset",params.bn3a_branch2a.Offset,"Scale",params.bn3a_branch2a.Scale,"TrainedMean",params.bn3a_branch2a.TrainedMean,"TrainedVariance",params.bn3a_branch2a.TrainedVariance)
    reluLayer("Name","activation_11_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3a_branch2b.Bias,"Weights",params.res3a_branch2b.Weights)
    batchNormalizationLayer("Name","bn3a_branch2b","Epsilon",0.001,"Offset",params.bn3a_branch2b.Offset,"Scale",params.bn3a_branch2b.Scale,"TrainedMean",params.bn3a_branch2b.TrainedMean,"TrainedVariance",params.bn3a_branch2b.TrainedVariance)
    reluLayer("Name","activation_12_relu")
    convolution2dLayer([1 1],512,"Name","res3a_branch2c","BiasLearnRateFactor",0,"Bias",params.res3a_branch2c.Bias,"Weights",params.res3a_branch2c.Weights)
    batchNormalizationLayer("Name","bn3a_branch2c","Epsilon",0.001,"Offset",params.bn3a_branch2c.Offset,"Scale",params.bn3a_branch2c.Scale,"TrainedMean",params.bn3a_branch2c.TrainedMean,"TrainedVariance",params.bn3a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res3a_branch1.Bias,"Weights",params.res3a_branch1.Weights)
    batchNormalizationLayer("Name","bn3a_branch1","Epsilon",0.001,"Offset",params.bn3a_branch1.Offset,"Scale",params.bn3a_branch1.Scale,"TrainedMean",params.bn3a_branch1.TrainedMean,"TrainedVariance",params.bn3a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_4")
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Bias",params.res3b_branch2a.Bias,"Weights",params.res3b_branch2a.Weights)
    batchNormalizationLayer("Name","bn3b_branch2a","Epsilon",0.001,"Offset",params.bn3b_branch2a.Offset,"Scale",params.bn3b_branch2a.Scale,"TrainedMean",params.bn3b_branch2a.TrainedMean,"TrainedVariance",params.bn3b_branch2a.TrainedVariance)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3b_branch2b.Bias,"Weights",params.res3b_branch2b.Weights)
    batchNormalizationLayer("Name","bn3b_branch2b","Epsilon",0.001,"Offset",params.bn3b_branch2b.Offset,"Scale",params.bn3b_branch2b.Scale,"TrainedMean",params.bn3b_branch2b.TrainedMean,"TrainedVariance",params.bn3b_branch2b.TrainedVariance)
    reluLayer("Name","activation_15_relu")
    convolution2dLayer([1 1],512,"Name","res3b_branch2c","BiasLearnRateFactor",0,"Bias",params.res3b_branch2c.Bias,"Weights",params.res3b_branch2c.Weights)
    batchNormalizationLayer("Name","bn3b_branch2c","Epsilon",0.001,"Offset",params.bn3b_branch2c.Offset,"Scale",params.bn3b_branch2c.Scale,"TrainedMean",params.bn3b_branch2c.TrainedMean,"TrainedVariance",params.bn3b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_5")
    reluLayer("Name","activation_16_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3c_branch2a","BiasLearnRateFactor",0,"Bias",params.res3c_branch2a.Bias,"Weights",params.res3c_branch2a.Weights)
    batchNormalizationLayer("Name","bn3c_branch2a","Epsilon",0.001,"Offset",params.bn3c_branch2a.Offset,"Scale",params.bn3c_branch2a.Scale,"TrainedMean",params.bn3c_branch2a.TrainedMean,"TrainedVariance",params.bn3c_branch2a.TrainedVariance)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],128,"Name","res3c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3c_branch2b.Bias,"Weights",params.res3c_branch2b.Weights)
    batchNormalizationLayer("Name","bn3c_branch2b","Epsilon",0.001,"Offset",params.bn3c_branch2b.Offset,"Scale",params.bn3c_branch2b.Scale,"TrainedMean",params.bn3c_branch2b.TrainedMean,"TrainedVariance",params.bn3c_branch2b.TrainedVariance)
    reluLayer("Name","activation_18_relu")
    convolution2dLayer([1 1],512,"Name","res3c_branch2c","BiasLearnRateFactor",0,"Bias",params.res3c_branch2c.Bias,"Weights",params.res3c_branch2c.Weights)
    batchNormalizationLayer("Name","bn3c_branch2c","Epsilon",0.001,"Offset",params.bn3c_branch2c.Offset,"Scale",params.bn3c_branch2c.Scale,"TrainedMean",params.bn3c_branch2c.TrainedMean,"TrainedVariance",params.bn3c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_6")
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3d_branch2a","BiasLearnRateFactor",0,"Bias",params.res3d_branch2a.Bias,"Weights",params.res3d_branch2a.Weights)
    batchNormalizationLayer("Name","bn3d_branch2a","Epsilon",0.001,"Offset",params.bn3d_branch2a.Offset,"Scale",params.bn3d_branch2a.Scale,"TrainedMean",params.bn3d_branch2a.TrainedMean,"TrainedVariance",params.bn3d_branch2a.TrainedVariance)
    reluLayer("Name","activation_20_relu")
    convolution2dLayer([3 3],128,"Name","res3d_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res3d_branch2b.Bias,"Weights",params.res3d_branch2b.Weights)
    batchNormalizationLayer("Name","bn3d_branch2b","Epsilon",0.001,"Offset",params.bn3d_branch2b.Offset,"Scale",params.bn3d_branch2b.Scale,"TrainedMean",params.bn3d_branch2b.TrainedMean,"TrainedVariance",params.bn3d_branch2b.TrainedVariance)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([1 1],512,"Name","res3d_branch2c","BiasLearnRateFactor",0,"Bias",params.res3d_branch2c.Bias,"Weights",params.res3d_branch2c.Weights)
    batchNormalizationLayer("Name","bn3d_branch2c","Epsilon",0.001,"Offset",params.bn3d_branch2c.Offset,"Scale",params.bn3d_branch2c.Scale,"TrainedMean",params.bn3d_branch2c.TrainedMean,"TrainedVariance",params.bn3d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_7")
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res4a_branch2a.Bias,"Weights",params.res4a_branch2a.Weights)
    batchNormalizationLayer("Name","bn4a_branch2a","Epsilon",0.001,"Offset",params.bn4a_branch2a.Offset,"Scale",params.bn4a_branch2a.Scale,"TrainedMean",params.bn4a_branch2a.TrainedMean,"TrainedVariance",params.bn4a_branch2a.TrainedVariance)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4a_branch2b.Bias,"Weights",params.res4a_branch2b.Weights)
    batchNormalizationLayer("Name","bn4a_branch2b","Epsilon",0.001,"Offset",params.bn4a_branch2b.Offset,"Scale",params.bn4a_branch2b.Scale,"TrainedMean",params.bn4a_branch2b.TrainedMean,"TrainedVariance",params.bn4a_branch2b.TrainedVariance)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([1 1],1024,"Name","res4a_branch2c","BiasLearnRateFactor",0,"Bias",params.res4a_branch2c.Bias,"Weights",params.res4a_branch2c.Weights)
    batchNormalizationLayer("Name","bn4a_branch2c","Epsilon",0.001,"Offset",params.bn4a_branch2c.Offset,"Scale",params.bn4a_branch2c.Scale,"TrainedMean",params.bn4a_branch2c.TrainedMean,"TrainedVariance",params.bn4a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res4a_branch1.Bias,"Weights",params.res4a_branch1.Weights)
    batchNormalizationLayer("Name","bn4a_branch1","Epsilon",0.001,"Offset",params.bn4a_branch1.Offset,"Scale",params.bn4a_branch1.Scale,"TrainedMean",params.bn4a_branch1.TrainedMean,"TrainedVariance",params.bn4a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_8")
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Bias",params.res4b_branch2a.Bias,"Weights",params.res4b_branch2a.Weights)
    batchNormalizationLayer("Name","bn4b_branch2a","Epsilon",0.001,"Offset",params.bn4b_branch2a.Offset,"Scale",params.bn4b_branch2a.Scale,"TrainedMean",params.bn4b_branch2a.TrainedMean,"TrainedVariance",params.bn4b_branch2a.TrainedVariance)
    reluLayer("Name","activation_26_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4b_branch2b.Bias,"Weights",params.res4b_branch2b.Weights)
    batchNormalizationLayer("Name","bn4b_branch2b","Epsilon",0.001,"Offset",params.bn4b_branch2b.Offset,"Scale",params.bn4b_branch2b.Scale,"TrainedMean",params.bn4b_branch2b.TrainedMean,"TrainedVariance",params.bn4b_branch2b.TrainedVariance)
    reluLayer("Name","activation_27_relu")
    convolution2dLayer([1 1],1024,"Name","res4b_branch2c","BiasLearnRateFactor",0,"Bias",params.res4b_branch2c.Bias,"Weights",params.res4b_branch2c.Weights)
    batchNormalizationLayer("Name","bn4b_branch2c","Epsilon",0.001,"Offset",params.bn4b_branch2c.Offset,"Scale",params.bn4b_branch2c.Scale,"TrainedMean",params.bn4b_branch2c.TrainedMean,"TrainedVariance",params.bn4b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_9")
    reluLayer("Name","activation_28_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4c_branch2a","BiasLearnRateFactor",0,"Bias",params.res4c_branch2a.Bias,"Weights",params.res4c_branch2a.Weights)
    batchNormalizationLayer("Name","bn4c_branch2a","Epsilon",0.001,"Offset",params.bn4c_branch2a.Offset,"Scale",params.bn4c_branch2a.Scale,"TrainedMean",params.bn4c_branch2a.TrainedMean,"TrainedVariance",params.bn4c_branch2a.TrainedVariance)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],256,"Name","res4c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4c_branch2b.Bias,"Weights",params.res4c_branch2b.Weights)
    batchNormalizationLayer("Name","bn4c_branch2b","Epsilon",0.001,"Offset",params.bn4c_branch2b.Offset,"Scale",params.bn4c_branch2b.Scale,"TrainedMean",params.bn4c_branch2b.TrainedMean,"TrainedVariance",params.bn4c_branch2b.TrainedVariance)
    reluLayer("Name","activation_30_relu")
    convolution2dLayer([1 1],1024,"Name","res4c_branch2c","BiasLearnRateFactor",0,"Bias",params.res4c_branch2c.Bias,"Weights",params.res4c_branch2c.Weights)
    batchNormalizationLayer("Name","bn4c_branch2c","Epsilon",0.001,"Offset",params.bn4c_branch2c.Offset,"Scale",params.bn4c_branch2c.Scale,"TrainedMean",params.bn4c_branch2c.TrainedMean,"TrainedVariance",params.bn4c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_10")
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4d_branch2a","BiasLearnRateFactor",0,"Bias",params.res4d_branch2a.Bias,"Weights",params.res4d_branch2a.Weights)
    batchNormalizationLayer("Name","bn4d_branch2a","Epsilon",0.001,"Offset",params.bn4d_branch2a.Offset,"Scale",params.bn4d_branch2a.Scale,"TrainedMean",params.bn4d_branch2a.TrainedMean,"TrainedVariance",params.bn4d_branch2a.TrainedVariance)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([3 3],256,"Name","res4d_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4d_branch2b.Bias,"Weights",params.res4d_branch2b.Weights)
    batchNormalizationLayer("Name","bn4d_branch2b","Epsilon",0.001,"Offset",params.bn4d_branch2b.Offset,"Scale",params.bn4d_branch2b.Scale,"TrainedMean",params.bn4d_branch2b.TrainedMean,"TrainedVariance",params.bn4d_branch2b.TrainedVariance)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([1 1],1024,"Name","res4d_branch2c","BiasLearnRateFactor",0,"Bias",params.res4d_branch2c.Bias,"Weights",params.res4d_branch2c.Weights)
    batchNormalizationLayer("Name","bn4d_branch2c","Epsilon",0.001,"Offset",params.bn4d_branch2c.Offset,"Scale",params.bn4d_branch2c.Scale,"TrainedMean",params.bn4d_branch2c.TrainedMean,"TrainedVariance",params.bn4d_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_11")
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4e_branch2a","BiasLearnRateFactor",0,"Bias",params.res4e_branch2a.Bias,"Weights",params.res4e_branch2a.Weights)
    batchNormalizationLayer("Name","bn4e_branch2a","Epsilon",0.001,"Offset",params.bn4e_branch2a.Offset,"Scale",params.bn4e_branch2a.Scale,"TrainedMean",params.bn4e_branch2a.TrainedMean,"TrainedVariance",params.bn4e_branch2a.TrainedVariance)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([3 3],256,"Name","res4e_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4e_branch2b.Bias,"Weights",params.res4e_branch2b.Weights)
    batchNormalizationLayer("Name","bn4e_branch2b","Epsilon",0.001,"Offset",params.bn4e_branch2b.Offset,"Scale",params.bn4e_branch2b.Scale,"TrainedMean",params.bn4e_branch2b.TrainedMean,"TrainedVariance",params.bn4e_branch2b.TrainedVariance)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 1],1024,"Name","res4e_branch2c","BiasLearnRateFactor",0,"Bias",params.res4e_branch2c.Bias,"Weights",params.res4e_branch2c.Weights)
    batchNormalizationLayer("Name","bn4e_branch2c","Epsilon",0.001,"Offset",params.bn4e_branch2c.Offset,"Scale",params.bn4e_branch2c.Scale,"TrainedMean",params.bn4e_branch2c.TrainedMean,"TrainedVariance",params.bn4e_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    reluLayer("Name","activation_37_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4f_branch2a","BiasLearnRateFactor",0,"Bias",params.res4f_branch2a.Bias,"Weights",params.res4f_branch2a.Weights)
    batchNormalizationLayer("Name","bn4f_branch2a","Epsilon",0.001,"Offset",params.bn4f_branch2a.Offset,"Scale",params.bn4f_branch2a.Scale,"TrainedMean",params.bn4f_branch2a.TrainedMean,"TrainedVariance",params.bn4f_branch2a.TrainedVariance)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([3 3],256,"Name","res4f_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res4f_branch2b.Bias,"Weights",params.res4f_branch2b.Weights)
    batchNormalizationLayer("Name","bn4f_branch2b","Epsilon",0.001,"Offset",params.bn4f_branch2b.Offset,"Scale",params.bn4f_branch2b.Scale,"TrainedMean",params.bn4f_branch2b.TrainedMean,"TrainedVariance",params.bn4f_branch2b.TrainedVariance)
    reluLayer("Name","activation_39_relu")
    convolution2dLayer([1 1],1024,"Name","res4f_branch2c","BiasLearnRateFactor",0,"Bias",params.res4f_branch2c.Bias,"Weights",params.res4f_branch2c.Weights)
    batchNormalizationLayer("Name","bn4f_branch2c","Epsilon",0.001,"Offset",params.bn4f_branch2c.Offset,"Scale",params.bn4f_branch2c.Scale,"TrainedMean",params.bn4f_branch2c.TrainedMean,"TrainedVariance",params.bn4f_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_13")
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],2048,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res5a_branch1.Bias,"Weights",params.res5a_branch1.Weights)
    batchNormalizationLayer("Name","bn5a_branch1","Epsilon",0.001,"Offset",params.bn5a_branch1.Offset,"Scale",params.bn5a_branch1.Scale,"TrainedMean",params.bn5a_branch1.TrainedMean,"TrainedVariance",params.bn5a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res5a_branch2a.Bias,"Weights",params.res5a_branch2a.Weights)
    batchNormalizationLayer("Name","bn5a_branch2a","Epsilon",0.001,"Offset",params.bn5a_branch2a.Offset,"Scale",params.bn5a_branch2a.Scale,"TrainedMean",params.bn5a_branch2a.TrainedMean,"TrainedVariance",params.bn5a_branch2a.TrainedVariance)
    reluLayer("Name","activation_41_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res5a_branch2b.Bias,"Weights",params.res5a_branch2b.Weights)
    batchNormalizationLayer("Name","bn5a_branch2b","Epsilon",0.001,"Offset",params.bn5a_branch2b.Offset,"Scale",params.bn5a_branch2b.Scale,"TrainedMean",params.bn5a_branch2b.TrainedMean,"TrainedVariance",params.bn5a_branch2b.TrainedVariance)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 1],2048,"Name","res5a_branch2c","BiasLearnRateFactor",0,"Bias",params.res5a_branch2c.Bias,"Weights",params.res5a_branch2c.Weights)
    batchNormalizationLayer("Name","bn5a_branch2c","Epsilon",0.001,"Offset",params.bn5a_branch2c.Offset,"Scale",params.bn5a_branch2c.Scale,"TrainedMean",params.bn5a_branch2c.TrainedMean,"TrainedVariance",params.bn5a_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"Bias",params.res5b_branch2a.Bias,"Weights",params.res5b_branch2a.Weights)
    batchNormalizationLayer("Name","bn5b_branch2a","Epsilon",0.001,"Offset",params.bn5b_branch2a.Offset,"Scale",params.bn5b_branch2a.Scale,"TrainedMean",params.bn5b_branch2a.TrainedMean,"TrainedVariance",params.bn5b_branch2a.TrainedVariance)
    reluLayer("Name","activation_44_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res5b_branch2b.Bias,"Weights",params.res5b_branch2b.Weights)
    batchNormalizationLayer("Name","bn5b_branch2b","Epsilon",0.001,"Offset",params.bn5b_branch2b.Offset,"Scale",params.bn5b_branch2b.Scale,"TrainedMean",params.bn5b_branch2b.TrainedMean,"TrainedVariance",params.bn5b_branch2b.TrainedVariance)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([1 1],2048,"Name","res5b_branch2c","BiasLearnRateFactor",0,"Bias",params.res5b_branch2c.Bias,"Weights",params.res5b_branch2c.Weights)
    batchNormalizationLayer("Name","bn5b_branch2c","Epsilon",0.001,"Offset",params.bn5b_branch2c.Offset,"Scale",params.bn5b_branch2c.Scale,"TrainedMean",params.bn5b_branch2c.TrainedMean,"TrainedVariance",params.bn5b_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_15")
    reluLayer("Name","activation_46_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5c_branch2a","BiasLearnRateFactor",0,"Bias",params.res5c_branch2a.Bias,"Weights",params.res5c_branch2a.Weights)
    batchNormalizationLayer("Name","bn5c_branch2a","Epsilon",0.001,"Offset",params.bn5c_branch2a.Offset,"Scale",params.bn5c_branch2a.Scale,"TrainedMean",params.bn5c_branch2a.TrainedMean,"TrainedVariance",params.bn5c_branch2a.TrainedVariance)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([3 3],512,"Name","res5c_branch2b","BiasLearnRateFactor",0,"Padding","same","Bias",params.res5c_branch2b.Bias,"Weights",params.res5c_branch2b.Weights)
    batchNormalizationLayer("Name","bn5c_branch2b","Epsilon",0.001,"Offset",params.bn5c_branch2b.Offset,"Scale",params.bn5c_branch2b.Scale,"TrainedMean",params.bn5c_branch2b.TrainedMean,"TrainedVariance",params.bn5c_branch2b.TrainedVariance)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 1],2048,"Name","res5c_branch2c","BiasLearnRateFactor",0,"Bias",params.res5c_branch2c.Bias,"Weights",params.res5c_branch2c.Weights)
    batchNormalizationLayer("Name","bn5c_branch2c","Epsilon",0.001,"Offset",params.bn5c_branch2c.Offset,"Scale",params.bn5c_branch2c.Scale,"TrainedMean",params.bn5c_branch2c.TrainedMean,"TrainedVariance",params.bn5c_branch2c.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_16")
    reluLayer("Name","activation_49_relu")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(2,"Name","fc1000","BiasLearnRateFactor",0)
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch1");
lgraph = connectLayers(lgraph,"max_pooling2d_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"bn2a_branch1","add_1/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2c","add_1/in1");
lgraph = connectLayers(lgraph,"activation_4_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"activation_4_relu","add_2/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2c","add_2/in1");
lgraph = connectLayers(lgraph,"activation_7_relu","res2c_branch2a");
lgraph = connectLayers(lgraph,"activation_7_relu","add_3/in2");
lgraph = connectLayers(lgraph,"bn2c_branch2c","add_3/in1");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"activation_10_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"bn3a_branch1","add_4/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2c","add_4/in1");
lgraph = connectLayers(lgraph,"activation_13_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"activation_13_relu","add_5/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2c","add_5/in1");
lgraph = connectLayers(lgraph,"activation_16_relu","res3c_branch2a");
lgraph = connectLayers(lgraph,"activation_16_relu","add_6/in2");
lgraph = connectLayers(lgraph,"bn3c_branch2c","add_6/in1");
lgraph = connectLayers(lgraph,"activation_19_relu","res3d_branch2a");
lgraph = connectLayers(lgraph,"activation_19_relu","add_7/in2");
lgraph = connectLayers(lgraph,"bn3d_branch2c","add_7/in1");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"activation_22_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch2c","add_8/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","add_8/in2");
lgraph = connectLayers(lgraph,"activation_25_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"activation_25_relu","add_9/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2c","add_9/in1");
lgraph = connectLayers(lgraph,"activation_28_relu","res4c_branch2a");
lgraph = connectLayers(lgraph,"activation_28_relu","add_10/in2");
lgraph = connectLayers(lgraph,"bn4c_branch2c","add_10/in1");
lgraph = connectLayers(lgraph,"activation_31_relu","res4d_branch2a");
lgraph = connectLayers(lgraph,"activation_31_relu","add_11/in2");
lgraph = connectLayers(lgraph,"bn4d_branch2c","add_11/in1");
lgraph = connectLayers(lgraph,"activation_34_relu","res4e_branch2a");
lgraph = connectLayers(lgraph,"activation_34_relu","add_12/in2");
lgraph = connectLayers(lgraph,"bn4e_branch2c","add_12/in1");
lgraph = connectLayers(lgraph,"activation_37_relu","res4f_branch2a");
lgraph = connectLayers(lgraph,"activation_37_relu","add_13/in2");
lgraph = connectLayers(lgraph,"bn4f_branch2c","add_13/in1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"activation_40_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"bn5a_branch2c","add_14/in1");
lgraph = connectLayers(lgraph,"bn5a_branch1","add_14/in2");
lgraph = connectLayers(lgraph,"activation_43_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"activation_43_relu","add_15/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2c","add_15/in1");
lgraph = connectLayers(lgraph,"activation_46_relu","res5c_branch2a");
lgraph = connectLayers(lgraph,"activation_46_relu","add_16/in2");
lgraph = connectLayers(lgraph,"bn5c_branch2c","add_16/in1");

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 1e-4, ...  % 調整初始學習率
    'MaxEpochs', 50, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imgTest, ...
    'ValidationFrequency', 5, ...
    'ExecutionEnvironment', 'gpu', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.01, ...  % 調整學習率下降因子
    'LearnRateDropPeriod',5, ...    % 調整學習率下降周期
    'ValidationPatience', 5, ...
    'L2Regularization', 0.01);  % 調整正則化項

net = trainNetwork(imgTrain,lgraph,options);

%保存訓練好參數
save('C:\Users\joe\Desktop\CNN\Restnettestmini.mat','net');

%測試模型精度
YPred = classify(net,imgTest);
YTest = imgTest.Labels;
Accuracy = sum(YPred == YTest)/numel(YTest)

%計時器
toc

yScores = predict(net, imgTest);
yTrue = grp2idx(YTest);

[~, yPred] = max(yScores, [], 2);

confMat = confusionmat(yTrue, yPred);

TP = confMat(2,2);
FP = confMat(1,2);
FN = confMat(2,1);

recallMetric = TP / (TP + FN);
precisionMetric = TP / (TP + FP);

fprintf('Recall: %.2f\n', recallMetric);
fprintf('Precision: %.2f\n', precisionMetric);

[prec, rec, ~] = perfcurve(yTrue, yScores(:,2), 1, 'xCrit', 'reca', 'yCrit', 'prec');
figure;
plot(rec, prec);
xlabel('Recall');
ylabel('Precision');
title('Precision-Recall Curve');
grid on;


