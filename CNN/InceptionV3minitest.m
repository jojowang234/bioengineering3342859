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
params = load("C:\Users\joe\Desktop\專題\params_2024_01_29__22_41_56.mat");

lgraph = layerGraph();
tempLayers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([3 3],32,"Name","conv2d_1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.conv2d_1.Bias,"Weights",params.conv2d_1.Weights)
    batchNormalizationLayer("Name","batch_normalization_1","Epsilon",0.001,"Offset",params.batch_normalization_1.Offset,"Scale",params.batch_normalization_1.Scale,"TrainedMean",params.batch_normalization_1.TrainedMean,"TrainedVariance",params.batch_normalization_1.TrainedVariance)
    reluLayer("Name","activation_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2d_2","BiasLearnRateFactor",0,"Bias",params.conv2d_2.Bias,"Weights",params.conv2d_2.Weights)
    batchNormalizationLayer("Name","batch_normalization_2","Epsilon",0.001,"Offset",params.batch_normalization_2.Offset,"Scale",params.batch_normalization_2.Scale,"TrainedMean",params.batch_normalization_2.TrainedMean,"TrainedVariance",params.batch_normalization_2.TrainedVariance)
    reluLayer("Name","activation_2_relu")
    convolution2dLayer([3 3],64,"Name","conv2d_3","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_3.Bias,"Weights",params.conv2d_3.Weights)
    batchNormalizationLayer("Name","batch_normalization_3","Epsilon",0.001,"Offset",params.batch_normalization_3.Offset,"Scale",params.batch_normalization_3.Scale,"TrainedMean",params.batch_normalization_3.TrainedMean,"TrainedVariance",params.batch_normalization_3.TrainedVariance)
    reluLayer("Name","activation_3_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_1","Stride",[2 2])
    convolution2dLayer([1 1],80,"Name","conv2d_4","BiasLearnRateFactor",0,"Bias",params.conv2d_4.Bias,"Weights",params.conv2d_4.Weights)
    batchNormalizationLayer("Name","batch_normalization_4","Epsilon",0.001,"Offset",params.batch_normalization_4.Offset,"Scale",params.batch_normalization_4.Scale,"TrainedMean",params.batch_normalization_4.TrainedMean,"TrainedVariance",params.batch_normalization_4.TrainedVariance)
    reluLayer("Name","activation_4_relu")
    convolution2dLayer([3 3],192,"Name","conv2d_5","BiasLearnRateFactor",0,"Bias",params.conv2d_5.Bias,"Weights",params.conv2d_5.Weights)
    batchNormalizationLayer("Name","batch_normalization_5","Epsilon",0.001,"Offset",params.batch_normalization_5.Offset,"Scale",params.batch_normalization_5.Scale,"TrainedMean",params.batch_normalization_5.TrainedMean,"TrainedVariance",params.batch_normalization_5.TrainedVariance)
    reluLayer("Name","activation_5_relu")
    maxPooling2dLayer([3 3],"Name","max_pooling2d_2","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","conv2d_7","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_7.Bias,"Weights",params.conv2d_7.Weights)
    batchNormalizationLayer("Name","batch_normalization_7","Epsilon",0.001,"Offset",params.batch_normalization_7.Offset,"Scale",params.batch_normalization_7.Scale,"TrainedMean",params.batch_normalization_7.TrainedMean,"TrainedVariance",params.batch_normalization_7.TrainedVariance)
    reluLayer("Name","activation_7_relu")
    convolution2dLayer([5 5],64,"Name","conv2d_8","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_8.Bias,"Weights",params.conv2d_8.Weights)
    batchNormalizationLayer("Name","batch_normalization_8","Epsilon",0.001,"Offset",params.batch_normalization_8.Offset,"Scale",params.batch_normalization_8.Scale,"TrainedMean",params.batch_normalization_8.TrainedMean,"TrainedVariance",params.batch_normalization_8.TrainedVariance)
    reluLayer("Name","activation_8_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_9","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_9.Bias,"Weights",params.conv2d_9.Weights)
    batchNormalizationLayer("Name","batch_normalization_9","Epsilon",0.001,"Offset",params.batch_normalization_9.Offset,"Scale",params.batch_normalization_9.Scale,"TrainedMean",params.batch_normalization_9.TrainedMean,"TrainedVariance",params.batch_normalization_9.TrainedVariance)
    reluLayer("Name","activation_9_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_10","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_10.Bias,"Weights",params.conv2d_10.Weights)
    batchNormalizationLayer("Name","batch_normalization_10","Epsilon",0.001,"Offset",params.batch_normalization_10.Offset,"Scale",params.batch_normalization_10.Scale,"TrainedMean",params.batch_normalization_10.TrainedMean,"TrainedVariance",params.batch_normalization_10.TrainedVariance)
    reluLayer("Name","activation_10_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_11","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_11.Bias,"Weights",params.conv2d_11.Weights)
    batchNormalizationLayer("Name","batch_normalization_11","Epsilon",0.001,"Offset",params.batch_normalization_11.Offset,"Scale",params.batch_normalization_11.Scale,"TrainedMean",params.batch_normalization_11.TrainedMean,"TrainedVariance",params.batch_normalization_11.TrainedVariance)
    reluLayer("Name","activation_11_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_1","Padding","same")
    convolution2dLayer([1 1],32,"Name","conv2d_12","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_12.Bias,"Weights",params.conv2d_12.Weights)
    batchNormalizationLayer("Name","batch_normalization_12","Epsilon",0.001,"Offset",params.batch_normalization_12.Offset,"Scale",params.batch_normalization_12.Scale,"TrainedMean",params.batch_normalization_12.TrainedMean,"TrainedVariance",params.batch_normalization_12.TrainedVariance)
    reluLayer("Name","activation_12_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_6","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_6.Bias,"Weights",params.conv2d_6.Weights)
    batchNormalizationLayer("Name","batch_normalization_6","Epsilon",0.001,"Offset",params.batch_normalization_6.Offset,"Scale",params.batch_normalization_6.Scale,"TrainedMean",params.batch_normalization_6.TrainedMean,"TrainedVariance",params.batch_normalization_6.TrainedVariance)
    reluLayer("Name","activation_6_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed0");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","conv2d_14","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_14.Bias,"Weights",params.conv2d_14.Weights)
    batchNormalizationLayer("Name","batch_normalization_14","Epsilon",0.001,"Offset",params.batch_normalization_14.Offset,"Scale",params.batch_normalization_14.Scale,"TrainedMean",params.batch_normalization_14.TrainedMean,"TrainedVariance",params.batch_normalization_14.TrainedVariance)
    reluLayer("Name","activation_14_relu")
    convolution2dLayer([5 5],64,"Name","conv2d_15","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_15.Bias,"Weights",params.conv2d_15.Weights)
    batchNormalizationLayer("Name","batch_normalization_15","Epsilon",0.001,"Offset",params.batch_normalization_15.Offset,"Scale",params.batch_normalization_15.Scale,"TrainedMean",params.batch_normalization_15.TrainedMean,"TrainedVariance",params.batch_normalization_15.TrainedVariance)
    reluLayer("Name","activation_15_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_16","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_16.Bias,"Weights",params.conv2d_16.Weights)
    batchNormalizationLayer("Name","batch_normalization_16","Epsilon",0.001,"Offset",params.batch_normalization_16.Offset,"Scale",params.batch_normalization_16.Scale,"TrainedMean",params.batch_normalization_16.TrainedMean,"TrainedVariance",params.batch_normalization_16.TrainedVariance)
    reluLayer("Name","activation_16_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_17","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_17.Bias,"Weights",params.conv2d_17.Weights)
    batchNormalizationLayer("Name","batch_normalization_17","Epsilon",0.001,"Offset",params.batch_normalization_17.Offset,"Scale",params.batch_normalization_17.Scale,"TrainedMean",params.batch_normalization_17.TrainedMean,"TrainedVariance",params.batch_normalization_17.TrainedVariance)
    reluLayer("Name","activation_17_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_18","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_18.Bias,"Weights",params.conv2d_18.Weights)
    batchNormalizationLayer("Name","batch_normalization_18","Epsilon",0.001,"Offset",params.batch_normalization_18.Offset,"Scale",params.batch_normalization_18.Scale,"TrainedMean",params.batch_normalization_18.TrainedMean,"TrainedVariance",params.batch_normalization_18.TrainedVariance)
    reluLayer("Name","activation_18_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_2","Padding","same")
    convolution2dLayer([1 1],64,"Name","conv2d_19","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_19.Bias,"Weights",params.conv2d_19.Weights)
    batchNormalizationLayer("Name","batch_normalization_19","Epsilon",0.001,"Offset",params.batch_normalization_19.Offset,"Scale",params.batch_normalization_19.Scale,"TrainedMean",params.batch_normalization_19.TrainedMean,"TrainedVariance",params.batch_normalization_19.TrainedVariance)
    reluLayer("Name","activation_19_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_13","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_13.Bias,"Weights",params.conv2d_13.Weights)
    batchNormalizationLayer("Name","batch_normalization_13","Epsilon",0.001,"Offset",params.batch_normalization_13.Offset,"Scale",params.batch_normalization_13.Scale,"TrainedMean",params.batch_normalization_13.TrainedMean,"TrainedVariance",params.batch_normalization_13.TrainedVariance)
    reluLayer("Name","activation_13_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_23","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_23.Bias,"Weights",params.conv2d_23.Weights)
    batchNormalizationLayer("Name","batch_normalization_23","Epsilon",0.001,"Offset",params.batch_normalization_23.Offset,"Scale",params.batch_normalization_23.Scale,"TrainedMean",params.batch_normalization_23.TrainedMean,"TrainedVariance",params.batch_normalization_23.TrainedVariance)
    reluLayer("Name","activation_23_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_24","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_24.Bias,"Weights",params.conv2d_24.Weights)
    batchNormalizationLayer("Name","batch_normalization_24","Epsilon",0.001,"Offset",params.batch_normalization_24.Offset,"Scale",params.batch_normalization_24.Scale,"TrainedMean",params.batch_normalization_24.TrainedMean,"TrainedVariance",params.batch_normalization_24.TrainedVariance)
    reluLayer("Name","activation_24_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_25","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_25.Bias,"Weights",params.conv2d_25.Weights)
    batchNormalizationLayer("Name","batch_normalization_25","Epsilon",0.001,"Offset",params.batch_normalization_25.Offset,"Scale",params.batch_normalization_25.Scale,"TrainedMean",params.batch_normalization_25.TrainedMean,"TrainedVariance",params.batch_normalization_25.TrainedVariance)
    reluLayer("Name","activation_25_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","conv2d_21","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_21.Bias,"Weights",params.conv2d_21.Weights)
    batchNormalizationLayer("Name","batch_normalization_21","Epsilon",0.001,"Offset",params.batch_normalization_21.Offset,"Scale",params.batch_normalization_21.Scale,"TrainedMean",params.batch_normalization_21.TrainedMean,"TrainedVariance",params.batch_normalization_21.TrainedVariance)
    reluLayer("Name","activation_21_relu")
    convolution2dLayer([5 5],64,"Name","conv2d_22","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_22.Bias,"Weights",params.conv2d_22.Weights)
    batchNormalizationLayer("Name","batch_normalization_22","Epsilon",0.001,"Offset",params.batch_normalization_22.Offset,"Scale",params.batch_normalization_22.Scale,"TrainedMean",params.batch_normalization_22.TrainedMean,"TrainedVariance",params.batch_normalization_22.TrainedVariance)
    reluLayer("Name","activation_22_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_3","Padding","same")
    convolution2dLayer([1 1],64,"Name","conv2d_26","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_26.Bias,"Weights",params.conv2d_26.Weights)
    batchNormalizationLayer("Name","batch_normalization_26","Epsilon",0.001,"Offset",params.batch_normalization_26.Offset,"Scale",params.batch_normalization_26.Scale,"TrainedMean",params.batch_normalization_26.TrainedMean,"TrainedVariance",params.batch_normalization_26.TrainedVariance)
    reluLayer("Name","activation_26_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_20","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_20.Bias,"Weights",params.conv2d_20.Weights)
    batchNormalizationLayer("Name","batch_normalization_20","Epsilon",0.001,"Offset",params.batch_normalization_20.Offset,"Scale",params.batch_normalization_20.Scale,"TrainedMean",params.batch_normalization_20.TrainedMean,"TrainedVariance",params.batch_normalization_20.TrainedVariance)
    reluLayer("Name","activation_20_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([3 3],"Name","max_pooling2d_3","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],384,"Name","conv2d_27","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.conv2d_27.Bias,"Weights",params.conv2d_27.Weights)
    batchNormalizationLayer("Name","batch_normalization_27","Epsilon",0.001,"Offset",params.batch_normalization_27.Offset,"Scale",params.batch_normalization_27.Scale,"TrainedMean",params.batch_normalization_27.TrainedMean,"TrainedVariance",params.batch_normalization_27.TrainedVariance)
    reluLayer("Name","activation_27_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv2d_28","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_28.Bias,"Weights",params.conv2d_28.Weights)
    batchNormalizationLayer("Name","batch_normalization_28","Epsilon",0.001,"Offset",params.batch_normalization_28.Offset,"Scale",params.batch_normalization_28.Scale,"TrainedMean",params.batch_normalization_28.TrainedMean,"TrainedVariance",params.batch_normalization_28.TrainedVariance)
    reluLayer("Name","activation_28_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_29","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_29.Bias,"Weights",params.conv2d_29.Weights)
    batchNormalizationLayer("Name","batch_normalization_29","Epsilon",0.001,"Offset",params.batch_normalization_29.Offset,"Scale",params.batch_normalization_29.Scale,"TrainedMean",params.batch_normalization_29.TrainedMean,"TrainedVariance",params.batch_normalization_29.TrainedVariance)
    reluLayer("Name","activation_29_relu")
    convolution2dLayer([3 3],96,"Name","conv2d_30","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.conv2d_30.Bias,"Weights",params.conv2d_30.Weights)
    batchNormalizationLayer("Name","batch_normalization_30","Epsilon",0.001,"Offset",params.batch_normalization_30.Offset,"Scale",params.batch_normalization_30.Scale,"TrainedMean",params.batch_normalization_30.TrainedMean,"TrainedVariance",params.batch_normalization_30.TrainedVariance)
    reluLayer("Name","activation_30_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(3,"Name","mixed3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_32","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_32.Bias,"Weights",params.conv2d_32.Weights)
    batchNormalizationLayer("Name","batch_normalization_32","Epsilon",0.001,"Offset",params.batch_normalization_32.Offset,"Scale",params.batch_normalization_32.Scale,"TrainedMean",params.batch_normalization_32.TrainedMean,"TrainedVariance",params.batch_normalization_32.TrainedVariance)
    reluLayer("Name","activation_32_relu")
    convolution2dLayer([1 7],128,"Name","conv2d_33","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_33.Bias,"Weights",params.conv2d_33.Weights)
    batchNormalizationLayer("Name","batch_normalization_33","Epsilon",0.001,"Offset",params.batch_normalization_33.Offset,"Scale",params.batch_normalization_33.Scale,"TrainedMean",params.batch_normalization_33.TrainedMean,"TrainedVariance",params.batch_normalization_33.TrainedVariance)
    reluLayer("Name","activation_33_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_34","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_34.Bias,"Weights",params.conv2d_34.Weights)
    batchNormalizationLayer("Name","batch_normalization_34","Epsilon",0.001,"Offset",params.batch_normalization_34.Offset,"Scale",params.batch_normalization_34.Scale,"TrainedMean",params.batch_normalization_34.TrainedMean,"TrainedVariance",params.batch_normalization_34.TrainedVariance)
    reluLayer("Name","activation_34_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_4","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_40","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_40.Bias,"Weights",params.conv2d_40.Weights)
    batchNormalizationLayer("Name","batch_normalization_40","Epsilon",0.001,"Offset",params.batch_normalization_40.Offset,"Scale",params.batch_normalization_40.Scale,"TrainedMean",params.batch_normalization_40.TrainedMean,"TrainedVariance",params.batch_normalization_40.TrainedVariance)
    reluLayer("Name","activation_40_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_35","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_35.Bias,"Weights",params.conv2d_35.Weights)
    batchNormalizationLayer("Name","batch_normalization_35","Epsilon",0.001,"Offset",params.batch_normalization_35.Offset,"Scale",params.batch_normalization_35.Scale,"TrainedMean",params.batch_normalization_35.TrainedMean,"TrainedVariance",params.batch_normalization_35.TrainedVariance)
    reluLayer("Name","activation_35_relu")
    convolution2dLayer([7 1],128,"Name","conv2d_36","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_36.Bias,"Weights",params.conv2d_36.Weights)
    batchNormalizationLayer("Name","batch_normalization_36","Epsilon",0.001,"Offset",params.batch_normalization_36.Offset,"Scale",params.batch_normalization_36.Scale,"TrainedMean",params.batch_normalization_36.TrainedMean,"TrainedVariance",params.batch_normalization_36.TrainedVariance)
    reluLayer("Name","activation_36_relu")
    convolution2dLayer([1 7],128,"Name","conv2d_37","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_37.Bias,"Weights",params.conv2d_37.Weights)
    batchNormalizationLayer("Name","batch_normalization_37","Epsilon",0.001,"Offset",params.batch_normalization_37.Offset,"Scale",params.batch_normalization_37.Scale,"TrainedMean",params.batch_normalization_37.TrainedMean,"TrainedVariance",params.batch_normalization_37.TrainedVariance)
    reluLayer("Name","activation_37_relu")
    convolution2dLayer([7 1],128,"Name","conv2d_38","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_38.Bias,"Weights",params.conv2d_38.Weights)
    batchNormalizationLayer("Name","batch_normalization_38","Epsilon",0.001,"Offset",params.batch_normalization_38.Offset,"Scale",params.batch_normalization_38.Scale,"TrainedMean",params.batch_normalization_38.TrainedMean,"TrainedVariance",params.batch_normalization_38.TrainedVariance)
    reluLayer("Name","activation_38_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_39","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_39.Bias,"Weights",params.conv2d_39.Weights)
    batchNormalizationLayer("Name","batch_normalization_39","Epsilon",0.001,"Offset",params.batch_normalization_39.Offset,"Scale",params.batch_normalization_39.Scale,"TrainedMean",params.batch_normalization_39.TrainedMean,"TrainedVariance",params.batch_normalization_39.TrainedVariance)
    reluLayer("Name","activation_39_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_31","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_31.Bias,"Weights",params.conv2d_31.Weights)
    batchNormalizationLayer("Name","batch_normalization_31","Epsilon",0.001,"Offset",params.batch_normalization_31.Offset,"Scale",params.batch_normalization_31.Scale,"TrainedMean",params.batch_normalization_31.TrainedMean,"TrainedVariance",params.batch_normalization_31.TrainedVariance)
    reluLayer("Name","activation_31_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_5","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_50","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_50.Bias,"Weights",params.conv2d_50.Weights)
    batchNormalizationLayer("Name","batch_normalization_50","Epsilon",0.001,"Offset",params.batch_normalization_50.Offset,"Scale",params.batch_normalization_50.Scale,"TrainedMean",params.batch_normalization_50.TrainedMean,"TrainedVariance",params.batch_normalization_50.TrainedVariance)
    reluLayer("Name","activation_50_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_45","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_45.Bias,"Weights",params.conv2d_45.Weights)
    batchNormalizationLayer("Name","batch_normalization_45","Epsilon",0.001,"Offset",params.batch_normalization_45.Offset,"Scale",params.batch_normalization_45.Scale,"TrainedMean",params.batch_normalization_45.TrainedMean,"TrainedVariance",params.batch_normalization_45.TrainedVariance)
    reluLayer("Name","activation_45_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_46","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_46.Bias,"Weights",params.conv2d_46.Weights)
    batchNormalizationLayer("Name","batch_normalization_46","Epsilon",0.001,"Offset",params.batch_normalization_46.Offset,"Scale",params.batch_normalization_46.Scale,"TrainedMean",params.batch_normalization_46.TrainedMean,"TrainedVariance",params.batch_normalization_46.TrainedVariance)
    reluLayer("Name","activation_46_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_47","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_47.Bias,"Weights",params.conv2d_47.Weights)
    batchNormalizationLayer("Name","batch_normalization_47","Epsilon",0.001,"Offset",params.batch_normalization_47.Offset,"Scale",params.batch_normalization_47.Scale,"TrainedMean",params.batch_normalization_47.TrainedMean,"TrainedVariance",params.batch_normalization_47.TrainedVariance)
    reluLayer("Name","activation_47_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_48","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_48.Bias,"Weights",params.conv2d_48.Weights)
    batchNormalizationLayer("Name","batch_normalization_48","Epsilon",0.001,"Offset",params.batch_normalization_48.Offset,"Scale",params.batch_normalization_48.Scale,"TrainedMean",params.batch_normalization_48.TrainedMean,"TrainedVariance",params.batch_normalization_48.TrainedVariance)
    reluLayer("Name","activation_48_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_49","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_49.Bias,"Weights",params.conv2d_49.Weights)
    batchNormalizationLayer("Name","batch_normalization_49","Epsilon",0.001,"Offset",params.batch_normalization_49.Offset,"Scale",params.batch_normalization_49.Scale,"TrainedMean",params.batch_normalization_49.TrainedMean,"TrainedVariance",params.batch_normalization_49.TrainedVariance)
    reluLayer("Name","activation_49_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_42","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_42.Bias,"Weights",params.conv2d_42.Weights)
    batchNormalizationLayer("Name","batch_normalization_42","Epsilon",0.001,"Offset",params.batch_normalization_42.Offset,"Scale",params.batch_normalization_42.Scale,"TrainedMean",params.batch_normalization_42.TrainedMean,"TrainedVariance",params.batch_normalization_42.TrainedVariance)
    reluLayer("Name","activation_42_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_43","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_43.Bias,"Weights",params.conv2d_43.Weights)
    batchNormalizationLayer("Name","batch_normalization_43","Epsilon",0.001,"Offset",params.batch_normalization_43.Offset,"Scale",params.batch_normalization_43.Scale,"TrainedMean",params.batch_normalization_43.TrainedMean,"TrainedVariance",params.batch_normalization_43.TrainedVariance)
    reluLayer("Name","activation_43_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_44","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_44.Bias,"Weights",params.conv2d_44.Weights)
    batchNormalizationLayer("Name","batch_normalization_44","Epsilon",0.001,"Offset",params.batch_normalization_44.Offset,"Scale",params.batch_normalization_44.Scale,"TrainedMean",params.batch_normalization_44.TrainedMean,"TrainedVariance",params.batch_normalization_44.TrainedVariance)
    reluLayer("Name","activation_44_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_41","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_41.Bias,"Weights",params.conv2d_41.Weights)
    batchNormalizationLayer("Name","batch_normalization_41","Epsilon",0.001,"Offset",params.batch_normalization_41.Offset,"Scale",params.batch_normalization_41.Scale,"TrainedMean",params.batch_normalization_41.TrainedMean,"TrainedVariance",params.batch_normalization_41.TrainedVariance)
    reluLayer("Name","activation_41_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_6","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_60","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_60.Bias,"Weights",params.conv2d_60.Weights)
    batchNormalizationLayer("Name","batch_normalization_60","Epsilon",0.001,"Offset",params.batch_normalization_60.Offset,"Scale",params.batch_normalization_60.Scale,"TrainedMean",params.batch_normalization_60.TrainedMean,"TrainedVariance",params.batch_normalization_60.TrainedVariance)
    reluLayer("Name","activation_60_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_51","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_51.Bias,"Weights",params.conv2d_51.Weights)
    batchNormalizationLayer("Name","batch_normalization_51","Epsilon",0.001,"Offset",params.batch_normalization_51.Offset,"Scale",params.batch_normalization_51.Scale,"TrainedMean",params.batch_normalization_51.TrainedMean,"TrainedVariance",params.batch_normalization_51.TrainedVariance)
    reluLayer("Name","activation_51_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_55","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_55.Bias,"Weights",params.conv2d_55.Weights)
    batchNormalizationLayer("Name","batch_normalization_55","Epsilon",0.001,"Offset",params.batch_normalization_55.Offset,"Scale",params.batch_normalization_55.Scale,"TrainedMean",params.batch_normalization_55.TrainedMean,"TrainedVariance",params.batch_normalization_55.TrainedVariance)
    reluLayer("Name","activation_55_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_56","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_56.Bias,"Weights",params.conv2d_56.Weights)
    batchNormalizationLayer("Name","batch_normalization_56","Epsilon",0.001,"Offset",params.batch_normalization_56.Offset,"Scale",params.batch_normalization_56.Scale,"TrainedMean",params.batch_normalization_56.TrainedMean,"TrainedVariance",params.batch_normalization_56.TrainedVariance)
    reluLayer("Name","activation_56_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_57","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_57.Bias,"Weights",params.conv2d_57.Weights)
    batchNormalizationLayer("Name","batch_normalization_57","Epsilon",0.001,"Offset",params.batch_normalization_57.Offset,"Scale",params.batch_normalization_57.Scale,"TrainedMean",params.batch_normalization_57.TrainedMean,"TrainedVariance",params.batch_normalization_57.TrainedVariance)
    reluLayer("Name","activation_57_relu")
    convolution2dLayer([7 1],160,"Name","conv2d_58","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_58.Bias,"Weights",params.conv2d_58.Weights)
    batchNormalizationLayer("Name","batch_normalization_58","Epsilon",0.001,"Offset",params.batch_normalization_58.Offset,"Scale",params.batch_normalization_58.Scale,"TrainedMean",params.batch_normalization_58.TrainedMean,"TrainedVariance",params.batch_normalization_58.TrainedVariance)
    reluLayer("Name","activation_58_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_59","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_59.Bias,"Weights",params.conv2d_59.Weights)
    batchNormalizationLayer("Name","batch_normalization_59","Epsilon",0.001,"Offset",params.batch_normalization_59.Offset,"Scale",params.batch_normalization_59.Scale,"TrainedMean",params.batch_normalization_59.TrainedMean,"TrainedVariance",params.batch_normalization_59.TrainedVariance)
    reluLayer("Name","activation_59_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],160,"Name","conv2d_52","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_52.Bias,"Weights",params.conv2d_52.Weights)
    batchNormalizationLayer("Name","batch_normalization_52","Epsilon",0.001,"Offset",params.batch_normalization_52.Offset,"Scale",params.batch_normalization_52.Scale,"TrainedMean",params.batch_normalization_52.TrainedMean,"TrainedVariance",params.batch_normalization_52.TrainedVariance)
    reluLayer("Name","activation_52_relu")
    convolution2dLayer([1 7],160,"Name","conv2d_53","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_53.Bias,"Weights",params.conv2d_53.Weights)
    batchNormalizationLayer("Name","batch_normalization_53","Epsilon",0.001,"Offset",params.batch_normalization_53.Offset,"Scale",params.batch_normalization_53.Scale,"TrainedMean",params.batch_normalization_53.TrainedMean,"TrainedVariance",params.batch_normalization_53.TrainedVariance)
    reluLayer("Name","activation_53_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_54","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_54.Bias,"Weights",params.conv2d_54.Weights)
    batchNormalizationLayer("Name","batch_normalization_54","Epsilon",0.001,"Offset",params.batch_normalization_54.Offset,"Scale",params.batch_normalization_54.Scale,"TrainedMean",params.batch_normalization_54.TrainedMean,"TrainedVariance",params.batch_normalization_54.TrainedVariance)
    reluLayer("Name","activation_54_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_7","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_70","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_70.Bias,"Weights",params.conv2d_70.Weights)
    batchNormalizationLayer("Name","batch_normalization_70","Epsilon",0.001,"Offset",params.batch_normalization_70.Offset,"Scale",params.batch_normalization_70.Scale,"TrainedMean",params.batch_normalization_70.TrainedMean,"TrainedVariance",params.batch_normalization_70.TrainedVariance)
    reluLayer("Name","activation_70_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_65","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_65.Bias,"Weights",params.conv2d_65.Weights)
    batchNormalizationLayer("Name","batch_normalization_65","Epsilon",0.001,"Offset",params.batch_normalization_65.Offset,"Scale",params.batch_normalization_65.Scale,"TrainedMean",params.batch_normalization_65.TrainedMean,"TrainedVariance",params.batch_normalization_65.TrainedVariance)
    reluLayer("Name","activation_65_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_66","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_66.Bias,"Weights",params.conv2d_66.Weights)
    batchNormalizationLayer("Name","batch_normalization_66","Epsilon",0.001,"Offset",params.batch_normalization_66.Offset,"Scale",params.batch_normalization_66.Scale,"TrainedMean",params.batch_normalization_66.TrainedMean,"TrainedVariance",params.batch_normalization_66.TrainedVariance)
    reluLayer("Name","activation_66_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_67","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_67.Bias,"Weights",params.conv2d_67.Weights)
    batchNormalizationLayer("Name","batch_normalization_67","Epsilon",0.001,"Offset",params.batch_normalization_67.Offset,"Scale",params.batch_normalization_67.Scale,"TrainedMean",params.batch_normalization_67.TrainedMean,"TrainedVariance",params.batch_normalization_67.TrainedVariance)
    reluLayer("Name","activation_67_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_68","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_68.Bias,"Weights",params.conv2d_68.Weights)
    batchNormalizationLayer("Name","batch_normalization_68","Epsilon",0.001,"Offset",params.batch_normalization_68.Offset,"Scale",params.batch_normalization_68.Scale,"TrainedMean",params.batch_normalization_68.TrainedMean,"TrainedVariance",params.batch_normalization_68.TrainedVariance)
    reluLayer("Name","activation_68_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_69","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_69.Bias,"Weights",params.conv2d_69.Weights)
    batchNormalizationLayer("Name","batch_normalization_69","Epsilon",0.001,"Offset",params.batch_normalization_69.Offset,"Scale",params.batch_normalization_69.Scale,"TrainedMean",params.batch_normalization_69.TrainedMean,"TrainedVariance",params.batch_normalization_69.TrainedVariance)
    reluLayer("Name","activation_69_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_61","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_61.Bias,"Weights",params.conv2d_61.Weights)
    batchNormalizationLayer("Name","batch_normalization_61","Epsilon",0.001,"Offset",params.batch_normalization_61.Offset,"Scale",params.batch_normalization_61.Scale,"TrainedMean",params.batch_normalization_61.TrainedMean,"TrainedVariance",params.batch_normalization_61.TrainedVariance)
    reluLayer("Name","activation_61_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_62","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_62.Bias,"Weights",params.conv2d_62.Weights)
    batchNormalizationLayer("Name","batch_normalization_62","Epsilon",0.001,"Offset",params.batch_normalization_62.Offset,"Scale",params.batch_normalization_62.Scale,"TrainedMean",params.batch_normalization_62.TrainedMean,"TrainedVariance",params.batch_normalization_62.TrainedVariance)
    reluLayer("Name","activation_62_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_63","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_63.Bias,"Weights",params.conv2d_63.Weights)
    batchNormalizationLayer("Name","batch_normalization_63","Epsilon",0.001,"Offset",params.batch_normalization_63.Offset,"Scale",params.batch_normalization_63.Scale,"TrainedMean",params.batch_normalization_63.TrainedMean,"TrainedVariance",params.batch_normalization_63.TrainedVariance)
    reluLayer("Name","activation_63_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_64","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_64.Bias,"Weights",params.conv2d_64.Weights)
    batchNormalizationLayer("Name","batch_normalization_64","Epsilon",0.001,"Offset",params.batch_normalization_64.Offset,"Scale",params.batch_normalization_64.Scale,"TrainedMean",params.batch_normalization_64.TrainedMean,"TrainedVariance",params.batch_normalization_64.TrainedVariance)
    reluLayer("Name","activation_64_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([3 3],"Name","max_pooling2d_4","Stride",[2 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_71","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_71.Bias,"Weights",params.conv2d_71.Weights)
    batchNormalizationLayer("Name","batch_normalization_71","Epsilon",0.001,"Offset",params.batch_normalization_71.Offset,"Scale",params.batch_normalization_71.Scale,"TrainedMean",params.batch_normalization_71.TrainedMean,"TrainedVariance",params.batch_normalization_71.TrainedVariance)
    reluLayer("Name","activation_71_relu")
    convolution2dLayer([3 3],320,"Name","conv2d_72","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.conv2d_72.Bias,"Weights",params.conv2d_72.Weights)
    batchNormalizationLayer("Name","batch_normalization_72","Epsilon",0.001,"Offset",params.batch_normalization_72.Offset,"Scale",params.batch_normalization_72.Scale,"TrainedMean",params.batch_normalization_72.TrainedMean,"TrainedVariance",params.batch_normalization_72.TrainedVariance)
    reluLayer("Name","activation_72_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","conv2d_73","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_73.Bias,"Weights",params.conv2d_73.Weights)
    batchNormalizationLayer("Name","batch_normalization_73","Epsilon",0.001,"Offset",params.batch_normalization_73.Offset,"Scale",params.batch_normalization_73.Scale,"TrainedMean",params.batch_normalization_73.TrainedMean,"TrainedVariance",params.batch_normalization_73.TrainedVariance)
    reluLayer("Name","activation_73_relu")
    convolution2dLayer([1 7],192,"Name","conv2d_74","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_74.Bias,"Weights",params.conv2d_74.Weights)
    batchNormalizationLayer("Name","batch_normalization_74","Epsilon",0.001,"Offset",params.batch_normalization_74.Offset,"Scale",params.batch_normalization_74.Scale,"TrainedMean",params.batch_normalization_74.TrainedMean,"TrainedVariance",params.batch_normalization_74.TrainedVariance)
    reluLayer("Name","activation_74_relu")
    convolution2dLayer([7 1],192,"Name","conv2d_75","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_75.Bias,"Weights",params.conv2d_75.Weights)
    batchNormalizationLayer("Name","batch_normalization_75","Epsilon",0.001,"Offset",params.batch_normalization_75.Offset,"Scale",params.batch_normalization_75.Scale,"TrainedMean",params.batch_normalization_75.TrainedMean,"TrainedVariance",params.batch_normalization_75.TrainedVariance)
    reluLayer("Name","activation_75_relu")
    convolution2dLayer([3 3],192,"Name","conv2d_76","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.conv2d_76.Bias,"Weights",params.conv2d_76.Weights)
    batchNormalizationLayer("Name","batch_normalization_76","Epsilon",0.001,"Offset",params.batch_normalization_76.Offset,"Scale",params.batch_normalization_76.Scale,"TrainedMean",params.batch_normalization_76.TrainedMean,"TrainedVariance",params.batch_normalization_76.TrainedVariance)
    reluLayer("Name","activation_76_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(3,"Name","mixed8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],320,"Name","conv2d_77","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_77.Bias,"Weights",params.conv2d_77.Weights)
    batchNormalizationLayer("Name","batch_normalization_77","Epsilon",0.001,"Offset",params.batch_normalization_77.Offset,"Scale",params.batch_normalization_77.Scale,"TrainedMean",params.batch_normalization_77.TrainedMean,"TrainedVariance",params.batch_normalization_77.TrainedVariance)
    reluLayer("Name","activation_77_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],448,"Name","conv2d_81","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_81.Bias,"Weights",params.conv2d_81.Weights)
    batchNormalizationLayer("Name","batch_normalization_81","Epsilon",0.001,"Offset",params.batch_normalization_81.Offset,"Scale",params.batch_normalization_81.Scale,"TrainedMean",params.batch_normalization_81.TrainedMean,"TrainedVariance",params.batch_normalization_81.TrainedVariance)
    reluLayer("Name","activation_81_relu")
    convolution2dLayer([3 3],384,"Name","conv2d_82","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_82.Bias,"Weights",params.conv2d_82.Weights)
    batchNormalizationLayer("Name","batch_normalization_82","Epsilon",0.001,"Offset",params.batch_normalization_82.Offset,"Scale",params.batch_normalization_82.Scale,"TrainedMean",params.batch_normalization_82.TrainedMean,"TrainedVariance",params.batch_normalization_82.TrainedVariance)
    reluLayer("Name","activation_82_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","conv2d_78","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_78.Bias,"Weights",params.conv2d_78.Weights)
    batchNormalizationLayer("Name","batch_normalization_78","Epsilon",0.001,"Offset",params.batch_normalization_78.Offset,"Scale",params.batch_normalization_78.Scale,"TrainedMean",params.batch_normalization_78.TrainedMean,"TrainedVariance",params.batch_normalization_78.TrainedVariance)
    reluLayer("Name","activation_78_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_8","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_85","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_85.Bias,"Weights",params.conv2d_85.Weights)
    batchNormalizationLayer("Name","batch_normalization_85","Epsilon",0.001,"Offset",params.batch_normalization_85.Offset,"Scale",params.batch_normalization_85.Scale,"TrainedMean",params.batch_normalization_85.TrainedMean,"TrainedVariance",params.batch_normalization_85.TrainedVariance)
    reluLayer("Name","activation_85_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_80","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_80.Bias,"Weights",params.conv2d_80.Weights)
    batchNormalizationLayer("Name","batch_normalization_80","Epsilon",0.001,"Offset",params.batch_normalization_80.Offset,"Scale",params.batch_normalization_80.Scale,"TrainedMean",params.batch_normalization_80.TrainedMean,"TrainedVariance",params.batch_normalization_80.TrainedVariance)
    reluLayer("Name","activation_80_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_79","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_79.Bias,"Weights",params.conv2d_79.Weights)
    batchNormalizationLayer("Name","batch_normalization_79","Epsilon",0.001,"Offset",params.batch_normalization_79.Offset,"Scale",params.batch_normalization_79.Scale,"TrainedMean",params.batch_normalization_79.TrainedMean,"TrainedVariance",params.batch_normalization_79.TrainedVariance)
    reluLayer("Name","activation_79_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_84","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_84.Bias,"Weights",params.conv2d_84.Weights)
    batchNormalizationLayer("Name","batch_normalization_84","Epsilon",0.001,"Offset",params.batch_normalization_84.Offset,"Scale",params.batch_normalization_84.Scale,"TrainedMean",params.batch_normalization_84.TrainedMean,"TrainedVariance",params.batch_normalization_84.TrainedVariance)
    reluLayer("Name","activation_84_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_83","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_83.Bias,"Weights",params.conv2d_83.Weights)
    batchNormalizationLayer("Name","batch_normalization_83","Epsilon",0.001,"Offset",params.batch_normalization_83.Offset,"Scale",params.batch_normalization_83.Scale,"TrainedMean",params.batch_normalization_83.TrainedMean,"TrainedVariance",params.batch_normalization_83.TrainedVariance)
    reluLayer("Name","activation_83_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","concatenate_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","mixed9_0");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","mixed9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([3 3],"Name","average_pooling2d_9","Padding","same")
    convolution2dLayer([1 1],192,"Name","conv2d_94","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_94.Bias,"Weights",params.conv2d_94.Weights)
    batchNormalizationLayer("Name","batch_normalization_94","Epsilon",0.001,"Offset",params.batch_normalization_94.Offset,"Scale",params.batch_normalization_94.Scale,"TrainedMean",params.batch_normalization_94.TrainedMean,"TrainedVariance",params.batch_normalization_94.TrainedVariance)
    reluLayer("Name","activation_94_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","conv2d_87","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_87.Bias,"Weights",params.conv2d_87.Weights)
    batchNormalizationLayer("Name","batch_normalization_87","Epsilon",0.001,"Offset",params.batch_normalization_87.Offset,"Scale",params.batch_normalization_87.Scale,"TrainedMean",params.batch_normalization_87.TrainedMean,"TrainedVariance",params.batch_normalization_87.TrainedVariance)
    reluLayer("Name","activation_87_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],448,"Name","conv2d_90","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_90.Bias,"Weights",params.conv2d_90.Weights)
    batchNormalizationLayer("Name","batch_normalization_90","Epsilon",0.001,"Offset",params.batch_normalization_90.Offset,"Scale",params.batch_normalization_90.Scale,"TrainedMean",params.batch_normalization_90.TrainedMean,"TrainedVariance",params.batch_normalization_90.TrainedVariance)
    reluLayer("Name","activation_90_relu")
    convolution2dLayer([3 3],384,"Name","conv2d_91","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_91.Bias,"Weights",params.conv2d_91.Weights)
    batchNormalizationLayer("Name","batch_normalization_91","Epsilon",0.001,"Offset",params.batch_normalization_91.Offset,"Scale",params.batch_normalization_91.Scale,"TrainedMean",params.batch_normalization_91.TrainedMean,"TrainedVariance",params.batch_normalization_91.TrainedVariance)
    reluLayer("Name","activation_91_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_92","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_92.Bias,"Weights",params.conv2d_92.Weights)
    batchNormalizationLayer("Name","batch_normalization_92","Epsilon",0.001,"Offset",params.batch_normalization_92.Offset,"Scale",params.batch_normalization_92.Scale,"TrainedMean",params.batch_normalization_92.TrainedMean,"TrainedVariance",params.batch_normalization_92.TrainedVariance)
    reluLayer("Name","activation_92_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_93","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_93.Bias,"Weights",params.conv2d_93.Weights)
    batchNormalizationLayer("Name","batch_normalization_93","Epsilon",0.001,"Offset",params.batch_normalization_93.Offset,"Scale",params.batch_normalization_93.Scale,"TrainedMean",params.batch_normalization_93.TrainedMean,"TrainedVariance",params.batch_normalization_93.TrainedVariance)
    reluLayer("Name","activation_93_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],320,"Name","conv2d_86","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_86.Bias,"Weights",params.conv2d_86.Weights)
    batchNormalizationLayer("Name","batch_normalization_86","Epsilon",0.001,"Offset",params.batch_normalization_86.Offset,"Scale",params.batch_normalization_86.Scale,"TrainedMean",params.batch_normalization_86.TrainedMean,"TrainedVariance",params.batch_normalization_86.TrainedVariance)
    reluLayer("Name","activation_86_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],384,"Name","conv2d_89","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_89.Bias,"Weights",params.conv2d_89.Weights)
    batchNormalizationLayer("Name","batch_normalization_89","Epsilon",0.001,"Offset",params.batch_normalization_89.Offset,"Scale",params.batch_normalization_89.Scale,"TrainedMean",params.batch_normalization_89.TrainedMean,"TrainedVariance",params.batch_normalization_89.TrainedVariance)
    reluLayer("Name","activation_89_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],384,"Name","conv2d_88","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2d_88.Bias,"Weights",params.conv2d_88.Weights)
    batchNormalizationLayer("Name","batch_normalization_88","Epsilon",0.001,"Offset",params.batch_normalization_88.Offset,"Scale",params.batch_normalization_88.Scale,"TrainedMean",params.batch_normalization_88.TrainedMean,"TrainedVariance",params.batch_normalization_88.TrainedVariance)
    reluLayer("Name","activation_88_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","mixed9_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","concatenate_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","mixed10")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_7");
lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_9");
lgraph = connectLayers(lgraph,"max_pooling2d_2","average_pooling2d_1");
lgraph = connectLayers(lgraph,"max_pooling2d_2","conv2d_6");
lgraph = connectLayers(lgraph,"activation_12_relu","mixed0/in4");
lgraph = connectLayers(lgraph,"activation_6_relu","mixed0/in1");
lgraph = connectLayers(lgraph,"activation_8_relu","mixed0/in2");
lgraph = connectLayers(lgraph,"activation_11_relu","mixed0/in3");
lgraph = connectLayers(lgraph,"mixed0","conv2d_14");
lgraph = connectLayers(lgraph,"mixed0","conv2d_16");
lgraph = connectLayers(lgraph,"mixed0","average_pooling2d_2");
lgraph = connectLayers(lgraph,"mixed0","conv2d_13");
lgraph = connectLayers(lgraph,"activation_19_relu","mixed1/in4");
lgraph = connectLayers(lgraph,"activation_15_relu","mixed1/in2");
lgraph = connectLayers(lgraph,"activation_13_relu","mixed1/in1");
lgraph = connectLayers(lgraph,"activation_18_relu","mixed1/in3");
lgraph = connectLayers(lgraph,"mixed1","conv2d_23");
lgraph = connectLayers(lgraph,"mixed1","conv2d_21");
lgraph = connectLayers(lgraph,"mixed1","average_pooling2d_3");
lgraph = connectLayers(lgraph,"mixed1","conv2d_20");
lgraph = connectLayers(lgraph,"activation_22_relu","mixed2/in2");
lgraph = connectLayers(lgraph,"activation_26_relu","mixed2/in4");
lgraph = connectLayers(lgraph,"activation_20_relu","mixed2/in1");
lgraph = connectLayers(lgraph,"activation_25_relu","mixed2/in3");
lgraph = connectLayers(lgraph,"mixed2","max_pooling2d_3");
lgraph = connectLayers(lgraph,"mixed2","conv2d_27");
lgraph = connectLayers(lgraph,"mixed2","conv2d_28");
lgraph = connectLayers(lgraph,"max_pooling2d_3","mixed3/in3");
lgraph = connectLayers(lgraph,"activation_27_relu","mixed3/in1");
lgraph = connectLayers(lgraph,"activation_30_relu","mixed3/in2");
lgraph = connectLayers(lgraph,"mixed3","conv2d_32");
lgraph = connectLayers(lgraph,"mixed3","average_pooling2d_4");
lgraph = connectLayers(lgraph,"mixed3","conv2d_35");
lgraph = connectLayers(lgraph,"mixed3","conv2d_31");
lgraph = connectLayers(lgraph,"activation_40_relu","mixed4/in4");
lgraph = connectLayers(lgraph,"activation_34_relu","mixed4/in2");
lgraph = connectLayers(lgraph,"activation_39_relu","mixed4/in3");
lgraph = connectLayers(lgraph,"activation_31_relu","mixed4/in1");
lgraph = connectLayers(lgraph,"mixed4","average_pooling2d_5");
lgraph = connectLayers(lgraph,"mixed4","conv2d_45");
lgraph = connectLayers(lgraph,"mixed4","conv2d_42");
lgraph = connectLayers(lgraph,"mixed4","conv2d_41");
lgraph = connectLayers(lgraph,"activation_50_relu","mixed5/in4");
lgraph = connectLayers(lgraph,"activation_41_relu","mixed5/in1");
lgraph = connectLayers(lgraph,"activation_49_relu","mixed5/in3");
lgraph = connectLayers(lgraph,"activation_44_relu","mixed5/in2");
lgraph = connectLayers(lgraph,"mixed5","average_pooling2d_6");
lgraph = connectLayers(lgraph,"mixed5","conv2d_51");
lgraph = connectLayers(lgraph,"mixed5","conv2d_55");
lgraph = connectLayers(lgraph,"mixed5","conv2d_52");
lgraph = connectLayers(lgraph,"activation_60_relu","mixed6/in4");
lgraph = connectLayers(lgraph,"activation_54_relu","mixed6/in2");
lgraph = connectLayers(lgraph,"activation_51_relu","mixed6/in1");
lgraph = connectLayers(lgraph,"activation_59_relu","mixed6/in3");
lgraph = connectLayers(lgraph,"mixed6","average_pooling2d_7");
lgraph = connectLayers(lgraph,"mixed6","conv2d_65");
lgraph = connectLayers(lgraph,"mixed6","conv2d_61");
lgraph = connectLayers(lgraph,"mixed6","conv2d_62");
lgraph = connectLayers(lgraph,"activation_61_relu","mixed7/in1");
lgraph = connectLayers(lgraph,"activation_64_relu","mixed7/in2");
lgraph = connectLayers(lgraph,"activation_70_relu","mixed7/in4");
lgraph = connectLayers(lgraph,"activation_69_relu","mixed7/in3");
lgraph = connectLayers(lgraph,"mixed7","max_pooling2d_4");
lgraph = connectLayers(lgraph,"mixed7","conv2d_71");
lgraph = connectLayers(lgraph,"mixed7","conv2d_73");
lgraph = connectLayers(lgraph,"max_pooling2d_4","mixed8/in3");
lgraph = connectLayers(lgraph,"activation_72_relu","mixed8/in1");
lgraph = connectLayers(lgraph,"activation_76_relu","mixed8/in2");
lgraph = connectLayers(lgraph,"mixed8","conv2d_77");
lgraph = connectLayers(lgraph,"mixed8","conv2d_81");
lgraph = connectLayers(lgraph,"mixed8","conv2d_78");
lgraph = connectLayers(lgraph,"mixed8","average_pooling2d_8");
lgraph = connectLayers(lgraph,"activation_77_relu","mixed9/in1");
lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_80");
lgraph = connectLayers(lgraph,"activation_78_relu","conv2d_79");
lgraph = connectLayers(lgraph,"activation_80_relu","mixed9_0/in2");
lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_84");
lgraph = connectLayers(lgraph,"activation_82_relu","conv2d_83");
lgraph = connectLayers(lgraph,"activation_84_relu","concatenate_1/in2");
lgraph = connectLayers(lgraph,"activation_83_relu","concatenate_1/in1");
lgraph = connectLayers(lgraph,"concatenate_1","mixed9/in3");
lgraph = connectLayers(lgraph,"activation_79_relu","mixed9_0/in1");
lgraph = connectLayers(lgraph,"mixed9_0","mixed9/in2");
lgraph = connectLayers(lgraph,"activation_85_relu","mixed9/in4");
lgraph = connectLayers(lgraph,"mixed9","average_pooling2d_9");
lgraph = connectLayers(lgraph,"mixed9","conv2d_87");
lgraph = connectLayers(lgraph,"mixed9","conv2d_90");
lgraph = connectLayers(lgraph,"mixed9","conv2d_86");
lgraph = connectLayers(lgraph,"activation_91_relu","conv2d_92");
lgraph = connectLayers(lgraph,"activation_91_relu","conv2d_93");
lgraph = connectLayers(lgraph,"activation_93_relu","concatenate_2/in2");
lgraph = connectLayers(lgraph,"activation_86_relu","mixed10/in1");
lgraph = connectLayers(lgraph,"activation_94_relu","mixed10/in4");
lgraph = connectLayers(lgraph,"activation_87_relu","conv2d_89");
lgraph = connectLayers(lgraph,"activation_87_relu","conv2d_88");
lgraph = connectLayers(lgraph,"activation_89_relu","mixed9_1/in2");
lgraph = connectLayers(lgraph,"activation_88_relu","mixed9_1/in1");
lgraph = connectLayers(lgraph,"mixed9_1","mixed10/in2");
lgraph = connectLayers(lgraph,"activation_92_relu","concatenate_2/in1");
lgraph = connectLayers(lgraph,"concatenate_2","mixed10/in3");

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


%訓練神經網路
net = trainNetwork(imgTrain,lgraph,options);

%保存訓練好參數
save('C:\Users\joe\Desktop\CNN\InceptionV3testmini.mat','net');

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