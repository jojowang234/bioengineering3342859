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
params = load("C:\Users\joe\Desktop\專題\params_2024_01_25__22_10_16.mat");

lgraph = layerGraph();
tempLayers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([3 3],32,"Name","Conv1","Padding","same","Stride",[2 2],"Bias",params.Conv1.Bias,"Weights",params.Conv1.Weights)
    batchNormalizationLayer("Name","bn_Conv1","Epsilon",0.001,"Offset",params.bn_Conv1.Offset,"Scale",params.bn_Conv1.Scale,"TrainedMean",params.bn_Conv1.TrainedMean,"TrainedVariance",params.bn_Conv1.TrainedVariance)
    clippedReluLayer(6,"Name","Conv1_relu")
    groupedConvolution2dLayer([3 3],1,32,"Name","expanded_conv_depthwise","Padding","same","Bias",params.expanded_conv_depthwise.Bias,"Weights",params.expanded_conv_depthwise.Weights)
    batchNormalizationLayer("Name","expanded_conv_depthwise_BN","Epsilon",0.001,"Offset",params.expanded_conv_depthwise_BN.Offset,"Scale",params.expanded_conv_depthwise_BN.Scale,"TrainedMean",params.expanded_conv_depthwise_BN.TrainedMean,"TrainedVariance",params.expanded_conv_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","expanded_conv_depthwise_relu")
    convolution2dLayer([1 1],16,"Name","expanded_conv_project","Padding","same","Bias",params.expanded_conv_project.Bias,"Weights",params.expanded_conv_project.Weights)
    batchNormalizationLayer("Name","expanded_conv_project_BN","Epsilon",0.001,"Offset",params.expanded_conv_project_BN.Offset,"Scale",params.expanded_conv_project_BN.Scale,"TrainedMean",params.expanded_conv_project_BN.TrainedMean,"TrainedVariance",params.expanded_conv_project_BN.TrainedVariance)
    convolution2dLayer([1 1],96,"Name","block_1_expand","Padding","same","Bias",params.block_1_expand.Bias,"Weights",params.block_1_expand.Weights)
    batchNormalizationLayer("Name","block_1_expand_BN","Epsilon",0.001,"Offset",params.block_1_expand_BN.Offset,"Scale",params.block_1_expand_BN.Scale,"TrainedMean",params.block_1_expand_BN.TrainedMean,"TrainedVariance",params.block_1_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_1_expand_relu")
    groupedConvolution2dLayer([3 3],1,96,"Name","block_1_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_1_depthwise.Bias,"Weights",params.block_1_depthwise.Weights)
    batchNormalizationLayer("Name","block_1_depthwise_BN","Epsilon",0.001,"Offset",params.block_1_depthwise_BN.Offset,"Scale",params.block_1_depthwise_BN.Scale,"TrainedMean",params.block_1_depthwise_BN.TrainedMean,"TrainedVariance",params.block_1_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_1_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_1_project","Padding","same","Bias",params.block_1_project.Bias,"Weights",params.block_1_project.Weights)
    batchNormalizationLayer("Name","block_1_project_BN","Epsilon",0.001,"Offset",params.block_1_project_BN.Offset,"Scale",params.block_1_project_BN.Scale,"TrainedMean",params.block_1_project_BN.TrainedMean,"TrainedVariance",params.block_1_project_BN.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],144,"Name","block_2_expand","Padding","same","Bias",params.block_2_expand.Bias,"Weights",params.block_2_expand.Weights)
    batchNormalizationLayer("Name","block_2_expand_BN","Epsilon",0.001,"Offset",params.block_2_expand_BN.Offset,"Scale",params.block_2_expand_BN.Scale,"TrainedMean",params.block_2_expand_BN.TrainedMean,"TrainedVariance",params.block_2_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_2_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_2_depthwise","Padding","same","Bias",params.block_2_depthwise.Bias,"Weights",params.block_2_depthwise.Weights)
    batchNormalizationLayer("Name","block_2_depthwise_BN","Epsilon",0.001,"Offset",params.block_2_depthwise_BN.Offset,"Scale",params.block_2_depthwise_BN.Scale,"TrainedMean",params.block_2_depthwise_BN.TrainedMean,"TrainedVariance",params.block_2_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_2_depthwise_relu")
    convolution2dLayer([1 1],24,"Name","block_2_project","Padding","same","Bias",params.block_2_project.Bias,"Weights",params.block_2_project.Weights)
    batchNormalizationLayer("Name","block_2_project_BN","Epsilon",0.001,"Offset",params.block_2_project_BN.Offset,"Scale",params.block_2_project_BN.Scale,"TrainedMean",params.block_2_project_BN.TrainedMean,"TrainedVariance",params.block_2_project_BN.TrainedVariance)];
 
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_2_add")
    convolution2dLayer([1 1],144,"Name","block_3_expand","Padding","same","Bias",params.block_3_expand.Bias,"Weights",params.block_3_expand.Weights)
    batchNormalizationLayer("Name","block_3_expand_BN","Epsilon",0.001,"Offset",params.block_3_expand_BN.Offset,"Scale",params.block_3_expand_BN.Scale,"TrainedMean",params.block_3_expand_BN.TrainedMean,"TrainedVariance",params.block_3_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_3_expand_relu")
    groupedConvolution2dLayer([3 3],1,144,"Name","block_3_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_3_depthwise.Bias,"Weights",params.block_3_depthwise.Weights)
    batchNormalizationLayer("Name","block_3_depthwise_BN","Epsilon",0.001,"Offset",params.block_3_depthwise_BN.Offset,"Scale",params.block_3_depthwise_BN.Scale,"TrainedMean",params.block_3_depthwise_BN.TrainedMean,"TrainedVariance",params.block_3_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_3_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_3_project","Padding","same","Bias",params.block_3_project.Bias,"Weights",params.block_3_project.Weights)
    batchNormalizationLayer("Name","block_3_project_BN","Epsilon",0.001,"Offset",params.block_3_project_BN.Offset,"Scale",params.block_3_project_BN.Scale,"TrainedMean",params.block_3_project_BN.TrainedMean,"TrainedVariance",params.block_3_project_BN.TrainedVariance)];
    
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_4_expand","Padding","same","Bias",params.block_4_expand.Bias,"Weights",params.block_4_expand.Weights)
    batchNormalizationLayer("Name","block_4_expand_BN","Epsilon",0.001,"Offset",params.block_4_expand_BN.Offset,"Scale",params.block_4_expand_BN.Scale,"TrainedMean",params.block_4_expand_BN.TrainedMean,"TrainedVariance",params.block_4_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_4_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_4_depthwise","Padding","same","Bias",params.block_4_depthwise.Bias,"Weights",params.block_4_depthwise.Weights)
    batchNormalizationLayer("Name","block_4_depthwise_BN","Epsilon",0.001,"Offset",params.block_4_depthwise_BN.Offset,"Scale",params.block_4_depthwise_BN.Scale,"TrainedMean",params.block_4_depthwise_BN.TrainedMean,"TrainedVariance",params.block_4_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_4_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_4_project","Padding","same","Bias",params.block_4_project.Bias,"Weights",params.block_4_project.Weights)
    batchNormalizationLayer("Name","block_4_project_BN","Epsilon",0.001,"Offset",params.block_4_project_BN.Offset,"Scale",params.block_4_project_BN.Scale,"TrainedMean",params.block_4_project_BN.TrainedMean,"TrainedVariance",params.block_4_project_BN.TrainedVariance)];
    
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_4_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","block_5_expand","Padding","same","Bias",params.block_5_expand.Bias,"Weights",params.block_5_expand.Weights)
    batchNormalizationLayer("Name","block_5_expand_BN","Epsilon",0.001,"Offset",params.block_5_expand_BN.Offset,"Scale",params.block_5_expand_BN.Scale,"TrainedMean",params.block_5_expand_BN.TrainedMean,"TrainedVariance",params.block_5_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_5_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_5_depthwise","Padding","same","Bias",params.block_5_depthwise.Bias,"Weights",params.block_5_depthwise.Weights)
    batchNormalizationLayer("Name","block_5_depthwise_BN","Epsilon",0.001,"Offset",params.block_5_depthwise_BN.Offset,"Scale",params.block_5_depthwise_BN.Scale,"TrainedMean",params.block_5_depthwise_BN.TrainedMean,"TrainedVariance",params.block_5_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_5_depthwise_relu")
    convolution2dLayer([1 1],32,"Name","block_5_project","Padding","same","Bias",params.block_5_project.Bias,"Weights",params.block_5_project.Weights)
    batchNormalizationLayer("Name","block_5_project_BN","Epsilon",0.001,"Offset",params.block_5_project_BN.Offset,"Scale",params.block_5_project_BN.Scale,"TrainedMean",params.block_5_project_BN.TrainedMean,"TrainedVariance",params.block_5_project_BN.TrainedVariance)];
    
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_5_add")
    convolution2dLayer([1 1],192,"Name","block_6_expand","Padding","same","Bias",params.block_6_expand.Bias,"Weights",params.block_6_expand.Weights)
    batchNormalizationLayer("Name","block_6_expand_BN","Epsilon",0.001,"Offset",params.block_6_expand_BN.Offset,"Scale",params.block_6_expand_BN.Scale,"TrainedMean",params.block_6_expand_BN.TrainedMean,"TrainedVariance",params.block_6_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_6_expand_relu")
    groupedConvolution2dLayer([3 3],1,192,"Name","block_6_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_6_depthwise.Bias,"Weights",params.block_6_depthwise.Weights)
    batchNormalizationLayer("Name","block_6_depthwise_BN","Epsilon",0.001,"Offset",params.block_6_depthwise_BN.Offset,"Scale",params.block_6_depthwise_BN.Scale,"TrainedMean",params.block_6_depthwise_BN.TrainedMean,"TrainedVariance",params.block_6_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_6_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_6_project","Padding","same","Bias",params.block_6_project.Bias,"Weights",params.block_6_project.Weights)
    batchNormalizationLayer("Name","block_6_project_BN","Epsilon",0.001,"Offset",params.block_6_project_BN.Offset,"Scale",params.block_6_project_BN.Scale,"TrainedMean",params.block_6_project_BN.TrainedMean,"TrainedVariance",params.block_6_project_BN.TrainedVariance)];
    
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_7_expand","Padding","same","Bias",params.block_7_expand.Bias,"Weights",params.block_7_expand.Weights)
    batchNormalizationLayer("Name","block_7_expand_BN","Epsilon",0.001,"Offset",params.block_7_expand_BN.Offset,"Scale",params.block_7_expand_BN.Scale,"TrainedMean",params.block_7_expand_BN.TrainedMean,"TrainedVariance",params.block_7_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_7_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_7_depthwise","Padding","same","Bias",params.block_7_depthwise.Bias,"Weights",params.block_7_depthwise.Weights)
    batchNormalizationLayer("Name","block_7_depthwise_BN","Epsilon",0.001,"Offset",params.block_7_depthwise_BN.Offset,"Scale",params.block_7_depthwise_BN.Scale,"TrainedMean",params.block_7_depthwise_BN.TrainedMean,"TrainedVariance",params.block_7_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_7_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_7_project","Padding","same","Bias",params.block_7_project.Bias,"Weights",params.block_7_project.Weights)
    batchNormalizationLayer("Name","block_7_project_BN","Epsilon",0.001,"Offset",params.block_7_project_BN.Offset,"Scale",params.block_7_project_BN.Scale,"TrainedMean",params.block_7_project_BN.TrainedMean,"TrainedVariance",params.block_7_project_BN.TrainedVariance)];
     
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_7_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_8_expand","Padding","same","Bias",params.block_8_expand.Bias,"Weights",params.block_8_expand.Weights)
    batchNormalizationLayer("Name","block_8_expand_BN","Epsilon",0.001,"Offset",params.block_8_expand_BN.Offset,"Scale",params.block_8_expand_BN.Scale,"TrainedMean",params.block_8_expand_BN.TrainedMean,"TrainedVariance",params.block_8_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_8_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_8_depthwise","Padding","same","Bias",params.block_8_depthwise.Bias,"Weights",params.block_8_depthwise.Weights)
    batchNormalizationLayer("Name","block_8_depthwise_BN","Epsilon",0.001,"Offset",params.block_8_depthwise_BN.Offset,"Scale",params.block_8_depthwise_BN.Scale,"TrainedMean",params.block_8_depthwise_BN.TrainedMean,"TrainedVariance",params.block_8_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_8_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_8_project","Padding","same","Bias",params.block_8_project.Bias,"Weights",params.block_8_project.Weights)
    batchNormalizationLayer("Name","block_8_project_BN","Epsilon",0.001,"Offset",params.block_8_project_BN.Offset,"Scale",params.block_8_project_BN.Scale,"TrainedMean",params.block_8_project_BN.TrainedMean,"TrainedVariance",params.block_8_project_BN.TrainedVariance)];
     
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_8_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],384,"Name","block_9_expand","Padding","same","Bias",params.block_9_expand.Bias,"Weights",params.block_9_expand.Weights)
    batchNormalizationLayer("Name","block_9_expand_BN","Epsilon",0.001,"Offset",params.block_9_expand_BN.Offset,"Scale",params.block_9_expand_BN.Scale,"TrainedMean",params.block_9_expand_BN.TrainedMean,"TrainedVariance",params.block_9_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_9_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_9_depthwise","Padding","same","Bias",params.block_9_depthwise.Bias,"Weights",params.block_9_depthwise.Weights)
    batchNormalizationLayer("Name","block_9_depthwise_BN","Epsilon",0.001,"Offset",params.block_9_depthwise_BN.Offset,"Scale",params.block_9_depthwise_BN.Scale,"TrainedMean",params.block_9_depthwise_BN.TrainedMean,"TrainedVariance",params.block_9_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_9_depthwise_relu")
    convolution2dLayer([1 1],64,"Name","block_9_project","Padding","same","Bias",params.block_9_project.Bias,"Weights",params.block_9_project.Weights)
    batchNormalizationLayer("Name","block_9_project_BN","Epsilon",0.001,"Offset",params.block_9_project_BN.Offset,"Scale",params.block_9_project_BN.Scale,"TrainedMean",params.block_9_project_BN.TrainedMean,"TrainedVariance",params.block_9_project_BN.TrainedVariance)];
    
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_9_add")
    convolution2dLayer([1 1],384,"Name","block_10_expand","Padding","same","Bias",params.block_10_expand.Bias,"Weights",params.block_10_expand.Weights)
    batchNormalizationLayer("Name","block_10_expand_BN","Epsilon",0.001,"Offset",params.block_10_expand_BN.Offset,"Scale",params.block_10_expand_BN.Scale,"TrainedMean",params.block_10_expand_BN.TrainedMean,"TrainedVariance",params.block_10_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_10_expand_relu")
    groupedConvolution2dLayer([3 3],1,384,"Name","block_10_depthwise","Padding","same","Bias",params.block_10_depthwise.Bias,"Weights",params.block_10_depthwise.Weights)
    batchNormalizationLayer("Name","block_10_depthwise_BN","Epsilon",0.001,"Offset",params.block_10_depthwise_BN.Offset,"Scale",params.block_10_depthwise_BN.Scale,"TrainedMean",params.block_10_depthwise_BN.TrainedMean,"TrainedVariance",params.block_10_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_10_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_10_project","Padding","same","Bias",params.block_10_project.Bias,"Weights",params.block_10_project.Weights)
    batchNormalizationLayer("Name","block_10_project_BN","Epsilon",0.001,"Offset",params.block_10_project_BN.Offset,"Scale",params.block_10_project_BN.Scale,"TrainedMean",params.block_10_project_BN.TrainedMean,"TrainedVariance",params.block_10_project_BN.TrainedVariance)];
     
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_11_expand","Padding","same","Bias",params.block_11_expand.Bias,"Weights",params.block_11_expand.Weights)
    batchNormalizationLayer("Name","block_11_expand_BN","Epsilon",0.001,"Offset",params.block_11_expand_BN.Offset,"Scale",params.block_11_expand_BN.Scale,"TrainedMean",params.block_11_expand_BN.TrainedMean,"TrainedVariance",params.block_11_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_11_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_11_depthwise","Padding","same","Bias",params.block_11_depthwise.Bias,"Weights",params.block_11_depthwise.Weights)
    batchNormalizationLayer("Name","block_11_depthwise_BN","Epsilon",0.001,"Offset",params.block_11_depthwise_BN.Offset,"Scale",params.block_11_depthwise_BN.Scale,"TrainedMean",params.block_11_depthwise_BN.TrainedMean,"TrainedVariance",params.block_11_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_11_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_11_project","Padding","same","Bias",params.block_11_project.Bias,"Weights",params.block_11_project.Weights)
    batchNormalizationLayer("Name","block_11_project_BN","Epsilon",0.001,"Offset",params.block_11_project_BN.Offset,"Scale",params.block_11_project_BN.Scale,"TrainedMean",params.block_11_project_BN.TrainedMean,"TrainedVariance",params.block_11_project_BN.TrainedVariance)];
     
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_11_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],576,"Name","block_12_expand","Padding","same","Bias",params.block_12_expand.Bias,"Weights",params.block_12_expand.Weights)
    batchNormalizationLayer("Name","block_12_expand_BN","Epsilon",0.001,"Offset",params.block_12_expand_BN.Offset,"Scale",params.block_12_expand_BN.Scale,"TrainedMean",params.block_12_expand_BN.TrainedMean,"TrainedVariance",params.block_12_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_12_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_12_depthwise","Padding","same","Bias",params.block_12_depthwise.Bias,"Weights",params.block_12_depthwise.Weights)
    batchNormalizationLayer("Name","block_12_depthwise_BN","Epsilon",0.001,"Offset",params.block_12_depthwise_BN.Offset,"Scale",params.block_12_depthwise_BN.Scale,"TrainedMean",params.block_12_depthwise_BN.TrainedMean,"TrainedVariance",params.block_12_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_12_depthwise_relu")
    convolution2dLayer([1 1],96,"Name","block_12_project","Padding","same","Bias",params.block_12_project.Bias,"Weights",params.block_12_project.Weights)
    batchNormalizationLayer("Name","block_12_project_BN","Epsilon",0.001,"Offset",params.block_12_project_BN.Offset,"Scale",params.block_12_project_BN.Scale,"TrainedMean",params.block_12_project_BN.TrainedMean,"TrainedVariance",params.block_12_project_BN.TrainedVariance)];
     
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","block_12_add")
    convolution2dLayer([1 1],576,"Name","block_13_expand","Padding","same","Bias",params.block_13_expand.Bias,"Weights",params.block_13_expand.Weights)
    batchNormalizationLayer("Name","block_13_expand_BN","Epsilon",0.001,"Offset",params.block_13_expand_BN.Offset,"Scale",params.block_13_expand_BN.Scale,"TrainedMean",params.block_13_expand_BN.TrainedMean,"TrainedVariance",params.block_13_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_13_expand_relu")
    groupedConvolution2dLayer([3 3],1,576,"Name","block_13_depthwise","Padding","same","Stride",[2 2],"Bias",params.block_13_depthwise.Bias,"Weights",params.block_13_depthwise.Weights)
    batchNormalizationLayer("Name","block_13_depthwise_BN","Epsilon",0.001,"Offset",params.block_13_depthwise_BN.Offset,"Scale",params.block_13_depthwise_BN.Scale,"TrainedMean",params.block_13_depthwise_BN.TrainedMean,"TrainedVariance",params.block_13_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_13_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_13_project","Padding","same","Bias",params.block_13_project.Bias,"Weights",params.block_13_project.Weights)
    batchNormalizationLayer("Name","block_13_project_BN","Epsilon",0.001,"Offset",params.block_13_project_BN.Offset,"Scale",params.block_13_project_BN.Scale,"TrainedMean",params.block_13_project_BN.TrainedMean,"TrainedVariance",params.block_13_project_BN.TrainedVariance)];
    
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_14_expand","Padding","same","Bias",params.block_14_expand.Bias,"Weights",params.block_14_expand.Weights)
    batchNormalizationLayer("Name","block_14_expand_BN","Epsilon",0.001,"Offset",params.block_14_expand_BN.Offset,"Scale",params.block_14_expand_BN.Scale,"TrainedMean",params.block_14_expand_BN.TrainedMean,"TrainedVariance",params.block_14_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_14_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_14_depthwise","Padding","same","Bias",params.block_14_depthwise.Bias,"Weights",params.block_14_depthwise.Weights)
    batchNormalizationLayer("Name","block_14_depthwise_BN","Epsilon",0.001,"Offset",params.block_14_depthwise_BN.Offset,"Scale",params.block_14_depthwise_BN.Scale,"TrainedMean",params.block_14_depthwise_BN.TrainedMean,"TrainedVariance",params.block_14_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_14_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_14_project","Padding","same","Bias",params.block_14_project.Bias,"Weights",params.block_14_project.Weights)
    batchNormalizationLayer("Name","block_14_project_BN","Epsilon",0.001,"Offset",params.block_14_project_BN.Offset,"Scale",params.block_14_project_BN.Scale,"TrainedMean",params.block_14_project_BN.TrainedMean,"TrainedVariance",params.block_14_project_BN.TrainedVariance)];
     
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","block_14_add");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],960,"Name","block_15_expand","Padding","same","Bias",params.block_15_expand.Bias,"Weights",params.block_15_expand.Weights)
    batchNormalizationLayer("Name","block_15_expand_BN","Epsilon",0.001,"Offset",params.block_15_expand_BN.Offset,"Scale",params.block_15_expand_BN.Scale,"TrainedMean",params.block_15_expand_BN.TrainedMean,"TrainedVariance",params.block_15_expand_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_15_expand_relu")
    groupedConvolution2dLayer([3 3],1,960,"Name","block_15_depthwise","Padding","same","Bias",params.block_15_depthwise.Bias,"Weights",params.block_15_depthwise.Weights)
    batchNormalizationLayer("Name","block_15_depthwise_BN","Epsilon",0.001,"Offset",params.block_15_depthwise_BN.Offset,"Scale",params.block_15_depthwise_BN.Scale,"TrainedMean",params.block_15_depthwise_BN.TrainedMean,"TrainedVariance",params.block_15_depthwise_BN.TrainedVariance)
    clippedReluLayer(6,"Name","block_15_depthwise_relu")
    convolution2dLayer([1 1],160,"Name","block_15_project","Padding","same","Bias",params.block_15_project.Bias,"Weights",params.block_15_project.Weights)
    batchNormalizationLayer("Name","block_15_project_BN","Epsilon",0.001,"Offset",params.block_15_project_BN.Offset,"Scale",params.block_15_project_BN.Scale,"TrainedMean",params.block_15_project_BN.TrainedMean,"TrainedVariance",params.block_15_project_BN.TrainedVariance)];
     
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2, "Name", "block_15_add")
    convolution2dLayer([1 1], 960, "Name", "block_16_expand", "Padding", "same", "Bias", params.block_16_expand.Bias, "Weights", params.block_16_expand.Weights)
    batchNormalizationLayer("Name", "block_16_expand_BN", "Epsilon", 0.001, "Offset", params.block_16_expand_BN.Offset, "Scale", params.block_16_expand_BN.Scale, "TrainedMean", params.block_16_expand_BN.TrainedMean, "TrainedVariance", params.block_16_expand_BN.TrainedVariance)
    clippedReluLayer(6, "Name", "block_16_expand_relu")
    groupedConvolution2dLayer([3 3], 1, 960, "Name", "block_16_depthwise", "Padding", "same", "Bias", params.block_16_depthwise.Bias, "Weights", params.block_16_depthwise.Weights)
    batchNormalizationLayer("Name", "block_16_depthwise_BN", "Epsilon", 0.001, "Offset", params.block_16_depthwise_BN.Offset, "Scale", params.block_16_depthwise_BN.Scale, "TrainedMean", params.block_16_depthwise_BN.TrainedMean, "TrainedVariance", params.block_16_depthwise_BN.TrainedVariance)
    clippedReluLayer(6, "Name", "block_16_depthwise_relu")
    convolution2dLayer([1 1], 320, "Name", "block_16_project", "Padding", "same", "Bias", params.block_16_project.Bias, "Weights", params.block_16_project.Weights)
    batchNormalizationLayer("Name", "block_16_project_BN", "Epsilon", 0.001, "Offset", params.block_16_project_BN.Offset, "Scale", params.block_16_project_BN.Scale, "TrainedMean", params.block_16_project_BN.TrainedMean, "TrainedVariance", params.block_16_project_BN.TrainedVariance)
    convolution2dLayer([1 1], 1280, "Name", "Conv_1", "Bias", params.Conv_1.Bias, "Weights", params.Conv_1.Weights)
    batchNormalizationLayer("Name", "Conv_1_bn", "Epsilon", 0.001, "Offset", params.Conv_1_bn.Offset, "Scale", params.Conv_1_bn.Scale, "TrainedMean", params.Conv_1_bn.TrainedMean, "TrainedVariance", params.Conv_1_bn.TrainedVariance)
    clippedReluLayer(6, "Name", "out_relu")
    globalAveragePooling2dLayer("Name", "global_average_pooling2d_1")
    fullyConnectedLayer(2, "Name", "fc")
    softmaxLayer("Name", "softmax")
    classificationLayer("Name", "classoutput")
];
lgraph = addLayers(lgraph, tempLayers);

% clean up helper variable
clear tempLayers;

% 連接層
lgraph = connectLayers(lgraph, "block_1_project_BN", "block_2_expand");
lgraph = connectLayers(lgraph, "block_1_project_BN", "block_2_add/in2");
lgraph = connectLayers(lgraph, "block_2_project_BN", "block_2_add/in1");
lgraph = connectLayers(lgraph, "block_3_project_BN", "block_4_expand");
lgraph = connectLayers(lgraph, "block_3_project_BN", "block_4_add/in2");
lgraph = connectLayers(lgraph, "block_4_project_BN", "block_4_add/in1");
lgraph = connectLayers(lgraph, "block_4_add", "block_5_expand");
lgraph = connectLayers(lgraph, "block_4_add", "block_5_add/in2");
lgraph = connectLayers(lgraph, "block_5_project_BN", "block_5_add/in1");
lgraph = connectLayers(lgraph, "block_6_project_BN", "block_7_expand");
lgraph = connectLayers(lgraph, "block_6_project_BN", "block_7_add/in2");
lgraph = connectLayers(lgraph, "block_7_project_BN", "block_7_add/in1");
lgraph = connectLayers(lgraph, "block_7_add", "block_8_expand");
lgraph = connectLayers(lgraph, "block_7_add", "block_8_add/in2");
lgraph = connectLayers(lgraph, "block_8_project_BN", "block_8_add/in1");
lgraph = connectLayers(lgraph, "block_8_add", "block_9_expand");
lgraph = connectLayers(lgraph, "block_8_add", "block_9_add/in2");
lgraph = connectLayers(lgraph, "block_9_project_BN", "block_9_add/in1");
lgraph = connectLayers(lgraph, "block_10_project_BN", "block_11_expand");
lgraph = connectLayers(lgraph, "block_10_project_BN", "block_11_add/in2");
lgraph = connectLayers(lgraph, "block_11_project_BN", "block_11_add/in1");
lgraph = connectLayers(lgraph, "block_11_add", "block_12_expand");
lgraph = connectLayers(lgraph, "block_11_add", "block_12_add/in2");
lgraph = connectLayers(lgraph, "block_12_project_BN", "block_12_add/in1");
lgraph = connectLayers(lgraph, "block_13_project_BN", "block_14_expand");
lgraph = connectLayers(lgraph, "block_13_project_BN", "block_14_add/in2");
lgraph = connectLayers(lgraph, "block_14_project_BN", "block_14_add/in1");
lgraph = connectLayers(lgraph, "block_14_add", "block_15_expand");
lgraph = connectLayers(lgraph, "block_14_add", "block_15_add/in2");
lgraph = connectLayers(lgraph, "block_15_project_BN", "block_15_add/in1");

% 配置訓練參數
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

net = trainNetwork(imgTrain, lgraph, options);

% 保存訓練好參數
save('C:\Users\joe\Desktop\CNN\MobileNettestmini.mat', 'net');

% 測試模型精度
YPred = classify(net, imgTest);
YTest = imgTest.Labels;
Accuracy = sum(YPred == YTest) / numel(YTest);

% 計時器
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