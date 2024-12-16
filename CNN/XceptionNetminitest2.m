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
params = load("C:\Users\joe\Desktop\專題\params_2024_01_25__21_00_13.mat");

lgraph = layerGraph();
tempLayers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([3 3],32,"Name","block1_conv1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.block1_conv1.Bias,"Weights",params.block1_conv1.Weights)
    batchNormalizationLayer("Name","block1_conv1_bn","Epsilon",0.001,"Offset",params.block1_conv1_bn.Offset,"Scale",params.block1_conv1_bn.Scale,"TrainedMean",params.block1_conv1_bn.TrainedMean,"TrainedVariance",params.block1_conv1_bn.TrainedVariance)
    reluLayer("Name","block1_conv1_act")
    convolution2dLayer([3 3],64,"Name","block1_conv2","BiasLearnRateFactor",0,"Bias",params.block1_conv2.Bias,"Weights",params.block1_conv2.Weights)
    batchNormalizationLayer("Name","block1_conv2_bn","Epsilon",0.001,"Offset",params.block1_conv2_bn.Offset,"Scale",params.block1_conv2_bn.Scale,"TrainedMean",params.block1_conv2_bn.TrainedMean,"TrainedVariance",params.block1_conv2_bn.TrainedVariance)
    reluLayer("Name","block1_conv2_act")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([3 3],1,64,"Name","block2_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block2_sepconv1_channel_wise.Bias,"Weights",params.block2_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],128,"Name","block2_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block2_sepconv1_point_wise.Bias,"Weights",params.block2_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block2_sepconv1_bn","Epsilon",0.001,"Offset",params.block2_sepconv1_bn.Offset,"Scale",params.block2_sepconv1_bn.Scale,"TrainedMean",params.block2_sepconv1_bn.TrainedMean,"TrainedVariance",params.block2_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block2_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,128,"Name","block2_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block2_sepconv2_channel_wise.Bias,"Weights",params.block2_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],128,"Name","block2_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block2_sepconv2_point_wise.Bias,"Weights",params.block2_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block2_sepconv2_bn","Epsilon",0.001,"Offset",params.block2_sepconv2_bn.Offset,"Scale",params.block2_sepconv2_bn.Scale,"TrainedMean",params.block2_sepconv2_bn.TrainedMean,"TrainedVariance",params.block2_sepconv2_bn.TrainedVariance)
    maxPooling2dLayer([3 3],"Name","block2_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv2d_1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.conv2d_1.Bias,"Weights",params.conv2d_1.Weights)
    batchNormalizationLayer("Name","batch_normalization_1","Epsilon",0.001,"Offset",params.batch_normalization_1.Offset,"Scale",params.batch_normalization_1.Scale,"TrainedMean",params.batch_normalization_1.TrainedMean,"TrainedVariance",params.batch_normalization_1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block3_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,128,"Name","block3_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block3_sepconv1_channel_wise.Bias,"Weights",params.block3_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],256,"Name","block3_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block3_sepconv1_point_wise.Bias,"Weights",params.block3_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block3_sepconv1_bn","Epsilon",0.001,"Offset",params.block3_sepconv1_bn.Offset,"Scale",params.block3_sepconv1_bn.Scale,"TrainedMean",params.block3_sepconv1_bn.TrainedMean,"TrainedVariance",params.block3_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block3_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,256,"Name","block3_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block3_sepconv2_channel_wise.Bias,"Weights",params.block3_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],256,"Name","block3_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block3_sepconv2_point_wise.Bias,"Weights",params.block3_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block3_sepconv2_bn","Epsilon",0.001,"Offset",params.block3_sepconv2_bn.Offset,"Scale",params.block3_sepconv2_bn.Scale,"TrainedMean",params.block3_sepconv2_bn.TrainedMean,"TrainedVariance",params.block3_sepconv2_bn.TrainedVariance)
    maxPooling2dLayer([3 3],"Name","block3_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv2d_2","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.conv2d_2.Bias,"Weights",params.conv2d_2.Weights)
    batchNormalizationLayer("Name","batch_normalization_2","Epsilon",0.001,"Offset",params.batch_normalization_2.Offset,"Scale",params.batch_normalization_2.Scale,"TrainedMean",params.batch_normalization_2.TrainedMean,"TrainedVariance",params.batch_normalization_2.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block4_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,256,"Name","block4_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block4_sepconv1_channel_wise.Bias,"Weights",params.block4_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block4_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block4_sepconv1_point_wise.Bias,"Weights",params.block4_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block4_sepconv1_bn","Epsilon",0.001,"Offset",params.block4_sepconv1_bn.Offset,"Scale",params.block4_sepconv1_bn.Scale,"TrainedMean",params.block4_sepconv1_bn.TrainedMean,"TrainedVariance",params.block4_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block4_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block4_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block4_sepconv2_channel_wise.Bias,"Weights",params.block4_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block4_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block4_sepconv2_point_wise.Bias,"Weights",params.block4_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block4_sepconv2_bn","Epsilon",0.001,"Offset",params.block4_sepconv2_bn.Offset,"Scale",params.block4_sepconv2_bn.Scale,"TrainedMean",params.block4_sepconv2_bn.TrainedMean,"TrainedVariance",params.block4_sepconv2_bn.TrainedVariance)
    maxPooling2dLayer([3 3],"Name","block4_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],728,"Name","conv2d_3","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.conv2d_3.Bias,"Weights",params.conv2d_3.Weights)
    batchNormalizationLayer("Name","batch_normalization_3","Epsilon",0.001,"Offset",params.batch_normalization_3.Offset,"Scale",params.batch_normalization_3.Scale,"TrainedMean",params.batch_normalization_3.TrainedMean,"TrainedVariance",params.batch_normalization_3.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block5_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block5_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block5_sepconv1_channel_wise.Bias,"Weights",params.block5_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block5_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block5_sepconv1_point_wise.Bias,"Weights",params.block5_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block5_sepconv1_bn","Epsilon",0.001,"Offset",params.block5_sepconv1_bn.Offset,"Scale",params.block5_sepconv1_bn.Scale,"TrainedMean",params.block5_sepconv1_bn.TrainedMean,"TrainedVariance",params.block5_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block5_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block5_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block5_sepconv2_channel_wise.Bias,"Weights",params.block5_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block5_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block5_sepconv2_point_wise.Bias,"Weights",params.block5_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block5_sepconv2_bn","Epsilon",0.001,"Offset",params.block5_sepconv2_bn.Offset,"Scale",params.block5_sepconv2_bn.Scale,"TrainedMean",params.block5_sepconv2_bn.TrainedMean,"TrainedVariance",params.block5_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block5_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block5_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block5_sepconv3_channel_wise.Bias,"Weights",params.block5_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block5_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block5_sepconv3_point_wise.Bias,"Weights",params.block5_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block5_sepconv3_bn","Epsilon",0.001,"Offset",params.block5_sepconv3_bn.Offset,"Scale",params.block5_sepconv3_bn.Scale,"TrainedMean",params.block5_sepconv3_bn.TrainedMean,"TrainedVariance",params.block5_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block6_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block6_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block6_sepconv1_channel_wise.Bias,"Weights",params.block6_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block6_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block6_sepconv1_point_wise.Bias,"Weights",params.block6_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block6_sepconv1_bn","Epsilon",0.001,"Offset",params.block6_sepconv1_bn.Offset,"Scale",params.block6_sepconv1_bn.Scale,"TrainedMean",params.block6_sepconv1_bn.TrainedMean,"TrainedVariance",params.block6_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block6_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block6_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block6_sepconv2_channel_wise.Bias,"Weights",params.block6_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block6_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block6_sepconv2_point_wise.Bias,"Weights",params.block6_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block6_sepconv2_bn","Epsilon",0.001,"Offset",params.block6_sepconv2_bn.Offset,"Scale",params.block6_sepconv2_bn.Scale,"TrainedMean",params.block6_sepconv2_bn.TrainedMean,"TrainedVariance",params.block6_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block6_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block6_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block6_sepconv3_channel_wise.Bias,"Weights",params.block6_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block6_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block6_sepconv3_point_wise.Bias,"Weights",params.block6_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block6_sepconv3_bn","Epsilon",0.001,"Offset",params.block6_sepconv3_bn.Offset,"Scale",params.block6_sepconv3_bn.Scale,"TrainedMean",params.block6_sepconv3_bn.TrainedMean,"TrainedVariance",params.block6_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block7_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block7_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block7_sepconv1_channel_wise.Bias,"Weights",params.block7_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block7_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block7_sepconv1_point_wise.Bias,"Weights",params.block7_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block7_sepconv1_bn","Epsilon",0.001,"Offset",params.block7_sepconv1_bn.Offset,"Scale",params.block7_sepconv1_bn.Scale,"TrainedMean",params.block7_sepconv1_bn.TrainedMean,"TrainedVariance",params.block7_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block7_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block7_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block7_sepconv2_channel_wise.Bias,"Weights",params.block7_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block7_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block7_sepconv2_point_wise.Bias,"Weights",params.block7_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block7_sepconv2_bn","Epsilon",0.001,"Offset",params.block7_sepconv2_bn.Offset,"Scale",params.block7_sepconv2_bn.Scale,"TrainedMean",params.block7_sepconv2_bn.TrainedMean,"TrainedVariance",params.block7_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block7_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block7_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block7_sepconv3_channel_wise.Bias,"Weights",params.block7_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block7_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block7_sepconv3_point_wise.Bias,"Weights",params.block7_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block7_sepconv3_bn","Epsilon",0.001,"Offset",params.block7_sepconv3_bn.Offset,"Scale",params.block7_sepconv3_bn.Scale,"TrainedMean",params.block7_sepconv3_bn.TrainedMean,"TrainedVariance",params.block7_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block8_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block8_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block8_sepconv1_channel_wise.Bias,"Weights",params.block8_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block8_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block8_sepconv1_point_wise.Bias,"Weights",params.block8_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block8_sepconv1_bn","Epsilon",0.001,"Offset",params.block8_sepconv1_bn.Offset,"Scale",params.block8_sepconv1_bn.Scale,"TrainedMean",params.block8_sepconv1_bn.TrainedMean,"TrainedVariance",params.block8_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block8_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block8_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block8_sepconv2_channel_wise.Bias,"Weights",params.block8_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block8_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block8_sepconv2_point_wise.Bias,"Weights",params.block8_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block8_sepconv2_bn","Epsilon",0.001,"Offset",params.block8_sepconv2_bn.Offset,"Scale",params.block8_sepconv2_bn.Scale,"TrainedMean",params.block8_sepconv2_bn.TrainedMean,"TrainedVariance",params.block8_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block8_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block8_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block8_sepconv3_channel_wise.Bias,"Weights",params.block8_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block8_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block8_sepconv3_point_wise.Bias,"Weights",params.block8_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block8_sepconv3_bn","Epsilon",0.001,"Offset",params.block8_sepconv3_bn.Offset,"Scale",params.block8_sepconv3_bn.Scale,"TrainedMean",params.block8_sepconv3_bn.TrainedMean,"TrainedVariance",params.block8_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block9_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block9_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block9_sepconv1_channel_wise.Bias,"Weights",params.block9_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block9_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block9_sepconv1_point_wise.Bias,"Weights",params.block9_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block9_sepconv1_bn","Epsilon",0.001,"Offset",params.block9_sepconv1_bn.Offset,"Scale",params.block9_sepconv1_bn.Scale,"TrainedMean",params.block9_sepconv1_bn.TrainedMean,"TrainedVariance",params.block9_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block9_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block9_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block9_sepconv2_channel_wise.Bias,"Weights",params.block9_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block9_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block9_sepconv2_point_wise.Bias,"Weights",params.block9_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block9_sepconv2_bn","Epsilon",0.001,"Offset",params.block9_sepconv2_bn.Offset,"Scale",params.block9_sepconv2_bn.Scale,"TrainedMean",params.block9_sepconv2_bn.TrainedMean,"TrainedVariance",params.block9_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block9_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block9_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block9_sepconv3_channel_wise.Bias,"Weights",params.block9_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block9_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block9_sepconv3_point_wise.Bias,"Weights",params.block9_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block9_sepconv3_bn","Epsilon",0.001,"Offset",params.block9_sepconv3_bn.Offset,"Scale",params.block9_sepconv3_bn.Scale,"TrainedMean",params.block9_sepconv3_bn.TrainedMean,"TrainedVariance",params.block9_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_8");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block10_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block10_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block10_sepconv1_channel_wise.Bias,"Weights",params.block10_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block10_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block10_sepconv1_point_wise.Bias,"Weights",params.block10_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block10_sepconv1_bn","Epsilon",0.001,"Offset",params.block10_sepconv1_bn.Offset,"Scale",params.block10_sepconv1_bn.Scale,"TrainedMean",params.block10_sepconv1_bn.TrainedMean,"TrainedVariance",params.block10_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block10_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block10_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block10_sepconv2_channel_wise.Bias,"Weights",params.block10_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block10_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block10_sepconv2_point_wise.Bias,"Weights",params.block10_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block10_sepconv2_bn","Epsilon",0.001,"Offset",params.block10_sepconv2_bn.Offset,"Scale",params.block10_sepconv2_bn.Scale,"TrainedMean",params.block10_sepconv2_bn.TrainedMean,"TrainedVariance",params.block10_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block10_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block10_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block10_sepconv3_channel_wise.Bias,"Weights",params.block10_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block10_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block10_sepconv3_point_wise.Bias,"Weights",params.block10_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block10_sepconv3_bn","Epsilon",0.001,"Offset",params.block10_sepconv3_bn.Offset,"Scale",params.block10_sepconv3_bn.Scale,"TrainedMean",params.block10_sepconv3_bn.TrainedMean,"TrainedVariance",params.block10_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_9");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block11_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block11_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block11_sepconv1_channel_wise.Bias,"Weights",params.block11_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block11_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block11_sepconv1_point_wise.Bias,"Weights",params.block11_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block11_sepconv1_bn","Epsilon",0.001,"Offset",params.block11_sepconv1_bn.Offset,"Scale",params.block11_sepconv1_bn.Scale,"TrainedMean",params.block11_sepconv1_bn.TrainedMean,"TrainedVariance",params.block11_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block11_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block11_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block11_sepconv2_channel_wise.Bias,"Weights",params.block11_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block11_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block11_sepconv2_point_wise.Bias,"Weights",params.block11_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block11_sepconv2_bn","Epsilon",0.001,"Offset",params.block11_sepconv2_bn.Offset,"Scale",params.block11_sepconv2_bn.Scale,"TrainedMean",params.block11_sepconv2_bn.TrainedMean,"TrainedVariance",params.block11_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block11_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block11_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block11_sepconv3_channel_wise.Bias,"Weights",params.block11_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block11_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block11_sepconv3_point_wise.Bias,"Weights",params.block11_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block11_sepconv3_bn","Epsilon",0.001,"Offset",params.block11_sepconv3_bn.Offset,"Scale",params.block11_sepconv3_bn.Scale,"TrainedMean",params.block11_sepconv3_bn.TrainedMean,"TrainedVariance",params.block11_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_10");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block12_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block12_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block12_sepconv1_channel_wise.Bias,"Weights",params.block12_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block12_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block12_sepconv1_point_wise.Bias,"Weights",params.block12_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block12_sepconv1_bn","Epsilon",0.001,"Offset",params.block12_sepconv1_bn.Offset,"Scale",params.block12_sepconv1_bn.Scale,"TrainedMean",params.block12_sepconv1_bn.TrainedMean,"TrainedVariance",params.block12_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block12_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block12_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block12_sepconv2_channel_wise.Bias,"Weights",params.block12_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block12_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block12_sepconv2_point_wise.Bias,"Weights",params.block12_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block12_sepconv2_bn","Epsilon",0.001,"Offset",params.block12_sepconv2_bn.Offset,"Scale",params.block12_sepconv2_bn.Scale,"TrainedMean",params.block12_sepconv2_bn.TrainedMean,"TrainedVariance",params.block12_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block12_sepconv3_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block12_sepconv3_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block12_sepconv3_channel_wise.Bias,"Weights",params.block12_sepconv3_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block12_sepconv3_point-wise","BiasLearnRateFactor",0,"Bias",params.block12_sepconv3_point_wise.Bias,"Weights",params.block12_sepconv3_point_wise.Weights)
    batchNormalizationLayer("Name","block12_sepconv3_bn","Epsilon",0.001,"Offset",params.block12_sepconv3_bn.Offset,"Scale",params.block12_sepconv3_bn.Scale,"TrainedMean",params.block12_sepconv3_bn.TrainedMean,"TrainedVariance",params.block12_sepconv3_bn.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","add_11");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],1024,"Name","conv2d_4","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.conv2d_4.Bias,"Weights",params.conv2d_4.Weights)
    batchNormalizationLayer("Name","batch_normalization_4","Epsilon",0.001,"Offset",params.batch_normalization_4.Offset,"Scale",params.batch_normalization_4.Scale,"TrainedMean",params.batch_normalization_4.TrainedMean,"TrainedVariance",params.batch_normalization_4.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    reluLayer("Name","block13_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block13_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block13_sepconv1_channel_wise.Bias,"Weights",params.block13_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],728,"Name","block13_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block13_sepconv1_point_wise.Bias,"Weights",params.block13_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block13_sepconv1_bn","Epsilon",0.001,"Offset",params.block13_sepconv1_bn.Offset,"Scale",params.block13_sepconv1_bn.Scale,"TrainedMean",params.block13_sepconv1_bn.TrainedMean,"TrainedVariance",params.block13_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block13_sepconv2_act")
    groupedConvolution2dLayer([3 3],1,728,"Name","block13_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block13_sepconv2_channel_wise.Bias,"Weights",params.block13_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],1024,"Name","block13_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block13_sepconv2_point_wise.Bias,"Weights",params.block13_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block13_sepconv2_bn","Epsilon",0.001,"Offset",params.block13_sepconv2_bn.Offset,"Scale",params.block13_sepconv2_bn.Scale,"TrainedMean",params.block13_sepconv2_bn.TrainedMean,"TrainedVariance",params.block13_sepconv2_bn.TrainedVariance)
    maxPooling2dLayer([3 3],"Name","block13_pool","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","add_12")
    groupedConvolution2dLayer([3 3],1,1024,"Name","block14_sepconv1_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block14_sepconv1_channel_wise.Bias,"Weights",params.block14_sepconv1_channel_wise.Weights)
    convolution2dLayer([1 1],1536,"Name","block14_sepconv1_point-wise","BiasLearnRateFactor",0,"Bias",params.block14_sepconv1_point_wise.Bias,"Weights",params.block14_sepconv1_point_wise.Weights)
    batchNormalizationLayer("Name","block14_sepconv1_bn","Epsilon",0.001,"Offset",params.block14_sepconv1_bn.Offset,"Scale",params.block14_sepconv1_bn.Scale,"TrainedMean",params.block14_sepconv1_bn.TrainedMean,"TrainedVariance",params.block14_sepconv1_bn.TrainedVariance)
    reluLayer("Name","block14_sepconv1_act")
    groupedConvolution2dLayer([3 3],1,1536,"Name","block14_sepconv2_channel-wise","BiasLearnRateFactor",0,"Padding","same","Bias",params.block14_sepconv2_channel_wise.Bias,"Weights",params.block14_sepconv2_channel_wise.Weights)
    convolution2dLayer([1 1],2048,"Name","block14_sepconv2_point-wise","BiasLearnRateFactor",0,"Bias",params.block14_sepconv2_point_wise.Bias,"Weights",params.block14_sepconv2_point_wise.Weights)
    batchNormalizationLayer("Name","block14_sepconv2_bn","Epsilon",0.001,"Offset",params.block14_sepconv2_bn.Offset,"Scale",params.block14_sepconv2_bn.Scale,"TrainedMean",params.block14_sepconv2_bn.TrainedMean,"TrainedVariance",params.block14_sepconv2_bn.TrainedVariance)
    reluLayer("Name","block14_sepconv2_act")
    globalAveragePooling2dLayer("Name","avg_pool")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"block1_conv2_act","block2_sepconv1_channel-wise");
lgraph = connectLayers(lgraph,"block1_conv2_act","conv2d_1");
lgraph = connectLayers(lgraph,"batch_normalization_1","add_1/in2");
lgraph = connectLayers(lgraph,"block2_pool","add_1/in1");
lgraph = connectLayers(lgraph,"add_1","block3_sepconv1_act");
lgraph = connectLayers(lgraph,"add_1","conv2d_2");
lgraph = connectLayers(lgraph,"batch_normalization_2","add_2/in2");
lgraph = connectLayers(lgraph,"block3_pool","add_2/in1");
lgraph = connectLayers(lgraph,"add_2","block4_sepconv1_act");
lgraph = connectLayers(lgraph,"add_2","conv2d_3");
lgraph = connectLayers(lgraph,"batch_normalization_3","add_3/in2");
lgraph = connectLayers(lgraph,"block4_pool","add_3/in1");
lgraph = connectLayers(lgraph,"add_3","block5_sepconv1_act");
lgraph = connectLayers(lgraph,"add_3","add_4/in2");
lgraph = connectLayers(lgraph,"block5_sepconv3_bn","add_4/in1");
lgraph = connectLayers(lgraph,"add_4","block6_sepconv1_act");
lgraph = connectLayers(lgraph,"add_4","add_5/in2");
lgraph = connectLayers(lgraph,"block6_sepconv3_bn","add_5/in1");
lgraph = connectLayers(lgraph,"add_5","block7_sepconv1_act");
lgraph = connectLayers(lgraph,"add_5","add_6/in2");
lgraph = connectLayers(lgraph,"block7_sepconv3_bn","add_6/in1");
lgraph = connectLayers(lgraph,"add_6","block8_sepconv1_act");
lgraph = connectLayers(lgraph,"add_6","add_7/in2");
lgraph = connectLayers(lgraph,"block8_sepconv3_bn","add_7/in1");
lgraph = connectLayers(lgraph,"add_7","block9_sepconv1_act");
lgraph = connectLayers(lgraph,"add_7","add_8/in2");
lgraph = connectLayers(lgraph,"block9_sepconv3_bn","add_8/in1");
lgraph = connectLayers(lgraph,"add_8","block10_sepconv1_act");
lgraph = connectLayers(lgraph,"add_8","add_9/in2");
lgraph = connectLayers(lgraph,"block10_sepconv3_bn","add_9/in1");
lgraph = connectLayers(lgraph,"add_9","block11_sepconv1_act");
lgraph = connectLayers(lgraph,"add_9","add_10/in2");
lgraph = connectLayers(lgraph,"block11_sepconv3_bn","add_10/in1");
lgraph = connectLayers(lgraph,"add_10","block12_sepconv1_act");
lgraph = connectLayers(lgraph,"add_10","add_11/in2");
lgraph = connectLayers(lgraph,"block12_sepconv3_bn","add_11/in1");
lgraph = connectLayers(lgraph,"add_11","conv2d_4");
lgraph = connectLayers(lgraph,"add_11","block13_sepconv1_act");
lgraph = connectLayers(lgraph,"batch_normalization_4","add_12/in2");
lgraph = connectLayers(lgraph,"block13_pool","add_12/in1");

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
net2 = trainNetwork(imgTrain,lgraph,options);

%保存訓練好參數
save('C:\Users\joe\Desktop\CNN\XceptionNettestmini2.mat','net2');

%測試模型精度
% 使用net2進行分類
YPred = classify(net2, imgTest);

% 提取真實標籤
YTest = imgTest.Labels;

% 計算準確度
Accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('準確度: %.2f\n', Accuracy);

% 計時器
toc

% 獲取測試圖片的預測分數
yScores = predict(net2, imgTest);

% 將真實標籤轉換為索引
yTrue = grp2idx(YTest);

% 獲取預測標籤（具有最高分數的類別）
[~, yPred] = max(yScores, [], 2);

% 生成混淆矩陣
confMat = confusionmat(yTrue, yPred);

% 提取真陽性、假陽性和假陰性值
TP = confMat(2, 2);
FP = confMat(1, 2);
FN = confMat(2, 1);

% 計算召回率和精確率指標
recallMetric = TP / (TP + FN);
precisionMetric = TP / (TP + FP);

% 列印召回率和精確率指標
fprintf('Recall: %.2f\n', recallMetric);
fprintf('Precision: %.2f\n', precisionMetric);

% 計算精確率-召回率曲線
[prec, rec, ~] = perfcurve(yTrue, yScores(:, 2), 1, 'xCrit', 'reca', 'yCrit', 'prec');

% 繪製精確率-召回率曲線
figure;
plot(rec, prec);
xlabel('Recall');
ylabel('Precision');
title('Precision-Recall Curve');
grid on;