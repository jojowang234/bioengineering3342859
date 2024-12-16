clc;
clear;

% 加載網絡
load('C:\Users\joe\Desktop\CNN\Efficientnettestmini.mat', 'net');

% 讀取圖像文件
[file, path] = uigetfile('*', '選擇圖像文件');
if isequal(file, 0)
    disp('用戶取消了文件選擇');
    return;
end

% 完整文件路徑
imagePath = fullfile(path, file);

% 讀取圖像
I = imread(imagePath);

% 檢查是否為灰度圖像，如有需要轉換為RGB
if size(I, 3) == 1
    I = cat(3, I, I, I); 
end

% 調整圖像尺寸到網絡輸入尺寸
inputSize = net.Layers(1).InputSize; % 獲取網絡輸入層的尺寸
I_resized = imresize(I, [inputSize(1), inputSize(2)]);

% 顯示文件名
disp(['選擇的文件：', file]);

% 開始計時
tic;

% 使用CNN進行預測以獲取概率分佈
probabilities = predict(net, I_resized);

% 找到最可能的標籤和相應的概率
[maxProb, index] = max(probabilities);
label = net.Layers(end).Classes(index);

% 停止計時
elapsedTime = toc;

% 顯示分類結果和準確率
disp(['該圖像最可能屬於類別：', char(label)]);
disp(['識別準確率（概率）：', num2str(maxProb)]);
disp(['預測時間：', num2str(elapsedTime), ' 秒']);

% 顯示圖像和預測結果
figure;
imshow(I);
title(['預測類別: ', char(label), ', 概率: ', num2str(maxProb)]);

% 打印預測信息到控制台
fprintf('預測時間: %.2f 秒\n', elapsedTime);