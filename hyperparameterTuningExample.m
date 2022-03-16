addpath('utils')

% Download the dataset if it does not already exist
% The full dev-train-100 dataset is around 6 GB of data and may take a while to download
downloadDatasetFolder = tempdir;
disp('Downloading...')
datasetFolder = downloadDataset(downloadDatasetFolder);
disp('Finished!')

% Create an audioDatastore object to access the LibriSpeech audio data
ADS = audioDatastore(datasetFolder,'IncludeSubfolders',1);

% Extract the speaker label from the file path
ADS.Labels = extractBetween(ADS.Files,fullfile(datasetFolder,filesep),filesep);

% To train the network with data from all 251 speakers, set reduceDataset to false
% To run this example quickly with data from just six speakers, set reduceDataset to true
reducedDataSet = true;
if reducedDataSet
    indices = cellfun(@(c)str2double(c)<50,ADS.Labels);  %#ok
    ADS = subset(ADS,indices);
end
ADS = splitEachLabel(ADS,0.3);

% Split the audio files into training and test data 
% 80% of the audio files are assigned to the training set and 20% are assigned to the test set
[ADSTrain,ADSTest] = splitEachLabel(ADS,0.8);

% Set the parameters for preprocessing
[audioIn,dsInfo] = read(ADSTrain);
Fs = dsInfo.SampleRate;
frameDuration = 200e-3;
overlapDuration = 40e-3;
frameLength = floor(Fs*frameDuration); 
overlapLength = round(Fs*overlapDuration);

disp('Preprocessing Data...')
[XTrain,YTrain] = preprocessAudioData(ADSTrain,frameLength,overlapLength,Fs);
[XTest,YTest] = preprocessAudioData(ADSTest,frameLength,overlapLength,Fs);

% Save training and test data to .mat
disp('Saving...')
save('SpeakerIdentificationProject/audioTrainingData.mat','XTrain','YTrain','XTest','YTest');

% Save reduced dataset with uniform distribution of labels
XTrain_red = [];
YTrain_red = [];
XTest_red = [];
YTest_red = [];
trainSize = 50;
testSize = 5;
labels = unique(YTrain);
for i = 1:length(labels)
    label = labels(i);
    trainIdx = find(YTrain == label, trainSize);
    testIdx = find(YTest == label, testSize);
    XTrain_red(end+1:end+trainSize,:,:) = XTrain(trainIdx,:,:);
    YTrain_red(end+1:end+trainSize,:) = YTrain(trainIdx,:);
    XTest_red(end+1:end+testSize,:,:) = XTest(testIdx,:,:);
    YTest_red(end+1:end+testSize,:) = YTest(testIdx,:);
end
XTrain = XTrain_red;
YTrain = YTrain_red;
XTest = XTest_red;
YTest = YTest_red;
save('SpeakerIdentificationProject/smallerAudioTrainingData.mat','XTrain','YTrain','XTest','YTest');

clear

% Check Python configuration
checkPythonSetup

% Open Experiment Manager
experimentManager
