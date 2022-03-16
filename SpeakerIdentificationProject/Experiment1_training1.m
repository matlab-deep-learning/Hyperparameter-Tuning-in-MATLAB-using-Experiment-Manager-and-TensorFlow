function output = Experiment1_training1(params,monitor)

    output.TrainedNetworkPath = [];
    % Setup training metrics and Info
    monitor.Metrics = ["TrainingLoss","TrainingAccuracy","TestLoss","TestAccuracy"];
    % Specify the file name for the saved network
    modelFile = 'mymodel';

    monitor.Status = "Loading";
    
    % Load preprocessed training and test data
    reducedDataset = true;
    if reducedDataset
        % Set reducedDataset to true to use the reduced dataset.
        % Set reducedDataset to false to use the full dataset.
        load('smallerAudioTrainingData.mat','XTrain','YTrain','XTest','YTest');
    else
        load('audioTrainingData.mat','XTrain','YTrain','XTest','YTest');
    end

    monitor.Status = "Training";
    monitor.Progress = 30;

    % Call the Python training function
    result = py.trainer.train( ...
        py.numpy.array(XTrain), ...
        py.numpy.array(YTrain), ...
        py.numpy.array(XTest), ...
        py.numpy.array(YTest), ...
        params, ...
        modelFile);
    
    % Create training plot, update results table metrics and info from
    % results (training loss and accuracy plots)
    numEpochs = double(result{1}.params{"epochs"});
    for i = 1:numEpochs
        loss = result{1}.history{"loss"}{i};
        acc = result{1}.history{"accuracy"}{i};
        monitor.recordMetrics(i,"TrainingLoss",loss,"TrainingAccuracy",acc);
    end

    % Update test loss and test accuracy
    monitor.recordMetrics(numEpochs,"TestLoss",result{2}{1},"TestAccuracy",result{2}{2});

    % Get the path to the trained network for the trial
    output.TrainedNetworkPath = fullfile(pwd,modelFile);

    monitor.Progress = 100;

end