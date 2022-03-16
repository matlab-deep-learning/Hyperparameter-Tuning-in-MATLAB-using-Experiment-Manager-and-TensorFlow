function [X,Y] = preprocessAudioData(ADS,SL,OL,Fs)

    if ~isempty(ver("parallel"))
        pool = gcp;
        numPar = numpartitions(ADS,pool);
    else
        numPar = 1;
    end
    
    parfor ii = 1:numPar
    
        X = zeros(1,SL,1,0);
        Y = zeros(0);
        subADS = partition(ADS,numPar,ii);
        
        while hasdata(subADS)
            [audioIn,dsInfo] = read(subADS);
            
            speechIdx = detectSpeech(audioIn,Fs);
            numChunks = size(speechIdx,1);
            audioData = zeros(1,SL,1,0);      
            
            for chunk = 1:numChunks
                % Remove trail end audio
                audio_chunk = audioIn(speechIdx(chunk,1):speechIdx(chunk,2));
                audio_chunk = buffer(audio_chunk,SL,OL);
                q = size(audio_chunk,2);
                
                % Split audio into 200 ms chunks
                audio_chunk = reshape(audio_chunk,1,SL,1,q);
                
                % Concatenate with existing audio
                audioData = cat(4,audioData,audio_chunk);
            end
            
            audioLabel = str2double(dsInfo.Label{1});
            
            % Generate labels for training and testing by replecating matrix
            audioLabelsTrain = repmat(audioLabel,1,size(audioData,4));
            
            % Add data points for current speaker to existing data
            X = cat(4,X,audioData);
            Y = cat(2,Y,audioLabelsTrain);
        end
            
        XC{ii} = X;
        YC{ii} = Y;
    end
    
    X = cat(4,XC{:});
    Y = cat(2,YC{:});
    
    % Permute the data as TensorFlow expects the first dimension to be the batch
    % dimension
    X = permute(X, [4,1,2,3]);
    Y = permute(Y, [2,1]);

    % TensorFlow expects labels to be in range [0,6)
    labels = sort(unique(Y));
    for i = 1:length(labels)
        label = labels(i);
        Y(Y==label) = i-1;
    end
    
end