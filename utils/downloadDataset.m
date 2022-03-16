function datasetFolder = downloadDataset(downloadDatasetFolder)
    filename = "train-clean-100.tar.gz";
    url = "http://www.openSLR.org/resources/12/" + filename;
    
    datasetFolder = fullfile(downloadDatasetFolder,"LibriSpeech","train-clean-100");
    
    if ~isfolder(datasetFolder)
        gunzip(url,downloadDatasetFolder);
        unzippedFile = fullfile(downloadDatasetFolder,filename);
        untar(unzippedFile{1}(1:end-3),downloadDatasetFolder);
    end
end