# almond-type-classification-using-ANN
function Almond_GUI()
    % Create the main figure
    fig = uifigure('Name', 'ANN Classifier for Almond Dataset', 'Position', [100, 100, 600, 400]);

    % Title Label
    uilabel(fig, 'Position', [200, 350, 200, 30], 'Text', 'Almond ANN Classifier', ...
        'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

    % File Selection
    uilabel(fig, 'Position', [50, 300, 100, 22], 'Text', 'Dataset File:');
    datasetField = uieditfield(fig, 'text', 'Position', [150, 300, 300, 22]);
    browseButton = uibutton(fig, 'Text', 'Browse', 'Position', [460, 300, 70, 22], ...
        'ButtonPushedFcn', @(btn, event) selectDataset(datasetField));

    % Start Training Button
    trainButton = uibutton(fig, 'Text', 'Train Model', 'Position', [150, 200, 100, 30], ...
        'ButtonPushedFcn', @(btn, event) trainANN(datasetField.Value));

    % Open Prediction Window Button
    predictionButton = uibutton(fig, 'Text', 'Predict Almond Type', 'Position', [300, 200, 150, 30], ...
        'ButtonPushedFcn', @(btn, event) openPredictionWindow(), 'Enable', 'off');

    % Status Display
    statusLabel = uilabel(fig, 'Position', [50, 100, 500, 22], 'Text', 'Status: Waiting for input.', ...
        'FontSize', 12, 'FontColor', [0, 0, 1]);

    % Variables to store trained model and normalization parameters
    trainedNet = [];
    normParams = struct('min', [], 'range', []);
    almondTypes = {};
    featureNames = {};

    % Function for Dataset Selection
    function selectDataset(field)
        [file, path] = uigetfile('*.csv', 'Select Dataset');
        if isequal(file, 0)
            field.Value = '';
            statusLabel.Text = 'Status: No file selected.';
            statusLabel.FontColor = [1, 0, 0];
        else
            field.Value = fullfile(path, file);
            statusLabel.Text = ['Status: Selected ', file];
            statusLabel.FontColor = [0, 0, 1];
        end
    end

    % Function to Train the ANN
    function trainANN(datasetPath)
        if isempty(datasetPath)
            uialert(fig, 'Please select a dataset before training.', 'Error', 'Icon', 'error');
            return;
        end
        try
            % Train the model and update the GUI
            [trainedNet, normParams, accuracy, cvAccuracy, almondTypes, featureNames] = trainAlmondANN(datasetPath);

            % Display results
            statusLabel.Text = sprintf('Status: Training completed! Accuracy: %.2f%%', accuracy * 100);
            uialert(fig, sprintf('Training Completed!\n\nAccuracy: %.2f%%\nMean CV Accuracy: %.2f%%', ...
                accuracy * 100, cvAccuracy * 100), 'Training Results', 'Icon', 'success');

            % Enable prediction window button
            predictionButton.Enable = 'on';

        catch ME
            statusLabel.Text = 'Status: Error during training.';
            statusLabel.FontColor = [1, 0, 0];
            uialert(fig, ME.message, 'Error', 'Icon', 'error');
        end
    end

    % Function to Open Prediction Window
    function openPredictionWindow()
        % Create new window for prediction
        predFig = uifigure('Name', 'Predict Almond Type', 'Position', [100, 100, 500, 400]);

        % Title
        uilabel(predFig, 'Position', [150, 350, 200, 30], 'Text', 'Enter Feature Values', ...
            'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

        % Create dynamic input fields for features
        numFeatures = numel(featureNames);
        featureInputs = cell(1, numFeatures);
        for i = 1:numFeatures
            uilabel(predFig, 'Position', [50, 300 - i * 40, 150, 22], 'Text', featureNames{i});
            featureInputs{i} = uieditfield(predFig, 'numeric', 'Position', [250, 300 - i * 40, 200, 22]);
        end

        % Predict Button
        uibutton(predFig, 'Text', 'Predict', 'Position', [200, 50, 100, 30], ...
            'ButtonPushedFcn', @(btn, event) predictType(featureInputs));
    end

    % Function to Predict Almond Type
    function predictType(featureInputs)
        if isempty(trainedNet)
            uialert(fig, 'Please train the model before predicting.', 'Error', 'Icon', 'error');
            return;
        end
        try
            % Read input features
            features = zeros(1, numel(featureInputs));
            for i = 1:numel(featureInputs)
                features(i) = str2double(featureInputs{i}.Value);
                if isnan(features(i))
                    uialert(fig, sprintf('Feature "%s" is invalid.', featureNames{i}), 'Error', 'Icon', 'error');
                    return;
                end
            end

            % Normalize the features
            normalizedFeatures = (features - normParams.min) ./ normParams.range;

            % Predict the type
            predictions = trainedNet(normalizedFeatures');
            [~, predictedLabelIdx] = max(predictions, [], 1);

            % Display prediction
            predictedType = almondTypes{predictedLabelIdx};
            uialert(fig, sprintf('Predicted Almond Type: %s', predictedType), 'Prediction Result', 'Icon', 'info');
        catch ME
            uialert(fig, ME.message, 'Error', 'Icon', 'error');
        end
    end
end

function [net, normParams, accuracy, cvAccuracy, almondTypes, featureNames] = trainAlmondANN(filename)
    % Load the dataset
    data = readtable(filename);

    % Clean and standardize feature names
    data.Properties.VariableNames = matlab.lang.makeValidName(data.Properties.VariableNames);

    % Handle missing values column by column
    for col = 2:width(data) - 1  % Skip the first (index) and last (Type) columns
        if isnumeric(data{:, col})  % Only process numeric columns
            colMean = mean(data{:, col}, 'omitnan');  % Calculate mean ignoring NaN
            data{:, col} = fillmissing(data{:, col}, 'constant', colMean);
        end
    end

    % Convert 'Type' to categorical
    if iscell(data.Type)
        data.Type = categorical(data.Type);
    end

    % Extract features and labels
    featureNames = data.Properties.VariableNames(2:end-1); % Extract valid feature names
    features = data{:, 2:end-1}; % All numeric columns except the label
    labels = data.Type;          % Categorical labels

    % Normalize the features
    normParams.min = min(features, [], 1);
    normParams.range = range(features, 1);
    features = (features - normParams.min) ./ normParams.range;

    % Convert labels to numeric for ANN
    almondTypes = categories(labels);
    labelsNumeric = grp2idx(labels);

    % Split the data into training and testing sets
    cv = cvpartition(labelsNumeric, 'HoldOut', 0.3);
    xTrain = features(training(cv), :);
    yTrain = labelsNumeric(training(cv));
    xTest = features(test(cv), :);
    yTest = labelsNumeric(test(cv));

    % Define the ANN architecture
    net = patternnet([50, 50, 50]); % Three hidden layers with 50 neurons each
    net.trainFcn = 'trainscg'; % Scaled conjugate gradient
    net.performFcn = 'crossentropy'; % Use cross-entropy loss
    net.trainParam.epochs = 1000; % Increase epochs

    % Train the network
    [net, ~] = train(net, xTrain', dummyvar(yTrain)');

    % Test the network
    predictions = net(xTest');
    [~, predictedLabels] = max(predictions, [], 1);

    % Evaluate the model
    confMat = confusionmat(yTest, predictedLabels);
    accuracy = sum(diag(confMat)) / sum(confMat, 'all');

    % Perform k-fold cross-validation
    k = 5; % Number of folds
    cv = cvpartition(labelsNumeric, 'KFold', k);
    accuracyList = zeros(1, k);

    for i = 1:k
        xTrain = features(training(cv, i), :);
        yTrain = labelsNumeric(training(cv, i));
        xTest = features(test(cv, i), :);
        yTest = labelsNumeric(test(cv, i));

        % Train the network
        [net, ~] = train(net, xTrain', dummyvar(yTrain)');

        % Test the network
        predictions = net(xTest');
        [~, predictedLabels] = max(predictions, [], 1);

        % Evaluate accuracy
        confMat = confusionmat(yTest, predictedLabels);
        accuracyList(i) = sum(diag(confMat)) / sum(confMat, 'all');
    end

    cvAccuracy = mean(accuracyList);
end
