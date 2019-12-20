function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

trialCVals     = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
trialSigmaVals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

bestError = 10^8;
bestC     = 1;    % arbitrary
bestSigma = 1;    % arbitrary

for trialC = trialCVals
    for trialSigma = trialSigmaVals
        % Train the SVM on the training set X and y using this combination
        % of trialC and trialSigma values
        model = svmTrain(X, y, trialC, ...
                         @(x1, x2) gaussianKernel(x1, x2, trialSigma));
        % Compute a prediction error for this trained model using the 
        % cross-validation set Xval
        predictions = svmPredict(model, Xval);
        % The error is defined as the fraction of the cross validation 
        % examples that were classified incorrectly
        predError = mean(double(predictions ~= yval));
        if predError < bestError
            bestError = predError;
            bestC     = trialC;
            bestSigma = trialSigma;
            %fprintf('New better predError=%0.5f: trial C=%0.5f, sigma=%0.5f \n', ...
            %        predError, bestC, bestSigma);
        end;
    end;
end;

C     = bestC;
sigma = bestSigma;
%fprintf('Returning best C=%0.5f, sigma=%0.5f with predError=%0.5f\n', ...
%        C, sigma, bestError);

% =========================================================================

end
