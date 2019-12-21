function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

% X has already had the initial ones column added, so it is (mx2).

% Options for fmincg, which is used to find the theta_i.
options = optimset('MaxIter', 200, 'GradObj', 'on');
    
for i = 1:m
    % Find theta_i for the "first i" training set X(1:i, :) and y(1:i).
    XFirstI = X(1:i, :);
    costFunction = @(t) linearRegCostFunction(XFirstI, y(1:i), t, lambda);
    % Initial theta passed to fmincg to find theta for this training set.
    initial_theta = zeros(size(XFirstI, 2), 1);    
    theta_i = fmincg(costFunction, initial_theta, options);
    
    % Now compute the errors on the training and cross-validation sets.
    % No regularization is used for these errors (lambda is zero).
    lambdaForCompErrors = 0;
    % Training set error uses just the "first i" training set.
    error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), ...
                                           theta_i, lambdaForCompErrors);
    % Cross-validation error uses the entire (Xval, yval) set.
    error_val(i) = linearRegCostFunction(Xval, yval, ...
                                         theta_i, lambdaForCompErrors);
end;

% -------------------------------------------------------------

% =========================================================================

end
