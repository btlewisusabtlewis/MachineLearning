function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X is mx2 matrix.  y is mx1 column vector.  theta is 2x1 column vector.
h = X*theta;       % a mx1 column vector of predicted (hypothesis) values
error = h - y;     % a mx1 column vector

% Compute the cost J, a scalar
J_orig = (1/(2*m)) * sum(error .^ 2);   
tmp = theta;
tmp(1) = 0;
J_reg = (lambda/(2*m)) * sum(tmp .^ 2); 
J = J_orig + J_reg;

% Now compute the gradient = (n+1) x 1 column vector of partial derivatives
grad_orig = (1/m) * (X' * error);
% Compute a regularization vector for each gradient element but the first
grad_reg = (lambda/m) * theta;
grad_reg(1) = 0;
grad = grad_orig + grad_reg;  % a 2x1 column vector
% =========================================================================

grad = grad(:);

end
