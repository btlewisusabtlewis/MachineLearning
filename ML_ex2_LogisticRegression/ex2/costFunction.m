function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% n is two
% theta is (n+1) x 1
% X is m x (n+1)
z = X * theta;
h_theta = sigmoid(z);  % column vector, m x 1

% Compute the cost function J
lhs = -y' * log(h_theta);  % scalar
rhs = (1 - y') * log(1 - h_theta);
J = (lhs - rhs)/m;

% Now compute the gradient = (n+1) x 1 column vector of partial derivatives
error = h_theta - y;
grad = (1/m) * (X' * error);

% =============================================================

end
