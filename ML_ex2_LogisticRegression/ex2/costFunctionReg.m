function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% theta is (n+1) x 1
% X is m x (n+1)
z = X * theta;
h_theta = sigmoid(z);  % column vector, m x 1

% Compute the cost function J
costIf1 = -y' * log(h_theta);           % scalar
costIf0 = (1 - y') * log(1 - h_theta);  % also calar
cost = (costIf1 - costIf0)/m;           % so scalar
thetaSquared = theta .* theta;
thetaSquared(1) = 0;                    % don't regularize theta(1)
reg = (lambda/(2*m)) * sum(thetaSquared);
J = cost + reg;

% Now compute the gradient = (n+1) x 1 column vector of partial derivatives
error = h_theta - y;
grad_orig = (1/m) * (X' * error);
% Compute a regularization vector for each gradient element but the first
grad_reg = (lambda/m) * theta;
grad_reg(1) = 0;
grad = grad_orig + grad_reg;

% =============================================================

end
