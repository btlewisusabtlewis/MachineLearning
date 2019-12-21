function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X     - num_movies x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Compute cost function J
pred = X*Theta';
variance = (pred-Y) .^ 2;
reg = (lambda/2) * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));
J = sum(sum(variance .* R))/2 + reg;

% Compute partial derivatives X_grad and Theta_grad
for i = 1:num_movies
    % For movie i, we want derivatives X_grad(i,:) wrt each feature.
    % First, create a list idx of all the _users_ who rated movie i.
    idx = find(R(i, :) == 1);  
    % Use idx to restrict the derivative computation to users who rated i
    Theta_temp = Theta(idx, :);
    Y_temp     = Y(i, idx);
    reg = (lambda * X(i, :));
    X_grad(i, :) = ((X(i, :)*Theta_temp' - Y_temp) * Theta_temp) + reg;
    % X:     (nm × nf)
    % Y:     (nm × nu).
    % Theta: (nu × nf).
    % ((1 x nf)*(nf x nu) - (1 x nu)) * (nu x nf) = (1 x nf)
end;
for j = 1:num_users
    % For user j, we want derivatives Theta_grad(j,:) wrt each feature.
    % Here we create a list idx of all the _movies_ that user j rated.
    idx = find(R(:, j) == 1);  
    % Now we restrict the derivative computation to _movies_ that j rated.
    Y_temp     = Y(idx, j);  % (nm x 1)
    X_temp     = X(idx, :);  % (nm x nf)
    reg = (lambda * Theta(j, :));
    Theta_grad(j, :) = ((X_temp*Theta(j, :)' - Y_temp)' * X_temp) + reg;
    % ((nm x nf)*(nf x 1) - (nm x 1))' * (nm x nf) = (1 x nm) * (nm x nf) = (1 x nf)
end;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
