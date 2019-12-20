function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Note: X      is m  x 400 
%       m      is 5000
%       Theta1 is 25 x 401
%       Theta2 is 10 x 26

% Add bias term to X, producing a (401 x m) matrix a1 for input layer 1
a1 = [ones(m, 1) X]';

% Compute hidden layer's values, a (26 x m) matrix a2 (with bias added)
z2 = Theta1 * a1;      % z2 is (25 x m) 
a2 = sigmoid(z2);
a2 = [ones(1, m); a2]; % a2 is (26 x m)

% Compute output layer's values, a (10 x m) matrix a3
z3 = Theta2 * a2;  % (10 x m) result matrix z2 for layer 2
a3 = sigmoid(z3);

% For each example [1:m], compute probability it matches each label
[maxVal, p] = max(a3', [], 2);  % p is (m x 1)

% =========================================================================

end
