function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Matrix whose cols/rows are the vectors of activation values for each label
% E.g., label_vec(3,:) = [0, 0, 1, 0, ... 0].
label_vec = diag(ones(num_labels, 1));  

% Feedforward computation
% Vectorized with one column per example
a1 = [ones(m, 1), X]';  % with bias terms added, a1 is (401 x m)

z2 = Theta1 * a1;      
a2 = sigmoid(z2);      % a2 is (25 x 401) * (401 x m) = (25 x m)
a2 = [ones(1, m); a2]; % with bias terms added, a2 is now (26 x m)
    
z3 = Theta2 * a2;      
a3 = sigmoid(z3);      % a3 is (10 x 26) * (26 x m) = (10 x m)
h_theta = a3;          % (10 x m), column i has prediction for example i

% Compute cost J
J = 0;
for i = 1:m
    y_vec = label_vec(y(i), :);  % y_vec is (1 x 10), a row vector
    pred  = h_theta(:, i);       % pred  is (10 x 1)
    cost_unreg = ((-y_vec * log(pred)) - ((1 - y_vec) * log(1 - pred)))/m;
    J = J + cost_unreg;
end;

% Compute a regularization term for the cost J
theta1_sum = sum(sum( Theta1(:, 2:end) .^ 2 ));
theta2_sum = sum(sum( Theta2(:, 2:end) .^ 2 ));
reg_term = (lambda/(2*m)) * (theta1_sum + theta2_sum);
J = J + reg_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Initialize the gradients being accumulated 
Delta1 = 0;
Delta2 = 0;
for t = 1:m
    %% Feedforward computation
    %
    a1 = X(t, :)';          % a1 is (400 x 1) column vector
    a1 = [1 ; a1];          % with bias term added, a1 is now (401 x 1)

    z2 = Theta1 * a1;
    a2 = sigmoid(z2);       % a2 is (25 x 401) * (401 x 1) = (25 x 1)
    a2 = [1 ; a2];          % with bias term added, a2 is now (26 x 1)
    
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);      % a3 is (10 x 26) * (26 x 1) = (10 x 1)
    h_theta = a3;          % (10 x 1) column vector prediction for example i
    
    %% Now start backpropogation
    %
    % Output layer error
    y_vec = label_vec(:, y(t));  % y_vec is (10 x 1), a column vector
    % Could have used a logical array comparison here
    d3 = (a3 - y_vec); % (10 x 1) column vector

    % Hidden layer error
    % Must discard delta2(1), so delta2 is (25 x 10) * (10 x 1) = (25 x 1)
    d2 = ((Theta2(:, 2:end))' * d3) .* sigmoidGradient(z2);
    
    % Accumulate gradients
    Delta2 = Delta2 + (d3 * a2'); % matrix D2 is (10 x 1) * (1 x 26)  = (10 x 26)
    Delta1 = Delta1 + (d2 * a1'); % matrix D1 is (25 x 1) * (1 x 401) = (25 x 401)
end;
% Get the (unregularized) gradient for the neural network cost function
Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

tmp = Theta1;
tmp(:, 1) = 0;
Reg1_grad = (1/m) * (lambda * tmp);

tmp = Theta2;
tmp(:, 1) = 0;
Reg2_grad = (1/m) * (lambda * tmp);

Theta1_grad = (Theta1_grad + Reg1_grad);
Theta2_grad = (Theta2_grad + Reg2_grad);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
