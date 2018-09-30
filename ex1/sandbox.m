clear
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];
% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters


    errors = (X*theta)-y;
    errorsX = errors.*X;
    theta = theta - (alpha*(1/m)*sum(errorsX))';  
    J = (1/(2*m))*sum(((X*theta)-y).^2);
end