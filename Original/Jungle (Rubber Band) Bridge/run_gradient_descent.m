%save in file named 'run_gradient_descent.m'
%this function runs gradient descent to find the local minimum
%of an input function given some initial guess
%INPUTS:
%fun: the function we want to optimize
%V0: the initial guess for gradient descent
%params: a struct defining the optimization parameters
%params.beta: threshold for choosing alpha (step size scaling) via backtracking line-search
%params.gamma: growth/decay multiplier for backtracking line-search
%params.max_iter: maximum number of iterations for gradient descent
%params.min_gradient: termination condition for gradient descent; If ||grad(f)||<min_gradient, terminate program
%OUTPUTS:
%Vopt: The guess for the local minimum of V0
function Vopt = run_gradient_descent_1(fun,V0,params)
    %unpack params
    beta = params.beta;
    gamma = params.gamma;
    max_iter = params.max_iter;
    min_gradient = params.min_gradient;

    %your code here

end