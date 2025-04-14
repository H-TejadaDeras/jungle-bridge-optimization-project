%% Run Gradient Descent
%% Return V_list value if at a certain iteration; save in above function
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
function Vopt = run_gradient_descent(fun,V0,params)
    %unpack params
    beta = params.beta;
    gamma = params.gamma;
    max_iter = params.max_iter;
    min_gradient = params.min_gradient;

    %set initial values for alpha, V, and n
    alpha = 1; V = V0; n = 0;
    
    %evaluate gradient and function for first time
    G = approximate_gradient(fun,V);
    F = fun(V);
    %iterate until either gradient is sufficiently small
    %or we hit the max iteration limit
    while n<max_iter && norm(G)>min_gradient
        %compute the scare of the norm of the gradient
        NG2 = norm(G)^2;

        %run line search algorithm to find alpha
        while fun(V-alpha*G)<F-beta*alpha*NG2
            alpha = alpha/gamma;
        end

        while fun(V-alpha*G)>F-beta*alpha*NG2
            alpha = alpha*gamma;
        end

        %once alpha has been found, update guess
        V = V-alpha*G;

        %evaluate gradient and function at new value of V
        G = approximate_gradient(fun,V);
        F = fun(V);

        %increment our iteration counter
        n = n+1;
    end
    %return final value of V as our numerical solution
    Vopt = V;

end