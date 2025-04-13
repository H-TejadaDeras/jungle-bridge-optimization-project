%% Approximate Gradient Function (First Order Approximation)
%INPUTS:
%fun: the function we want to optimize (2 variables only)
%OUTPUTS:
%G: approximate gradient
function G = approximate_gradient(fun, V)
    % Variable Declarations
    h = 1e-5;
    G = zeros(length(V), 1);
    
    delta_x = zeros(length(V), 1);

    for i = 1:length(V)
        delta_x(i) = h;

        f_minus = fun(V-delta_x);
        f_plus = fun(V+delta_x);

        G(i) = (f_plus - f_minus)/(2*h);

        delta_x(i) = 0;
    end
end