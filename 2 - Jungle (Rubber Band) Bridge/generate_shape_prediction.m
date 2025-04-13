%% use gradient descent to predict the shape of the bridge
%INPUTS:
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of rubber bands in bridge
% param_struct.k_list = [k_1;...;k_n]: list of stiffnesses
% param_struct.l0_list = [l0_1;...;l0_n]: list of natural lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
%OUTPUTS:
%x_list = [x_0;x_1;...;x_n]: x coordinates of predicted vertex positions
%y_list = [y_0;y_1;...;y_n]: x coordinates of predicted vertex positions
function [x_list,y_list] = generate_shape_prediction(param_struct)
    %specify optimization parameters
    opt_params = struct();
    opt_params.beta = 0.5;
    opt_params.gamma = 0.25;
    opt_params.max_iter = 1000;
    opt_params.min_gradient = 1e-7;

    %use anonymous function syntax to define the cost func
    %define cost func as the total potential energy function
    %using the current values in param_struct
    f_cost = @(V_in) total_potential_func(V_in,param_struct);

    %generate an initial guess for the coordinate locations
    %coords_guess = [x_1;y_1;...;x_(n-1);y_(n-1)]

    x0 = param_struct.r0(1);
    y0 = param_struct.r0(2);
    xn = param_struct.rn(1);
    yn = param_struct.rn(2);
    x_guess = linspace(x0,xn,param_struct.num_links+1);
    disp('-')
    disp(x_guess)
    disp('-')
    y_guess = linspace(y0,yn,param_struct.num_links+1);
    disp('-')
    disp(y_guess)
    disp('-')
    coords_guess = zeros(2*(param_struct.num_links-1),1);

    for n = 1:(param_struct.num_links-1)
        coords_guess(2*n-1,1) = x_guess(n+1);
        coords_guess(2*n,1) = y_guess(n+1);
    end
    

    %use gradient descent function to compute
    %the predicted vertex locations
    coords_sol = run_gradient_descent(f_cost,coords_guess,opt_params);
    % coords_sol_y = zeros(5, 1);
    % coords_sol_x = zeros(5, 1);
    % for i = 1:5
    %     coords_sol_y(i) = coords_sol(2*i);
    %     coords_sol_x(i) = coords_sol(2*i - 1);
    % end
    % disp('-------')
    % disp(coords_sol_y)
    % disp('--')
    % disp(coords_sol_x)
    % disp('-------')
    %unpack result and combine with r0 and rn from param_struct
    %to generate list of positions, x_list and y_list
    V_list = [param_struct.r0;coords_sol;param_struct.rn];
    x_list = V_list(1:2:(end-1));
    y_list = V_list(2:2:end);
end