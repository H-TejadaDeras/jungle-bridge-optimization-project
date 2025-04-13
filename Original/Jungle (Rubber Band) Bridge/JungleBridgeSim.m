%% Main function that runs the jungle bridge simulation code
function JungleBridgeSim()
    %% Import Data
    data = readtable("JungleBridgeTemplate.xlsx");
    
    % Get Specified Range of Data
    row_range = 1:7;
    col_range = 2:3;
    
    % Save data to MATLAB Matrix
    data_mat = table2array(data(row_range,col_range));
    
    % Save Mass Data to MATLAB Matrix as Column Vector
    mass = table2array(data(1:5, 6))';
    % Convert Mass from g to kg
    mass = mass ./ 1000;

    % Save Measured Coordinates
    measured_coords = data_mat;
    % Convert Measured Coordinates from cm to m
    measured_coords = measured_coords ./ 100;

    %% initialize the system parameters
    %which contains parameters describing behavior/measurements of bridge
    % param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
    % param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
    % param_struct.num_links: number of rubber bands in bridge
    % param_struct.k_list = [k_1;...;k_n]: list of stiffnesses
    % param_struct.l0_list = [l0_1;...;l0_n]: list of natural lengths
    % param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
    % param_struct.g = 9.8 m/sec^2: gravitational acceleration
    param_struct = struct();
    param_struct.r0 = measured_coords(1, :)'; % Transposed so it is saved as a column vector; divided by 100 so it is in m
    param_struct.rn = measured_coords(7, :)'; % Transposed so it is saved as a column vector; divided by 100 so it is in m
    param_struct.num_links = size(measured_coords, 1) - 1;
    param_struct.k_list = load("stiffness.mat"); % stiffnesses (N/m)
    param_struct.l0_list = load("natural_length.mat"); % natural lengths (meters)
    param_struct.m_list = mass; % mass list (kg)
    param_struct.g = 9.8; % gravitational acceleration (m/s^2)

    %% compute the predicted bridge shape
    % Run first gradient descent iteration
    [x_list,y_list] = generate_shape_prediction(param_struct);
    % Run all other iterations
    % for i = 2:5
    %     % Create Previous Coordinates Variable
    %     prev_coords_x = x_list(:, i - 1);
    %     prev_coords_y = y_list(:, i - 1);
    %     for n = 1:(param_struct.num_links-1)
    %         prev_coords(2*n-1,1) = prev_coords_x(n+1);
    %         prev_coords(2*n,1) = prev_coords_y(n+1);
    %     end
    %     % disp(x_list(:, i - 1))
    %     % disp(y_list(:, i - 1))
    %     % disp(prev_coords)
    %     % disp('----------------')
    %     [x_list(:, i),y_list(:, i)] = generate_shape_prediction_v2(param_struct, prev_coords);
    % end
    % disp(y_list)
    % disp(y_list)

    %% generate a plot comparing the predicted and measured bridge shape
    figure()
    hold on
    % Plot Measured Bridge Data
    plot(measured_coords(:, 1), measured_coords(:, 2), "k", DisplayName="Measured Bridge Shape")
    scatter(measured_coords(:, 1), measured_coords(:, 2), "ko", DisplayName="Measured Bridge Shape Points")

    % Plot Gradient Descent Output
    plot(x_list(:, 1), y_list(:, 1), "r", DisplayName='Predicted Bridge Shape')
    scatter(x_list(:, 1), y_list(:, 1), "ro", DisplayName="Predicted Bridge Shape Points")
    % plot(x_list(:, 2), y_list(:, 2), "r", DisplayName='Predicted Bridge Shape')
    % scatter(x_list(:, 2), y_list(:, 2), "ro", DisplayName="Predicted Bridge Shape Points")

    % Legend and other things
    xlabel("x (m)")
    ylabel("y (m)")
    legend(Location="southoutside")
    title("Predicted vs. Measured Jungle Bridge Shape")
end

%% compute the potential energy of a SINGLE rubber band
%INPUTS
%(xA,yA): coordinates of left end of rubber band
%(xB,yB): coordinates of right end of rubber band
%k: stiffness of rubber band
%l0: natural length of rubber band
%OUTPUTS:
%U_RB_i: potential energy of rubber band
function U_RB_i = single_RB_potential_func(xA,yA,xB,yB,k,l0)
    %compute stretched length of rubber band
    l = norm([xA, yA] - [xB, yB]);
    %compute potential energy (remember to use max function!)
    % U_RB_i = max((k * l^2) / 2, 0);
    U_RB_i = 0.5 * k * max(0, (l - l0))^2; % Equation above was our old version, it was incorrect.
end

%% compute the total potential energy of all rubber bands in bridge
%INPUTS:
%coords: vector of vertex positions from i=1 to i=(n-1)
% [x_1;y_1;...;x_(n-1),y_(n-1)]
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of rubber bands in bridge
% param_struct.k_list = [k_1;...;k_n]: list of stiffnesses
% param_struct.l0_list = [l0_1;...;l0_n]: list of natural lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
%OUTPUTS:
%U_RB_total: total potential energy of rubber bands in bridge
function U_RB_total = total_RB_potential_func(coords,param_struct)
    %initialize total spring potential energy
    U_RB_total = 0;

    %add the first and last vertex positions to the coordinate list
    coords_full = [param_struct.r0;coords;param_struct.rn];

    %iterate through each rubber band link
    for i = 1:param_struct.num_links
        %extract the ith stiffness and natural length
        l0 = struct2array(param_struct.l0_list);
        k = struct2array(param_struct.k_list);

        %extract the coordinates of the rubber band ends
        xA = coords_full(2 * i - 1);
        yA = coords_full(2 * i);
        xB = coords_full(2 * i + 1);
        yB = coords_full(2 * 1 + 2);

        %compute the potential energy of the ith rubber band
        U_RB_i = single_RB_potential_func(xA, yA, xB, yB, k(i), l0(i));

        %add the ith potential to the total
        U_RB_total = U_RB_total + U_RB_i;
    end
end

%% compute the gravitational potential energy of a SINGLE weight
%INPUTS
%(x,y): coordinates of a the current vertex
%m: mass of the weight
%g: gravitational acceleration
%OUTPUTS:
%U_g_i: gravitational potential energy of weight
function U_g_i = single_G_potential_func(x,y,m,g)
    %compute gravitational potential energy of weight
    U_g_i = m * g * y;
end

%% compute the total gravitational potential energy
%of all weights in bridge
%INPUTS:
%coords: vector of vertex positions from i=1 to i=(n-1)
% [x_1;y_1;...;x_(n-1),y_(n-1)]
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of rubber bands in bridge
% param_struct.k_list = [k_1;...;k_n]: list of stiffnesses
% param_struct.l0_list = [l0_1;...;l0_n]: list of natural lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
%OUTPUTS:
%U_g_total: total gravitational potential energy
function U_g_total = total_G_potential_func(coords,param_struct)
    %initialize total gravitational potential energy
    U_g_total = 0;

    %iterate through each weight
    for i = 1:(param_struct.num_links-1)
        %extract the coordinates of the ith vertex
        x = coords(2 * i - 1);
        y = coords(2 * i);

        %extract the ith mass and the gravitational acceleration
        m = param_struct.m_list(i);
        g = param_struct.g;

        %compute the gravitational potential energy of the ith mass
        U_g_i = single_G_potential_func(x,y,m,g);

        %add the ith potential to the total
        U_g_total = U_g_total + U_g_i;
    end
end

%% compute the total potential energy of the bridge
%INPUTS:
%coords: vector of vertex positions from i=1 to i=(n-1)
% [x_1;y_1;...;x_(n-1),y_(n-1)]
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of rubber bands in bridge
% param_struct.k_list = [k_1;...;k_n]: list of stiffnesses
% param_struct.l0_list = [l0_1;...;l0_n]: list of natural lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
%OUTPUTS:
%U_total: total potential energy of the bridge
function U_total = total_potential_func(coords,param_struct)
    %compute the gravitational potential energy of the weights
    U_g_total = total_G_potential_func(coords, param_struct);
    %compute the spring potential energy of the rubber bands
    U_RB_total = total_RB_potential_func(coords, param_struct);
    %sum the two results
    U_total = U_g_total + U_RB_total;
end

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

%% Run Gradient Descent
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

%% use gradient descent to predict the shape of the bridge - V2
% Used for second and so on iterations
%INPUTS:
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of rubber bands in bridge
% param_struct.k_list = [k_1;...;k_n]: list of stiffnesses
% param_struct.l0_list = [l0_1;...;l0_n]: list of natural lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
% prev_coords: previous coordinates outputted from running gradient descent
%OUTPUTS:
%x_list = [x_0;x_1;...;x_n]: x coordinates of predicted vertex positions
%y_list = [y_0;y_1;...;y_n]: x coordinates of predicted vertex positions
function [x_list,y_list] = generate_shape_prediction_v2(param_struct, prev_coords)
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

    % This section is replaced with the previous coordinates from running
    % the function of gradient descent

    %use gradient descent function to compute
    %the predicted vertex locations
    coords_sol = run_gradient_descent(f_cost,prev_coords,opt_params);

    %unpack result and combine with r0 and rn from param_struct
    %to generate list of positions, x_list and y_list
    V_list = [param_struct.r0;coords_sol;param_struct.rn];
    x_list = V_list(1:2:(end-1));
    y_list = V_list(2:2:end);
end