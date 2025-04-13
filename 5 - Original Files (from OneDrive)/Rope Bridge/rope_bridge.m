function rope_bridge()
    %% Import Data
    data = readtable("RopeBridgeTemplate.xlsx");
    
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

    % Save Natural Lengths
    natural_lengths = table2array(data(1:6, 9))';
    
    % Convert Natural Lengths from cm to m
    natural_lengths = natural_lengths ./ 100;
    disp(natural_lengths)
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
    param_struct.k_list = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10]; % stiffnesses (N/m)
    param_struct.l0_list = natural_lengths; % natural lengths (meters)
    param_struct.m_list = mass; % mass list (kg)
    param_struct.g = 9.8; % gravitational acceleration (m/s^2)

    %% compute the predicted bridge shape
    % Run first gradient descent iteration
    [x_list,y_list] = generate_shape_prediction_fmincon(param_struct);
    % disp(y_list)
    % Run all other iterations
    % for i = 2:5
    %     % Create Previous Coordinates Variable
    %     prev_coords_x = x_list(:, i - 1);
    %     prev_coords_y = y_list(:, i - 1);
    %     for n = 1:(param_struct.num_links-1)
    %         prev_coords(2*n-1,1) = prev_coords_x(n+1);
    %         prev_coords(2*n,1) = prev_coords_y(n+1);
    %     end
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
    % disp(y_list)
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

%% computes the distance constraint error for a pair of vertices
%INPUTS:
%(xA, yA): coordinates of first vertex
%(xB, yB): coordinates of second vertex
%l_max: maximum allowable distance between two vertices
%OUTPUTS:
%e_len: constraint error.
% e_len<=0 when constraint is satisfied
% e_len>0 when constraint is violated
function e_len = single_string_error_func(xA,yA,xB,yB,l_max)
    % Calculate Deviation from Constraint
    e_len = sqrt((xB - xA)^2 + (yB - yA)^2) - l_max;
end

%% evaluates the distance constraint error across all links
%INPUTS:
%coords: vector of vertex positions from i=1 to i=(n-1)
% [x_1;y_1;...;x_(n-1),y_(n-1)]
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of links in bridge
% param_struct.l0_list = [l0_1;...;l0_n]: list of link lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
%OUTPUTS:
%e_val = [e_len1; ... ; e_len_n]: the vector of distance constraint errors
%dummy = []: empty vector used to satisfy fmincon syntax
function [e_vec,dummy] = bridge_error_func(coords,param_struct)
    %initialize error vector
    e_vec = zeros(param_struct.num_links,1);

    %initialize dummy output for fmincon equality constraints
    dummy = [];

    %add the first and last vertex positions to the coordinate list
    coords = [param_struct.r0;coords;param_struct.rn];
    
    %iterate through each rubber band link
    for i = 1:param_struct.num_links
        %extract the ith segment length
        % class(param_struct.l0_list)
        l_max = param_struct.l0_list;

        %extract the coordinates of the string ends
        xA = coords(2 * (i + 1) - 1);
        yA = coords(2 * (i + 1));
        xB = coords(2 * i - 1);
        yB = coords(2 * i);

        %evaluate the ith distance constraint
        e_vec(i) = single_string_error_func(xA,yA,xB,yB,l_max(i));
    end
end

%% use fmincon to predict the shape of the string bridge
%INPUTS:
%param_struct: struct containing parameters of the bridge
% param_struct.r0 = [x_0;y_0]: coordinates of leftmost vertex
% param_struct.rn = [x_n;y_n]: coordinates of rightmost vertex
% param_struct.num_links: number of links in bridge
% param_struct.l0_list = [l_1;...;l_n]: list of link lengths
% param_struct.m_list = [m_1;...;m_(n-1)]: list of weight masses
% param_struct.g = 9.8 m/sec^2: gravitational acceleration
%OUTPUTS:
%x_list = [x_0;x_1;...;x_n]: x coordinates of predicted vertex positions
%y_list = [y_0;y_1;...;y_n]: x coordinates of predicted vertex positions
function [x_list,y_list] = generate_shape_prediction_fmincon(param_struct)
    %generate an initial guess for the coordinate locations
    %coords_guess = [x_1;y_1;...;x_(n-1);y_(n-1)]

    x0 = param_struct.r0(1);
    y0 = param_struct.r0(2);
    xn = param_struct.rn(1);
    yn = param_struct.rn(2);
    x_guess = linspace(x0,xn,param_struct.num_links+1);
    % y_guess = linspace(y0,yn,param_struct.num_links+1);
    y_guess = [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001];
    disp(y_guess)

    coords_guess = zeros(2*(param_struct.num_links-1),1);
    for n = 1:(param_struct.num_links-1)
        coords_guess(2*n-1,1) = x_guess(n+1);
        coords_guess(2*n,1) = y_guess(n+1);
    end

    %use anonymous function syntax to define the cost func
    %define cost func as the gravitational potential energy function
    %using the current values in param_struct
    f_cost = @(V_in) total_G_potential_func(V_in,param_struct);

    %use anonymous function syntax to define the constraint func
    %define cost func as the distance constraint function
    %using the current values in param_struct
    f_cstr = @(V_in) bridge_error_func(V_in,param_struct);

    %use fmincon to compute the predicted vertex locations
    coords_sol = fmincon(f_cost, coords_guess, [], [], [], [], [], [], f_cstr);

    %unpack result and combine with r0 and rn from param_struct
    %to generate list of positions, x_list and y_list
    V_list = [param_struct.r0;coords_sol;param_struct.rn];
    x_list = V_list(1:2:(end-1));
    y_list = V_list(2:2:end);
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

