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
    [x_list,y_list] = generate_shape_prediction(param_struct);

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