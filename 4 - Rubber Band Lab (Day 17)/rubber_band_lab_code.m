% Load Data
data = readtable('RubberBandTemplate.xlsx');

% Get Specified Range of Data
row_range = 1:12;
col_range = 3:6;

% Save data to MATLAB Matrix
data_mat = table2array(data(row_range,col_range));

% Iterate through data of each rubber band (excluding untensed measurement)
for rubber_band = 1:1:(size(data_mat, 1) / 2)
    % Get Mass Hanged from Rubber Band (g)
    mass = data_mat(((rubber_band * 2) - 1), :)';
    % Get Length Rubber Band Stretched (cm)
    length = data_mat((rubber_band * 2), :)';

    % Construct A - [Length Rubber Band Stretched, 1]
    A = [(length ./ 100), ones(size(length, 1), 1)];
    % Construct Y - Force of Weight on Rubber Band (F = ma; a = g = 9.8 m/s^2)
    Y = (mass ./ 1000) .* 9.8;

    % Compute Approximation
    Q = (A' * A) \ (A' * Y);

    % Extract Stiffness (N/m) and Natural Length (m)
    k = Q(1); % Stiffness (N/m)
    l_0 = -1 * Q(2) / Q(1); % Natural Length (m)

    % Store Stiffness (N/m) and Natural Length Values (m)
    stiffness(rubber_band) = k;
    natural_length(rubber_band) = l_0;
end

% Plot Data for Rubber Band #1
rubber_band = 1;
% Get Mass Hanged from Rubber Band (g)
mass = data_mat(((rubber_band * 2) - 1), :)';
% Get Length Rubber Band Stretched (cm)
length = data_mat((rubber_band * 2), :)';

y = linspace(0, 110, 1000); % (g)
x = (((y / 1000) * 9.8 / stiffness(rubber_band)) + natural_length(rubber_band)) * 100; % (cm)

figure
hold on
plot(x, y)
scatter(length, mass)
title('Line of Best Fit for Rubberband #', rubber_band)
xlabel('Length Stretched by Rubber Band (cm)')
ylabel('Mass (g)')
legend("Line of Best Fit", "Measured Points", Location="southoutside")
hold off

% Specific Point for Contour Map
m = 77.7811;
b = -8.9773;

% Plot Contour Map around optimal point
figure
% Define Function
E = @(m, b) (m * (length(1) / 100) + b - (mass(1) / 1000 * 9.8))^2 + (m * (length(2) / 100) + b - (mass(2) / 1000 * 9.8))^2 + (m * (length(3) / 100) + b - (mass(3) / 1000 * 9.8))^2 + (m * (length(4) / 100) + b - (mass(4) / 1000 * 9.8))^2;
x = m - 100:m + 100; y = b - 100:b + 100;
[X,Y] = meshgrid(x,y);
E_num = zeros(201, 201);

% Calculate cost function output at various points
for i = 1:201
    for j = 1:201
        E_num(i,j) = E(X(i,j),Y(i,j));
    end
end

% Create levels for contour plot
min_val = min(min(E_num));
max_val = max(max(E_num));
dVal = sqrt(max_val-min_val);
levels = min_val + linspace(0,dVal,20).^2;

contourf(X, Y, E_num, levels)
hold on

% Plot Optimal Point
plot3(m, b, E(m, b), 'r.', MarkerSize=5)

colormap("parula")
colorbar
xlabel("m")
ylabel("b")
zlabel("E(m, b)")
title('Cost Function for Rubberband #', rubber_band)
legend("E(m, b) Cost Function", "Optimal Point", Location="southoutside")

