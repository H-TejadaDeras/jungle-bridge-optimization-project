clear all
close all

% Set up some example values
sample_length = [71, 63, 45, 89];
sample_mass = [22, 18, 16, 25];

figure(); hold on
plot(sample_length, sample_mass, 'o')

A = [sample_length', ones(4,1)];
Y = sample_mass';

% Calculate optimal m and b
p = (A'*A) \ A'*Y
plot(sample_length, p(1).*sample_length + p(2))

xlabel('Length')
ylabel('Mass')
legend('Data', 'Derived Fit', 'Location','Southeast')

% Define error function
X = sample_length';
Y = sample_mass';
DY = @(m,b) norm(m.*X + b.*ones(4,1) - Y);
E = @(m,b) DY(m,b)^2;
minimum = E(p(1),p(2));
disp(strcat('Minimum Error:', num2str(minimum)))

figure();
% This is the important part: set up axis scaling
DX  = .004;
DY = .2;

fsurf(E,[p(1)-DX p(1)+DX p(2)-DY p(2)+DY]); hold on
plot3(p(1), p(2), E(p(1),p(2)), 'ro')

figure();

x = linspace(p(1)-DX,p(1)+DX, 1000);
y = linspace(p(2)-DY, p(2)+DY, 1000);
[X,Y] = meshgrid(x,y);

E_num = zeros(1000, 1000);
for i = 1:1000
    for j = 1:1000
        E_num(i,j) = E(X(i,j),Y(i,j));
    end
end

min_val = min(min(E_num));
max_val = max(max(E_num));

dVal = sqrt(max_val-min_val);

levels = min_val + linspace(0,dVal,15).^2;

contourf(X,Y,E_num, levels); hold on
colormap("parula")
plot(p(1), p(2), 'ro')