%gradient descent template.
function gradient_descent_()
    %set your value of alpha here
    %alpha =

    %set your initial value of x and y here
    %x =
    %y =

    %define two lists to store visited (x,y) coordinates
    xlist = []; ylist = [];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %your gradient descent code here!
    %%%%%%%%%%%%%%%%%%%%%%%%%

    %define a range of x vals and y vals
    x_range = linspace(-3,1,251);
    y_range = linspace(-3,1,251);
    %define a grid of x and y vals
    [xgrid,ygrid] = meshgrid(x_range,y_range);
    %evaluate the test function at each grid point
    fgrid = test_func01(xgrid,ygrid);

    %creates a new figure
    figure();
    %sets figure so calls to plot do not erase previous drawings
    hold on;
    %set x and y axis scaling to be equal
    axis equal;
    %create a colored contour plot of the test function
    contourf(xgrid,ygrid,fgrid,-8 + [0:.2:3].^2);
    %add a colorbar to the side of the contour plot indicating level vals
    colorbar;

    %%%%%%%%%%%%%%%%%%%%%%%%%
    %your plotting code here!
    %%%%%%%%%%%%%%%%%%%%%%%%%

    %Create axis labels and title
    xlabel('x'); ylabel('y'); title('Gradient Descent Visualization');
end

%definition of the test function
function f = test_func01(x,y)
    f = x.^2+y.^2-x.*y+2*x+2*y-4;
end

%definition of the gradient of the test function
function gf = test_gradient01(x,y)
    gf = [2*x-y+2; 2*y-x+2];
end