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