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