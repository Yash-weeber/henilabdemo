function new_config = update_robot_sim(target_position, current_config)
% update_robot_sim Animates the robot from the current configuration to
% the configuration computed from the target end-effector position.
%
%   target_position: a 1x3 vector [x, y, z] in the robot workspace.
%   current_config: current robot configuration (array of joint structs).
%
%   new_config: resulting robot configuration after moving toward the target.
%
% This function uses a persistent robot model and IK solver so that the
% robot is loaded only once. The end-effector orientation is fixed.

    persistent robot gik nSteps frameRate

    % On the first call, load the robot and initialize the IK solver.
    if isempty(robot)
        % Adjust the path to your URDF file as needed.
        robot = importrobot('E:\ras\henilabdemo\my_pro600.urdf');
        robot.DataFormat = 'row';
        
        % Set up the generalized inverse kinematics solver.
        gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs', {'pose'});
        gik.SolverParameters.MaxIterations = 500;
        
        % Animation parameters.
        frameRate = 30;           % frames per second
        totalDuration = 0.5;      % seconds for each movement
        nSteps = totalDuration * frameRate;
        
        % Initialize display.
        figure(1);
        config = homeConfiguration(robot);
        show(robot, config);
        view(-100, 90);
        axis([-1 1 -1 1 0 1.5]);
        grid on;
        hold on;
        camzoom(1.20);
        light('Style', 'infinite', 'Position', [0.2, 0.7, 0.9], 'Color', [1, 1, 1]);
        light('Style', 'local', 'Position', [-0.4, -0.7, -0.9], 'Color', [1, 1, 1]);
    end

    % Set a fixed end-effector orientation.
    % Here we use Euler angles (in degrees) converted to a transformation.
    eulerAngles = [178, 0, 0];  % Adjust if needed.
    eulerAnglesRad = deg2rad(eulerAngles);
    tform = eul2tform(eulerAnglesRad, 'XYZ');
    
    % Set the position part of the transformation.
    tform(1:3, 4) = target_position(:);
    
    % Create a pose target constraint for the end-effector (assumed to be 'link6').
    poseConstraint = constraintPoseTarget('link6');
    poseConstraint.TargetTransform = tform;
    
    % Use the current configuration as the initial guess.
    initialGuess = current_config;
    
    % Compute the new configuration with IK.
    [configSoln, ~] = gik(initialGuess, poseConstraint);
    new_config = configSoln;
    
    % Animate movement from current_config to new_config.
    for i = 0:nSteps
        alpha = i / nSteps;
        interp_config = current_config;
        for j = 1:length(current_config)
            interp_config(j).JointPosition = (1 - alpha) * current_config(j).JointPosition + alpha * new_config(j).JointPosition;
        end
        show(robot, interp_config, 'PreservePlot', false);
        drawnow;
        pause(1 / frameRate);
    end
end

