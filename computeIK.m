function result = computeIK(targetPose)
    % computeIK computes the inverse kinematics for the custom robot given a target 3D position.
    % targetPose is a 1x3 vector [x y z] in world coordinates.
    
    % Load the robot from the URDF file (adjust the path as needed)
    robot = importrobot('E:\ras\henilabdemo\my_pro600.urdf');
    robot.DataFormat = 'row';  % Use numeric configuration vectors
    
    % Create an IK solver for the robot
    ik = inverseKinematics('RigidBodyTree', robot);
    % Set weights for the position and orientation error (tune as needed)
    weights = [0.25, 0.25, 0.25, 1, 1, 1];
    % Specify the end-effector link name (adjust if different)
    endEffector = 'link6';
    
    % Create a homogeneous transform with translation equal to the targetPose.
    tform = trvec2tform(targetPose);
    
    % Use the robot’s home configuration as an initial guess (a numeric row vector)
    initialguess = robot.homeConfiguration;
    
    % Solve the IK problem (returns a numeric row vector because DataFormat is 'row')
    configSol = ik(endEffector, tform, weights, initialguess);
    
    % Directly use configSol as the joint angles vector
    jointAngles = configSol;
    
    % Optionally update the simulation visualization
    updateRobotVisualization(robot, configSol);
    
    % Return a scalar struct with the field 'jointAngles'
    result.jointAngles = jointAngles;
end

function updateRobotVisualization(robot, config)
    % updateRobotVisualization updates the robot's visualization in a MATLAB figure.
    % This function can be called repeatedly to animate the robot.
    
    persistent figHandle;
    if isempty(figHandle) || ~isvalid(figHandle)
        figHandle = figure;
        show(robot, config);
        view(-100, 90);
        axis([-1 1 -1 1 0 1.5]);
        grid on;
        hold on;
    else
        show(robot, config, 'PreservePlot', true);
        drawnow;
    end
end
