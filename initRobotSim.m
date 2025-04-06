function homeConfig = initRobotSim(urdfPath)
    % initRobotSim Loads the robot from a URDF file and returns its home configuration.
    %
    %   Input:
    %       urdfPath - String with the path to the URDF file.
    %   Output:
    %       homeConfig - The robot's home configuration.
    
    % Load the robot and set its data format.
    robot = importrobot(urdfPath);
    robot.DataFormat = 'row';
    
    % Return the home configuration.
    homeConfig = robot.homeConfiguration;
end
