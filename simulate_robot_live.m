function simulate_robot_live()
    % simulate_robot_live loads the robot model, displays it, and then continuously
    % receives live target positions (here via user input) to update the robot's pose.
    
    % Load the robot (adjust the URDF file path as needed)
    robot = importrobot('E:\ras\henilabdemo\my_pro600.urdf');
    robot.DataFormat = 'row';
    
    % Create a figure for simulation
    figure;
    % Show the robot at its home configuration
    config0 = robot.homeConfiguration;
    show(robot, config0);
    view(-100, 90);
    axis([-1 1 -1 1 0 1.5]);
    grid on;
    hold on;
    
    % Initialize the IK solver
    ik = inverseKinematics('RigidBodyTree', robot);
    weights = [0.25, 0.25, 0.25, 1, 1, 1];
    endEffector = 'link6';
    currentConfig = config0;
    
    % Simulation loop: here we prompt the user for a target position.
    % In a real application, you might replace this with live data from a sensor.
    while true
        targetPose = get_live_target();
        if isempty(targetPose)
            disp('Exiting simulation.');
            break;
        end
        
        % Create target transform with fixed orientation
        tform = trvec2tform(targetPose);
        [configSol, solInfo] = ik(endEffector, tform, weights, currentConfig);
        
        % Animate smooth movement from current to new configuration
        nSteps = 30;
        for i = 0:nSteps
            alpha = i / nSteps;
            interpConfig = currentConfig;
            for j = 1:numel(currentConfig)
                interpConfig(j).JointPosition = (1 - alpha) * currentConfig(j).JointPosition + alpha * configSol(j).JointPosition;
            end
            show(robot, interpConfig, 'PreservePlot', false);
            drawnow;
            pause(0.02);
        end
        currentConfig = configSol;
    end
end

function targetPose = get_live_target()
    % get_live_target prompts the user for a target [x y z] coordinate.
    % Replace or extend this function to obtain live data from a file, sensor, or network.
    
    prompt = 'Enter target X Y Z coordinates separated by spaces (or press Enter to exit): ';
    userInput = input(prompt, 's');
    if isempty(userInput)
        targetPose = [];
    else
        nums = str2num(userInput);  %#ok<ST2NM>
        if numel(nums) == 3
            targetPose = nums;
        else
            disp('Invalid input. Using default target [0 0 0.5].');
            targetPose = [0 0 0.5];
        end
    end
end
