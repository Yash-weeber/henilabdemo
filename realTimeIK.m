% % Real-Time IK Solver and Simulation Update (MATLAB)
% function realTimeIK()
%     % Initialize robot
%     robot = importrobot('E:\ras\henilabdemo\my_pro600.urdf');
%     fig = figure;
%     ax = show(robot, homeConfiguration(robot));
%     axis([-1 1 -1 1 0 1.5]);
%     view(-100, 90);
%     grid on;
%     hold on;
% 
%     % Initialize IK solver
%     gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs', {'pose'});
%     gik.SolverParameters.MaxIterations = 500;
%     initialGuess = homeConfiguration(robot);
% 
%     % Create pose constraint
%     poseConstraint = constraintPoseTarget('link6');
% 
%     % UDP communication setup (alternative to Python engine)
%     u = udpport('IPV4', 'LocalPort', 12345);
%     flush(u);
% 
%     while true
%         if u.NumBytesAvailable > 0
%             % Read incoming data
%             data = read(u, u.NumBytesAvailable/8, 'double');
%             targetPos = data(1:3)';
% 
%             % Update target transform
%             tform = eul2tform(deg2rad([178, 0, 0]), 'XYZ');
%             tform(1:3, 4) = targetPos;
%             poseConstraint.TargetTransform = tform;
% 
%             % Solve IK
%             [configSoln, ~] = gik(initialGuess, poseConstraint);
% 
%             % Update visualization
%             show(robot, configSoln, 'Parent', ax, 'PreservePlot', false);
%             drawnow;
% 
%             % Update initial guess
%             initialGuess = configSoln;
%         end
%         pause(0.01);
%     end
% 
% function realTimeIK()
%     % Initialize robot
%     robot = importrobot('E:\ras\henilabdemo\my_pro600.urdf');
%     fig = figure;
%     ax = show(robot, homeConfiguration(robot));
%     axis([-1 1 -1 1 0 1.5]);
%     view(-100, 90);
%     grid on;
%     hold on;
% 
%     % Initialize IK solver
%     gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs', {'pose'});
%     gik.SolverParameters.MaxIterations = 500;
%     initialGuess = homeConfiguration(robot);
% 
%     % Create pose constraint
%     poseConstraint = constraintPoseTarget('link6');
%     eulerAngles = [178, 0, 0];
% 
%     % UDP communication
%     u = udpport('IPV4', 'LocalPort', 12345);
%     flush(u);
% 
%     while true
%         if u.NumBytesAvailable > 0
%             % Read and format data
%             data = read(u, u.NumBytesAvailable/8, 'double');
%             targetPos = data(1:3)';
% 
%             % Create target transform
%             tform = eul2tform(deg2rad(eulerAngles), 'XYZ');
%             tform(1:3, 4) = targetPos;
%             poseConstraint.TargetTransform = tform;
% 
%             % Solve IK
%             [configSoln, ~] = gik(initialGuess, poseConstraint);
% 
%             % Update visualization
%             show(robot, configSoln, 'Parent', ax, 'PreservePlot', false);
%             drawnow;
% 
%             % Update initial guess
%             initialGuess = configSoln;
%         end
%         pause(0.01);
%     end
% end
function realTimeIK()
    % Initialize robot with specified DataFormat
    robot = importrobot('E:\ras\henilabdemo\my_pro600.urdf', 'DataFormat', 'struct');
    fig = figure;
    ax = show(robot, homeConfiguration(robot));
    axis([-1 1 -1 1 0 1.5]);
    view(-100, 90);
    grid on;
    hold on;
    
    % Initialize IK solver
    gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs', {'pose'});
    gik.SolverParameters.MaxIterations = 500;
    initialGuess = homeConfiguration(robot);
    
    % Create pose constraint
    poseConstraint = constraintPoseTarget('link6');
    eulerAngles = [178, 0, 0];
    
    % UDP communication
    u = udpport('IPV4', 'LocalPort', 12345);
    flush(u);
    
    while true
        if u.NumBytesAvailable > 0
            % Read and format data
            data = read(u, u.NumBytesAvailable/8, 'double');
            targetPos = data(1:3)';
            
            % Create target transform
            tform = eul2tform(deg2rad(eulerAngles), 'XYZ');
            tform(1:3, 4) = targetPos;
            poseConstraint.TargetTransform = tform;
            
            % Solve IK
            [configSoln, ~] = gik(initialGuess, poseConstraint);
            
            % Update visualization
            show(robot, configSoln, 'Parent', ax, 'PreservePlot', false);
            drawnow;
            
            % Update initial guess
            initialGuess = configSoln;
        end
        pause(0.01);
    end
end

