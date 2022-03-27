clear;
close all;
file = "hands";
if file == "ellipses"
    data = load('../data/ellipses2D.mat');
    pointSets = data.pointSets;
else
    data = load('../data/hands2D.mat');
    pointSets = data.shapes;
end
numIters = 1000;
z_init = rand(size(pointSets(:,:,1)));
z_init = z_init - reshape(mean(z_init,2),[],1);
z_init = z_init / norm(z_init,'fro');

%% Plot pointsets
figure("Name","Initial pointsets");
pause(1);
plotPointSets(pointSets)

%% Code11
z = z_init;
for i = 1:numIters
    [z,~] = Code11(z,pointSets);
end

[~,alignedPointSets] = Code11(z,pointSets);

makePlots(z, alignedPointSets, "Code11");

%% Code22
z = z_init;
for i = 1:numIters
    [z,~] = Code22(z,pointSets);
end

[~,alignedPointSets] = Code22(z,pointSets);

makePlots(z, alignedPointSets, "Code22");


