clc
clear all
close all
% Give image directory and extension
imPath = 'C:\Users\Archana Narayanan\Desktop\DIP\Tracking ALgorithm\Template Matching\Biker\Biker\img'; imExt = 'jpg';
filearray = dir([imPath filesep '*.' imExt]); % get all files in the directory
num_images = size(filearray,1); % get the number of images

% Get image parameters
image_name = [imPath filesep filearray(1).name]; % get image name
I = imread(image_name); 
video_w = size(I,2);
video_h = size(I,1);
image_sequence = zeros(video_h, video_w, num_images);

for mean=1:num_images
    image_name = [imPath filesep filearray(mean).name]; % get image name
    image_sequence(:,:,mean) = (rgb2gray(imread(image_name))); % load image
end


%%  ---------------- BACKGROUND SUBTRACTION--------------------------------------------------------------------------------
for i = 1:video_h
    for j = 1:video_w
        % Taking median of frame to frame
        I = median(image_sequence(i,j,:));
        background(i,j) = I;    % background Image
    end
end

% Moving object in every frame with threshold
for m = 1:size(image_sequence,3)    
    M = image_sequence(:,:,m);               % Current Frame
    foreground = M- background;        
    fore_thresh = foreground>25; 
end
%------------------------------------------------------------------
% Initializing mean, variance and alpha 
mean = image_sequence(:,:,1);                         
sigma = 1000*ones(video_h,video_w); 
alpha = 0.01;
final = zeros(video_h,video_w);
figure;

for m = 1: size(image_sequence,3)
    M = image_sequence(:,:,m);   % Current Frame
    foreground = M - background;     
    
    %Mean
    mean(:,:,m+1) = alpha*image_sequence(:,:,m)+(1-alpha)*mean(:,:,m);
    d = abs(image_sequence(:,:,m)-mean(:,:,m+1));
    %Variance
    sigma(:,:,m+1) = (d.^2)*alpha+(1-alpha)*(sigma(:,:,m));

    final = d>2*sqrt(sigma(:,:,m+1));   % Computed Foreground
    subplot(221),imshow(background,[]), title('Background Image');
    subplot(222),imshow(M,[]), title('Current Image');
    subplot(223),imshow(final,[]), title('Running average gaussian');
    subplot(224),imshow(foreground,[]), title('Object Detection');
    drawnow;
end
%%=================== Eigen background ===========================  
% Initialising Variables
N = 30; k = 15; T = 25;
% Reshaping Images
for i = 1:size(image_sequence,3)
   x(:,i) = reshape(image_sequence(:,:,i), [], 1);
end
% The Mean Image
mean = 1/N*sum(x(:,1:N),2);
% Compute mean-normalized image vectors
normalised_mean = x - repmat(mean,1,size(x,2));
% SVD of the matrix X
[U, S, V] = svd( normalised_mean, 'econ');

% Keep the k principal components of U
eigen_background = U(:,1:k);
figure;
 Es = cell(100,1);
for i = 1:size(image_sequence,3)
    img_cons = x(:,i);
    p = eigen_background' * (img_cons - mean);
    Sub_Image = eigen_background * p + mean;

    mask = abs(Sub_Image - img_cons) > T;
    Image_Reshaped = reshape(img_cons .* mask, video_h, video_w);
    
    %Get the bounding box
    Final_Image = im2bw(Image_Reshaped);
    %% Morphological Operations
    Final_Image = imerode(Final_Image, strel('rectangle', [2 2]));
    Final_Image = imdilate(Final_Image, strel('rectangle', [5 5]));
    bounding_box = regionprops(Final_Image, 'BoundingBox', 'Area');
    area_val = cat(1, bounding_box.Area);
    if(area_val)
        [~,ind] = max(area_val);
        B_box = bounding_box(ind).BoundingBox;
        Es{i} = B_box;
    end
    
    imshow(image_sequence(:,:,i),[]), title('Tracking Algorithm Result');
    if(area_val)
        hold on;
        rectangle('Position', B_box,'EdgeColor','m');
        hold off;
    end
    drawnow;
end

fid = fopen('groundtruth_rect.txt', 'rt');
tline = fgetl(fid);
headers = strsplit(tline, ',');     %a cell array of strings
datacell = textscan(fid, '%f%f%f%f', 'Delimiter',',', 'CollectOutput', 1);
fclose(fid);
datavalues = datacell{1};    %as a numeric array

% bounding box values obtained from Es{i}
fid = fopen('girl.txt', 'rt');
tline = fgetl(fid);
headers = strsplit(tline, ',');     %a cell array of strings
datacell_predicted = textscan(fid, '%f%f%f%f', 'Delimiter',',', 'CollectOutput', 1);
fclose(fid);
predicted_values = datacell_predicted{1};    %as a numeric array

%% IoU calculation
Overlap_value = cell(100,1);
for i = 1: 499
    rect1 = datavalues(i,:,:,:);
    rect2 = predicted_values(i,:,:,:);
    intersectionArea = rectint(rect1,rect2);
    unionArea = datavalues(i,3)* predicted_values(i,4) + predicted_values(i,3)* predicted_values(i,4) - intersectionArea;
    overlap = intersectionArea/unionArea * 100;    
    Overlap_value{i} = overlap;
    overlapRatio = bboxOverlapRatio(rect1,rect2);
end
plot_value = cell2mat(Overlap_value);
%% Plot IoU Value
figure();
x_plot = 1:1:499;
plot(x_plot,plot_value,'-s','MarkerSize',3,...
    'LineWidth',2,...
    'MarkerEdgeColor','red',...
    'MarkerFaceColor',[1 .4 .4])
title('Value of Intersection Over Union in % for all frames')
xlabel('Frame Number') 
ylabel('IoU value in %') 
    

