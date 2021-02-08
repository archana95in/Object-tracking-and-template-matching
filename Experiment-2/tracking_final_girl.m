rgbImage = imread('C:\Users\Archana Narayanan\Desktop\DIP\Tracking ALgorithm\Template Matching\Girl\Girl\img\0001.jpg');
comparison_Image = imread('C:\Users\Archana Narayanan\Desktop\DIP\Tracking ALgorithm\Template Matching\Girl\Girl\img\0500.jpg');
tMatcher = vision.TemplateMatcher
[rows ,columns, colours] = size(rgbImage);
subplot(2, 2, 1);
imshow(rgbImage, []);
axis on;
title('First Frame', 'FontSize', 15);
set(gcf, 'units','normalized','outerposition',[0, 0, 1, 1]);

%% Template extraction
templateWidth = 36
templateHeight = 43
%[J, rect] = imcrop(rgbImage);---Use to create template else comment
template_I = imcrop(rgbImage, [51, 26, templateWidth, templateHeight]);
subplot(2, 2, 2);
imshow(template_I, []);
axis on;
title('Template Extracted', 'FontSize', 15);

%% Normalized Cross Correlation
colour_chanel = 1;  
correlation_output = normxcorr2(template_I(:,:,1), comparison_Image(:,:, 1));
subplot(2, 2, 3);
imshow(correlation_output, []);
axis on;
title('Normalized Cross Correlation Output', 'FontSize', 15);
[max_corr, index_val] = max(abs(correlation_output(:)));
[y_peak, x_peak] = ind2sub(size(correlation_output),index_val(1))
corr_offset = [(x_peak-size(template_I,2)) (y_peak-size(template_I,1))]

%%   Plot it on the original image.
subplot(2, 2, 4); 
imshow(comparison_Image);
ROI = [corr_offset(1) corr_offset(2) templateWidth, templateHeight];
[location] = tMatcher(rgb2gray(rgbImage),rgb2gray(template_I));
axis on; 
hold on;

%%  Draw the rectangle for the template box
boxRect = [corr_offset(1), corr_offset(2),43,48]
rectangle('position', boxRect, 'edgecolor', 'm', 'linewidth',4);
title('Last Frame', 'FontSize', 15);


figure;
surf(correlation_output), shading flat


rect1 = [corr_offset(1), corr_offset(2),48,43] 
rect2 = [51, 26,36,43];
intersectionArea = rectint(rect1,rect2);
unionArea = 48*43 + 36*43 - intersectionArea;
overlap = intersectionArea/unionArea * 100;
