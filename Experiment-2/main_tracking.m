rgbImage = imread('C:\Users\Archana Narayanan\Desktop\DIP\Tracking ALgorithm\Template Matching\Girl\Girl\img\0001.jpg');
comparison_Image = imread('C:\Users\Archana Narayanan\Desktop\DIP\Tracking ALgorithm\Template Matching\Girl\Girl\img\0500.jpg');
tMatcher = vision.TemplateMatcher
[rows ,columns, colours] = size(rgbImage);
subplot(2, 2, 1);
imshow(rgbImage, []);
axis on;
title('Frame 1', 'FontSize', 10);
set(gcf, 'units','normalized','outerposition',[0, 0, 1, 1]);

%% Template extraction
templateWidth = 64
templateHeight = 66
%[J, rect] = imcrop(rgbImage);
template_I = imcrop(rgbImage, [43, 20, templateWidth, templateHeight]);
subplot(2, 2, 2);
imshow(template_I, []);
axis on;
title('Template Extracted', 'FontSize', 10);

%% Normalized Cross Correlation
colour_chanel = 1;  
correlation_output = normxcorr2(template_I(:,:,1), comparison_Image(:,:, 1));
subplot(2, 2, 3);
imshow(correlation_output, []);
axis on;
title('Normalized Cross Correlation', 'FontSize', 10);
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

%%  Calculate the rectangle for the template box.  Rect = [xLeft, yTop, widthInColumns, heightInRows]
boxRect = [corr_offset(1), corr_offset(2),64,66]
rectangle('position', boxRect, 'edgecolor', 'b', 'linewidth',4);
title('Last_frame', 'FontSize', 10);

