function color_feature = color(I)

num_bins =16;
c = 3;
y1 = zeros(3, 16);
y2 = zeros(3, 16);
y3 = zeros(2, 16);
sum_I=size(I,1)*size(I,2);
% color feature
% RGB
for i = 1: c
        binsx = vl_binsearch(linspace(0,255,16+1), double(I(:,:,i)));
        binsx(binsx == 17) = 16;
        y1(i,:) = vl_binsum(zeros(1,num_bins), ones(size(binsx)), binsx);
        y1(i,:) = y1(i,:)./sum_I;
end

% YCbCR
x_ycc = rgb2ycbcr(I);
for i = 1: c
        binsx = vl_binsearch(linspace(0,255,16+1), double(x_ycc(:,:,i)));
        binsx(binsx == 17) = 16;
        y2(i,:) = vl_binsum(zeros(1,num_bins), ones(size(binsx)), binsx);
        y2(i,:) = y2(i,:)./sum_I;
end
% HS
x_hsv = rgb2hsv(I);
for i = 1: 2
        binsx = vl_binsearch(linspace(0,1,16+1), x_hsv(:,:,i));
        binsx(binsx == 17) = 16;
        y3(i,:) = vl_binsum(zeros(1,num_bins), ones(size(binsx)), binsx);
        y3(i,:) = y3(i,:)./sum_I;
end

color_feature = [y1(:); y2(:); y3(:)];
end