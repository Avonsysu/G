function color_feature = color_2(im,cell_size,num_bins)

    [M,N,L] = size(im);

    a = round(M/cell_size);
    b = round(N/cell_size);
    
    im = imresize(im,[a*cell_size,b*cell_size]);
    
    c = L;

    f1 = zeros(a,b,48);
    f2 = zeros(a,b,48);
    f3 = zeros(a,b,32);

    sum_I = cell_size * cell_size;
    % color feature
    %RGB
    parfor i = 1:a,      
        for j = 1:b, 

            patch = im((i-1) * cell_size+1 : i*cell_size , (j-1) * cell_size+1 : j * cell_size,:); 
            y1 = zeros(3, 16);

            for k = 1:c,
                binsx = vl_binsearch(linspace(0,255,16+1), double(patch(:,:,k)));
                binsx(binsx == 17) = 16;
                y1(k,:) = vl_binsum(zeros(1,num_bins), ones(size(binsx)), binsx);
                y1(k,:) = y1(k,:)./sum_I;
            end

            f1(i,j,:) = y1(:);

        end
    end

    % YCbCR
    x_ycc = rgb2ycbcr(im);

    parfor i = 1:a,      
        for j = 1:b, 

            patch = x_ycc((i-1) * cell_size+1 : i*cell_size , (j-1) * cell_size+1 : j * cell_size,:); 
            y2 = zeros(3, 16);

            for k = 1:c,
               binsx = vl_binsearch(linspace(0,255,16+1), double(patch(:,:,k)));
               binsx(binsx == 17) = 16;
               y2(k,:) = vl_binsum(zeros(1,num_bins), ones(size(binsx)), binsx);
               y2(k,:) = y2(k,:)./sum_I;

            end

            f2(i,j,:) = y2(:);

        end
    end

    % HS
    x_hsv = rgb2hsv(im);

    parfor i = 1:a,      
        for j = 1:b, 

           patch = x_hsv((i-1)*cell_size + 1 : i*cell_size , (j-1)*cell_size + 1 : j*cell_size,:); 
           y3 = zeros(2, 16);

           for k = 1: 2
               binsx = vl_binsearch(linspace(0,1,16+1), patch(:,:,k));
               binsx(binsx == 17) = 16;
               y3(k,:) = vl_binsum(zeros(1,num_bins), ones(size(binsx)), binsx);
               y3(k,:) = y3(k,:)./sum_I;
           end

           f3(i,j,:) = y3(:);

        end
    end

    color_feature = cat(3,f1,f2,f3);
end