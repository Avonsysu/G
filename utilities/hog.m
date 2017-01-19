function hog_feature = hog(I)
   I = im2single(I);
   cellSize =8 ;
   hog_feature = vl_hog(I, cellSize) ;
   hog_feature=hog_feature(:);
end

