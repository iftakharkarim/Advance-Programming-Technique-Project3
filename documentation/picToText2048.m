% makes Tower2048.txt 

% take in a 2048x2048 picture, output it like the other towers
pic = imread('2048picture.jpg');
pic = rgb2gray(pic);

out = zeros(2049,2048);
out(1,1:2) = [2048, 2048];
out(2:end,:) = pic;

fi = fopen('Tower2048.txt','w');

fprintf(fi,'2048 2048\n');

for i = 1:2048
    for j = 1:2048
        fprintf(fi,'%i ',pic(i,j));
    end
    fprintf(fi,'\n');
end

fclose(fi);

% to see the pic to make sure its good
%imshow(mat2gray(out(2:end,:)));