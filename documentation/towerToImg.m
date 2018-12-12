close all
file = fopen('Tower256.txt');

a = fscanf(file,'%i');
dim = a(1);
a = a(3:end);
b = zeros(dim,dim);

for i = 1:(dim*dim)
   b(i) = a(i); 
    
end

b = b;

I = mat2gray(b);
imshow(I)
figure
b2 = fft2(b);
% outputs comma delimited fft2 to file
dlmwrite('outFile.txt',b2);

imshow(log(abs(b2)),[])
% highest freq above is in the middle

imshow(fftshift(log(abs(b2))),[])
% now the lowest frequency is in the middle, highest are at the edges