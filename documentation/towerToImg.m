close all
format long;
file = fopen('Tower256.txt','r');
cc = fopen('formatMatched.txt','w');
com = fopen('comma.txt','w');

a = fscanf(file,'%i');
dim = a(1);
size = a(1);
a = a(3:end);
b = zeros(dim,dim);

for i = 1:(dim*dim)
   b(i) = a(i); 
    
end

b = b';

I = mat2gray(b);
%imshow(I)
%figure
b2 = fft2(b);
% outputs comma delimited fft2 to file
dlmwrite('outFile.txt',b2);
dlmwrite('outFilefftshift.txt',fftshift(b2));

breal = real(b2);
bimag = imag(b2);
fprintf(cc,'%d %d\n',size,size);
for i = 1:size
   for j = 1:size
      fprintf(cc,'(%.4f,%.4f) ', breal(i,j), bimag(i,j));
      fprintf(com, '%.4f %.4f ',breal(i,j), bimag(i,j));
   end
   fprintf(cc,'\n');
   fprintf(com,'\n');
end

imshow(log(abs(b2)),[])
title('log(abs(fft2))');
% highest freq above is in the middle

%imshow(fftshift(log(abs(b2))),[])
% now the lowest frequency is in the middle, highest are at the edges

fclose(file);
fclose(cc);
fclose(com);



