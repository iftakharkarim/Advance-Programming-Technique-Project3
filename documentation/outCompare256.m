close all

% uses text parser program and cuts out dimensions at top
% rename this file to the codes output
ar = import256('recentcudaoutput.txt');
ar = ar(2:end,:);

% first value doesn't come with imag, need to put it there
fl = [ar(1), 0, ar(1,2:end-1)];
ar(1,:) = fl;

% turns into complex pairs
re = ar(:,1:2:end);
im = ar(:,2:2:end);
c = complex(re,im);

imshow(log(abs(c)),[])
title('Our output');

% parses and fft2s the original file
t256 = fopen('Tower256.txt','r');
a = fscanf(t256,'%i');
dim = a(1);
size = a(1);
a = a(3:end);
b = zeros(dim,dim);
for i = 1:(dim*dim)
   b(i) = a(i); 
end
b = b';
b2 = fft2(b);

figure
imshow(log(abs(b2)),[]);
title('Matlabs output');
fclose(t256);

diff = c-b2;
figure
imshow(abs(diff),[]);
title('Difference between outputs');

figure
imshow(log(abs(diff)),[]);
title('log of difference, exaggerates differences');








