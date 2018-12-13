close all

% uses text parser program and cuts out dimensions at top
% rename this file to the codes output
ar = import256('MpiOutput256.txt');
ar = ar(2:end,:);

% first value doesn't come with imag, need to put it there
fl = [ar(1), 0, ar(1,2:end-1)];
ar(1,:) = fl;

% turns into complex pairs
re = ar(:,1:2:end);
im = ar(:,2:2:end);
c = complex(re,im);

figure
sp = subplot(2,2,1);
imshow(log(abs(c)),[])
title('Our output abs logged');

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
% figure out this transposing issue
b = b';
b2 = fft2(b);


subplot(2,2,2);
imshow(log(abs(b2)),[]);
title('Matlabs output abs logged');
fclose(t256);

diff = abs(c-b2);
subplot(2,2,3);
imshow(diff,[]);
title('Difference between outputs');

subplot(2,2,4);
imshow(log(diff),[]);
title('log of difference, exaggerates differences');








