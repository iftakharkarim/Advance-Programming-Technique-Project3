% makes Tower128.txt 

file = fopen('Tower256.txt','r');

% matrixifies it
a = fscanf(file,'%i');
fclose(file);
dim = a(1);
size = a(1);
a = a(3:end);
b = zeros(dim,dim);

for i = 1:(dim*dim)
   b(i) = a(i); 
end
%b = b';
format long;
c = imresize(b,.5);


fi = fopen('Tower128.txt','w');

fprintf(fi,'128 128\n');

for i = 1:128
    for j = 1:128
        fprintf(fi,'%.0f ',c(i,j));
    end
    fprintf(fi,'\n');
end

fclose(fi);








