% makes Tower512.txt 

file = fopen('Tower1024.txt','r');

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


fi = fopen('Tower512.txt','w');

fprintf(fi,'512 512\n');

for i = 1:512
    for j = 1:512
        fprintf(fi,'%.0f ',c(i,j));
    end
    fprintf(fi,'\n');
end

fclose(fi);








