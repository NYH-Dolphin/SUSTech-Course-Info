function img2 = jpg2coe(infile, outfile)
% RGB img to RGB coe
    img = imread(infile) ;
    height = size(img, 1);
    width = size(img, 2);
    s = fopen(outfile,'wb');
    fprintf(s, '%s\n', 'memory_initialization_radix=16;');
    fprintf(s, '%s\n', 'memory_initialization_vector=');
    cnt = 0;
    img2 = img;
    for row = 1:height
        for col = 1: width 
            cnt = cnt + 1;
            R =img(row,col,1);
            G =img(row,col,2);
            B =img(row,col,3);
            Rb = dec2bin(double(R),8);
            Gb = dec2bin(double(G),8);
            Bb = dec2bin(double(B),8);
            img2(row,col,1) = bin2dec([Rb(1:5),'000']);
            img2(row,col,2) = bin2dec([Gb(1:6),'00']);
            img2(row,col,3) = bin2dec([Bb(1:5),'000']);
            Outbyte = [Rb(1:5),Gb(1:6),Bb(1:5)]; %RGB565
            % Outbyte = [Rb(1:8),Gb(1:8),Bb(1:8)]; %RGB888
            % Outbyte = [Rb(1:4),Gb(1:4),Bb(1:4)]; %RGB444
            if (strcmp(Outbyte(1:5),'00000'))
                fprintf(s, '0%X', bin2dec(Outbyte));
            else
                fprintf(s, '%X', bin2dec(Outbyte));
            end
            if ((col==width)&&(row==height))
                fprintf(s, '%c', ';');
            else
                fprintf(s, '%c', ',');
            end
        end
    end
    fclose(s);
end
% Lena_640x480
% img2 = jpg2coe('test.jpg', 'test.coe');
% img2 = jpg2coe('Lena_640x480.jpg', 'Lena_640x480.coe');
% img2 = jpg2coe('Lena_320x240.jpg', 'Lena_320x240_16b.coe');