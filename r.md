
  RGB = imread("LOCATION OF THE MASK FILE");
  
  I = rgb2gray(RGB);
  
  csvwrite('LOCATION WHERE WE WANT TO SAVE THIS FILE(csv)',I)


- a 
