function [gamma, lambda] = inverse_colorization(imgInput, segmentMap)
% Inverse Colorization
% Input
%   imgInput: original image in RGB with size n x m x 3
%   constr:   template of constraints with n*m x k

n = size(imgInput, 1); m = size(imgInput, 2);
imgSize = n*m;

assert(size(segmentMap,1) == n && size(segmentMap,2) == m);

ntscIm = rgb2ntsc(double(imgInput) / 255);

H=affinity_matrix(ntscIm(:,:,1));
M=speye(imgSize, imgSize);
q=zeros(imgSize,1);

counts=sqrt(histc(segmentMap(:), 1:max(segmentMap(:))));
D=sparse(1:imgSize, segmentMap(:), 1./counts(segmentMap(:)));

inputImage=ntscIm(:,:,3);
[gamma, lambda, xx] = stc_qp(H, q, M, D, reshape(inputImage, imgSize, 1), 0.05);
outputImage= reshape(xx, n, m);
figure;
imshow(inputImage, []);
figure;
imshow(outputImage, []);