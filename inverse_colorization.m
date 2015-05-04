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

counts=histc(segmentMap(:), 1:max(segmentMap(:)));
D=sparse(1:imgSize, segmentMap(:), 1./counts(segmentMap(:)));
[gamma, lambda] = stc_qp(H, q, M, D, reshape(ntscIm(:,:,2), imgSize, 1), 1E-4);
