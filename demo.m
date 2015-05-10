%% load an image and compute superpixels
im=imread('../colorization/example_res.bmp');
addpath(genpath('/gpfs/home/jxy198/work/ya-imagekit/src/misc/MatlabFns'));
[l, Am, Sp, d] = slic(im, 320, 10, 1., 'median');

%% compute inverse colorization based on STC-QP
[gamma, lambda]=inverse_colorization(im, l);

%% visualize results
h=l;
for i=find(abs(lambda)<1E-6)'; h(l(:)==i) = 0; end
show(drawregionboundaries(h, im, [255 255 255]));
