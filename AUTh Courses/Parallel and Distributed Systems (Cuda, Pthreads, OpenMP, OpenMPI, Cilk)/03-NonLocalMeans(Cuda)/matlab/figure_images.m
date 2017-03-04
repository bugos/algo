%% SCRIPT: PIPELINE_NON_LOCAL_MEANS
%
% Pipeline for non local means algorithm as described in [1].
%
% The code thus far is implemented in CPU.
%
% DEPENDENCIES
%
% [1] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local
%     algorithm for image denoising. In 2005 IEEE Computer Society
%     Conference on Computer Vision and Pattern Recognition (CVPR’05),
%      volume 2, pages 60–65. IEEE, 2005.
%
  
  clear all %#ok
  close all

  %% PARAMETERS
  
  % input image
  pathImg   = '../data/house.mat';
  strImgVar = 'house';
  
  %input Noisy image
  pathNoisyImg     = '../data/NoisyHouse64.mat';
  strNoisyImgVar   = 'J';

  %input Denoisy image
  pathDeNoisyImg     = '../data/DeHouse64.mat';
  strDeNoisyImgVar   = 'If';

  % OUR image
  pathImgOur   = '../data/OURimage.mat';
  strImgVarOur = 'If';


  %% The Original         
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar); 
  figure('Name','Original Image')
  imagesc(I)
  colormap gray;
  
  %% INPUT THE NOISY IMAGE DATA
  fprintf('...loading the noisy image...\n')
  ioNoisyImg = matfile(pathNoisyImg);
  N          = ioNoisyImg.(strNoisyImgVar);
  figure('Name','Noisy Image');
  imagesc(N)
  colormap gray;

  %% INPUT THE DeNOISY IMAGE DATA
  fprintf('...loading the denoisy image...\n')
  ioDeNoisyImg = matfile(pathDeNoisyImg);
  D          = ioDeNoisyImg.(strDeNoisyImgVar);
  figure('Name','DeNoisy Image');
  imagesc(D)
  colormap gray;

   %% INPUT THE DeNOISY IMAGE DATA
  fprintf('...loading the our image...\n')
  OurImg = matfile(pathImgOur );
  O          = OurImg.(strImgVarOur);
  figure('Name','Our Image');
  imagesc(O)
  colormap gray;



