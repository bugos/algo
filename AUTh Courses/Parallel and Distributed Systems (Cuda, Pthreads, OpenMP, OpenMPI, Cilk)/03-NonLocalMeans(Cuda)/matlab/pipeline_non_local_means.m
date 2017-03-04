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
  
  clear all 
  close all

  %% PARAMETERS

  % input Original image
  pathImg          = '../data/house.mat';
  strImgVar        = 'house';
  
  %input Noisy image
  pathNoisyImg     = '../data/NoisyHouse512.mat';
  strNoisyImgVar   = 'J';

  %input Denoisy image
  pathDeNoisyImg     = '../data/DeHouse64.mat';
  strDeNoisyImgVar   = 'If';
   
  % filter sigma value
  filtSigma = 0.02;
  patchSigma = 5/3;
  neighSize=5;
  

  %% USEFUL FUNCTIONS

  %%Find the gaussian table
  MAX_NSZ=7;
  k =floor(MAX_NSZ/2); 
  for i= -k:k
    for j= -k:k        
      gauss(k+1+i,1+k+j)=exp(-sqrt(i^2+j^2))/patchSigma;    
    end
  end
  G = reshape(gauss,1,MAX_NSZ*MAX_NSZ);
  totalsum=sum(G);
  G=G./totalsum;
  G=single(G);
  
  %% (BEGIN)
  fprintf('...Begin %s...\n',mfilename);  

  %% INPUT DATA
  fprintf('...loading input data...\n') 
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);
  
  %% PREPROCESS  
  
  %% INPUT THE NOISY IMAGE DATA
  fprintf('...loading the noisy image...\n')
  ioNoisyImg = matfile(pathNoisyImg);
  N          = ioNoisyImg.(strNoisyImgVar);

  
  %%padding the perimetre of the table
  N=padarray(N,[(neighSize-1)./2,(neighSize-1)./2], 'symmetric','both');

  %Reshape from 2d to 1d
  N=N(:)' ;

  %% INPUT THE DeNOISY IMAGE DATA
  fprintf('...loading the denoisy image...\n')
  ioDeNoisyImg = matfile(pathDeNoisyImg);
  D          = ioDeNoisyImg.(strDeNoisyImgVar);

  %% KERNEL
  fprintf('====================KERNEL=================\n')
  k = parallel.gpu.CUDAKernel( '../cuda/nonLocalMeans8.ptx', ...
                               '../cuda/nonLocalMeans8.cu');

  multi = 1
  Size = 512;                             
  threadsPerBlock = [16 16];
  numberOfBlocks  = ceil( [Size Size] ./ (multi .* threadsPerBlock) );                          
  k.GridSize        = numberOfBlocks;                              
  k.ThreadBlockSize = threadsPerBlock;
  setConstantMemory(k, 'gaussDistW', G);

  %% DATA
  If = zeros([Size Size]);
  input  = gpuArray(N);
  If = gpuArray(If);
  input = input(:)';
  If = reshape(If, [1,Size*Size]);
  input=single(input);
  If=single(If);

  tic;
  If = gather( feval(k,input, If, Size, filtSigma) );
  toc
  fprintf('====================END OF KERNEL=================\n')
  
  If = reshape(If,[Size,Size]);  

  %%Save the image with noise
  imwrite(If,'/home/alexandre/denoisedimg.png');
  fprintf(' - Save OUR img..\n')
  save('OURimage.mat','If')

  %%compare Denoised imgs
  fprintf('====================Compare denoised imgs==================\n')
  size(If);
  D=single(D);
  size(D);
  norm(If(:)-D(:))
  peaksnr = psnr(If(:), single(D(:)), 1)

  %% (END)
  fprintf('...end %s...\n',mfilename);


