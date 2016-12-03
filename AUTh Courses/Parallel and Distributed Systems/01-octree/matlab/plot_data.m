threads=[1; 2; 4; 8; 16; 32; 128];
threads=[1:5];
N_number=[1048576; 2097152; 4194304; 8388608; 16777216; ];
colors = ['r', 'g', 'c', 'm', 'k', 'b'];

d = read_data('diades.cilk-2016-11-28-02:41:37-test-tree-Plummer.txt', 1, 1, 5, 6, 4);
%hash, morton, sort, dataR

%data serial
ds = read_data('diades.code-2016-11-27-00:29:24-test-tree-cube.txt', 1, 1, 5, 1, 4);

figure;
fields = fieldnames(d);
for i = 1:numel(fields)
    subplot(2,2,i);
    func = d.(fields{i});
    func = squeeze(func(1,1,:,:));
    
    funcs = ds.(fields{i});
    funcs = squeeze(funcs(1,1,:,:));

    title(strcat( fields{i}, ' function, Cilk, Sphere input'));
    ylabel('Parallel / Serial time %');
    xlabel('T where 2^T are the running threads ');
    hold on;
    for N=1:5  
       if (N==4 && 0) % openmp fix
          continue;
      end
      funcN = squeeze(func(N,:));
      funcNs = squeeze(funcs(N,:));
      if (N==4 && 0) % pthreads fix
          funcNs = squeeze(funcs(5,:));
      end
      funcN = funcN(2:end);
      color = colors(N);
      plot(threads, funcN ./ funcNs * 100, colors(N));
    end
    legend('N=2\^20','N=2\^21','N=2\^22','N=2\^23', 'N=2\^24');
end