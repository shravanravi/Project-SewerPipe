function inpaintImage = inpaint ( original , mask )

% Takes an image (with overlaying data) and it's colored version as inputs and generates the
% weighted minmum norm reconstruction of the image which is free of the
% overlaying data

% Get size of image
[N1, N2, bytesppix] = size (original);
inpaintImage = zeros([N1, N2, bytesppix]);
h = [1; -1];
L = numel(h);

for itr = 1:bytesppix
    Imsplit_original = original(:,:,itr);
    
    T1 = @(x) reshape(real(ifft(bsxfun(@times,...
         fft(reshape(x,[N1,N2]),N1+L-1,1),...
         fft(h,N1+L-1,1)),N1+L-1,1)),(N1+L-1)*N2,1);
    S1.type = '()';
    S1.subs = {1:N1,':'};
    T1h = @(x) reshape(subsref(real(ifft(bsxfun(@times,...
    fft(reshape(x,[N1+L-1,N2]),N1+L-1,1),...
    conj(fft(h,N1+L-1,1))),N1+L-1,1)),S1),N1*N2,1);

    T2 = @(x) reshape(real(ifft(bsxfun(@times,...
         fft(reshape(x,[N1,N2]),N2+L-1,2),...
         fft(h.',N2+L-1,2)),N2+L-1,2)),N1*(N2+L-1),1);
    S2.type = '()';
    S2.subs = {':',1:N2};
    T2h = @(x) reshape(subsref(real(ifft(bsxfun(@times,...
          fft(reshape(x,[N1,N2+L-1]),N2+L-1,2),...
          conj(fft(h.',N2+L-1,2))),N2+L-1,2)),S2),N1*N2,1);

    S.type = '()';
    S.subs = {find(not(mask))};
    H = @(x) subsref(x,S);
    Hh = @(x) subsasgn(zeros(N1*N2,1),S,x);

    T = @(x) [T1(x);T2(x)];
    Th = @(x) (T1h(x(1:(N1+L-1)*N2)) + T2h(x((N1+L-1)*N2+1:(N1+L-1)*N2 + ...
         N1*(N2+L-1))));
    xp = Hh(H(Imsplit_original));
    
    Q.type = '()';
    Q.subs = {find(mask)};
    Z  = @(x) subsasgn(zeros(N1*N2,1),Q,x);
    Zh = @(x) subsref(x,Q);

    c = pcg(@(x) Zh(Th(T(Z(x)))),-Zh(Th(T(xp))), 1e-6, 1000);
    
    inpaintImage(:,:,itr) = reshape(Z(c)+xp,N1,N2);
end

end
