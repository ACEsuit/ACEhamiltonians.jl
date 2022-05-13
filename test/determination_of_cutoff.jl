using ACEhamiltonians, LinearAlgebra, Plots

file_name = ["data/FCC-MD-500K/SK-supercell-001.h5", "data/FCC-MD-500K/SK-supercell-002.h5"]
index = [map(rand, (1:728, 1:729)) for i = 1:600]
for (i,k) in enumerate(index)
   if k[2] == k[1]
      index[i] = (k[1],k[2]+1)
   elseif k[2] < k[1]
      index[i] = (k[2],k[1])
   end
end

data = data_read(file_name,index)

function data2HS(data)
   len = length(data[1][1])
   set_of_Hblock = zeros(14,14,len)
   set_of_Sblock = zeros(14,14,len)
   for i = 1:len
      for j = 1:3, k = 1:3
         set_of_Hblock[j,k,i] = data[1][1][i][j,k][1]
         set_of_Sblock[j,k,i] = data[1][2][i][j,k][1]
      end
      for j = 1:2, k = 1:3
         set_of_Hblock[4+3(j-1):6+3(j-1),k,i] = data[2][1][i][j,k]
         set_of_Sblock[4+3(j-1):6+3(j-1),k,i] = data[2][2][i][j,k]
      end
      for j = 1:1, k = 1:3
         set_of_Hblock[10:14,k,i] = data[3][1][i][j,k]
         set_of_Sblock[10:14,k,i] = data[3][2][i][j,k]
      end
      for j = 1:2, k = 1:2
         set_of_Hblock[4+3(j-1):6+3(j-1),4+3(k-1):6+3(k-1),i] = data[4][1][i][j,k]
         set_of_Sblock[4+3(j-1):6+3(j-1),4+3(k-1):6+3(k-1),i] = data[4][2][i][j,k]
      end
      for j = 1:1, k = 1:2
         set_of_Hblock[10:14,4+3(k-1):6+3(k-1),i] = data[5][1][i][j,k]
         set_of_Sblock[10:14,4+3(k-1):6+3(k-1),i] = data[5][2][i][j,k]
      end
      set_of_Hblock[10:14,10:14,i] = data[6][1][i][1]
      set_of_Sblock[10:14,10:14,i] = data[6][2][i][1]
      set_of_Hblock[1:3,4:9,i] = set_of_Hblock[4:9,1:3,i]'
      set_of_Sblock[1:3,4:9,i] = set_of_Sblock[4:9,1:3,i]'
      set_of_Hblock[1:3,10:14,i] = set_of_Hblock[10:14,1:3,i]'
      set_of_Sblock[1:3,10:14,i] = set_of_Sblock[10:14,1:3,i]'
      set_of_Hblock[4:9,10:14,i] = set_of_Hblock[10:14,4:9,i]'
      set_of_Sblock[4:9,10:14,i] = set_of_Sblock[10:14,4:9,i]'
   end
   return set_of_Hblock, set_of_Sblock
end

function norm_of_subblock(data)
   H, S = data2HS(data)
   nosbH = zeros(6, size(H)[3])
   nosbS = zeros(6, size(H)[3])
   for i = 1:size(H)[3]
      nosbH[1, i] = norm(H[1:3,1:3,i])
      nosbH[2, i] = norm(H[4:9,1:3,i])
      nosbH[3, i] = norm(H[10:14,1:3,i])
      nosbH[4, i] = norm(H[4:9,4:9,i])
      nosbH[5, i] = norm(H[10:14,4:9,i])
      nosbH[6, i] = norm(H[10:14,10:14,i])
      
      nosbS[1, i] = norm(S[1:3,1:3,i])
      nosbS[2, i] = norm(S[4:9,1:3,i])
      nosbS[3, i] = norm(S[10:14,1:3,i])
      nosbS[4, i] = norm(S[4:9,4:9,i])
      nosbS[5, i] = norm(S[10:14,4:9,i])
      nosbS[6, i] = norm(S[10:14,10:14,i])
   end
   return nosbH,nosbS
end

norm_of_block(H::Array{Float64, 3}) = [ norm(H[:,:,l]) for l = 1:size(H)[3] ]
norm_of_block(data) = norm_of_block.(data2HS(data))
bondlength_of_block(data) = [ norm(data[1][3][l][1].rr0) for l = 1:length(data[1][1]) ]


H, S = data2HS(data)
# Norm of blocks, sub-blocks
nb = norm_of_block(data)
nsb = norm_of_subblock(data)
nsb_h_pp = nsb[1][4,:]
nsb_h_dd = nsb[2][6,:]
# Bond lengths
bb = bondlength_of_block(data)
# sorted from small to large
order = sortperm(bb)

plot(bb[order],nb[1][order],xlabel = "length of the (i,j)-th bond", ylabel = "norm of H_{ij}/S_{ij}", label = "H")
plot!(bb[order],nb[2][order],label = "S")
plot(bb[order],nsb_h_pp[order])
plot(bb[order],nsb_h_dd[order])

plot(bb[order][2000:end],nb[1][order][2000:end].+1e-12,xlabel = "length of the (i,j)-th bond",ylabel = "norm of H_{ij}/S_{ij}",label = "H", yaxis = :log)
plot!(bb[order][2000:end],nb[2][order][2000:end].+1e-12,label = "S", yaxis = :log)

## check different cutoff for different blocks...

Rs = data[1][3]
norm_rs = norm.([ Rs[i][1].rr0 for i = 1:length(Rs) ])
order = sortperm(norm_rs)
len = length(Rs)

i = 1
data_ss =  data[i]
Hsubs_ss = data_ss[1]
Ssubs_ss = data_ss[2]

norm_hss1 = norm.([ Hsubs_ss[i][1,1] for i = 1:len ])
norm_hss2 = norm.([ Hsubs_ss[i][1,2] for i = 1:len ])
norm_hss3 = norm.([ Hsubs_ss[i][1,3] for i = 1:len ])
norm_hss4 = norm.([ Hsubs_ss[i][2,1] for i = 1:len ])
norm_hss5 = norm.([ Hsubs_ss[i][2,2] for i = 1:len ])
norm_hss6 = norm.([ Hsubs_ss[i][2,3] for i = 1:len ])
norm_hss7 = norm.([ Hsubs_ss[i][3,1] for i = 1:len ])
norm_hss8 = norm.([ Hsubs_ss[i][3,2] for i = 1:len ])
norm_hss9 = norm.([ Hsubs_ss[i][3,3] for i = 1:len ])

plot(norm_rs[order],norm_hss1[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss2[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss3[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss4[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss5[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss6[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss7[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss8[order].+1e-5,yaxis=:log)
plot!(norm_rs[order],norm_hss9[order].+1e-5,yaxis=:log) # 10+

i = 2
data_sd = data[i]
Hsubs_sd = data_sd[1]
Ssubs_sd = data_sd[2]

norm_hsd1 = norm.([ Hsubs_sd[i][1,1] for i = 1:len ])
norm_hsd2 = norm.([ Hsubs_sd[i][1,2] for i = 1:len ])
norm_hsd3 = norm.([ Hsubs_sd[i][1,3] for i = 1:len ])
norm_hsd4 = norm.([ Hsubs_sd[i][2,1] for i = 1:len ])
norm_hsd5 = norm.([ Hsubs_sd[i][2,2] for i = 1:len ])
norm_hsd6 = norm.([ Hsubs_sd[i][2,3] for i = 1:len ])

plot(norm_rs[order],1./(norm_hsd1[order].+1e-20),xlabel = "length of the (i,j)-th bond",ylabel = "norm of H^{sp}_{ij}",label = "H^{sp}[1,1]",yaxis=:log) # 5
plot!(norm_rs[order],norm_hsd2[order].+1e-20,xlabel = "length of the (i,j)-th bond",ylabel = "norm of H^{sp}_{ij}",label = "H^{sp}[1,2]") # 7
plot!(norm_rs[order],norm_hsd3[order].+1e-20,xlabel = "length of the (i,j)-th bond",ylabel = "norm of H^{sp}_{ij}",label = "H^{sp}[1,3]") # 10
plot!(norm_rs[order],norm_hsd4[order].+1e-20,xlabel = "length of the (i,j)-th bond",ylabel = "norm of H^{sp}_{ij}",label = "H^{sp}[2,1]") # 7
plot!(norm_rs[order],norm_hsd5[order].+1e-20,xlabel = "length of the (i,j)-th bond",ylabel = "norm of H^{sp}_{ij}",label = "H^{sp}[2,2]") # 9
plot!(norm_rs[order],norm_hsd6[order].+1e-20,xlabel = "length of the (i,j)-th bond",ylabel = "norm of H^{sp}_{ij}",label = "H^{sp}[2,3]") # 10+

i = 3
data_sp = data[i]
Hsubs_sp = data_sp[1]
Ssubs_sp = data_sp[2]

norm_hsp1 = norm.([ Hsubs_sp[i][1,1] for i = 1:len ])
norm_hsp2 = norm.([ Hsubs_sp[i][1,2] for i = 1:len ])
norm_hsp3 = norm.([ Hsubs_sp[i][1,3] for i = 1:len ])


plot(norm_rs[order],norm_hsp1[order].+1e-5,yaxis=:log) # 7
plot!(norm_rs[order],norm_hsp2[order].+1e-5,yaxis=:log) # 9
plot!(norm_rs[order],norm_hsp3[order].+1e-5,yaxis=:log) # 10+

i = 4
data_pp = data[i]
Hsubs_pp = data_pp[1]
Ssubs_pp = data_pp[2]

norm_hpp1 = norm.([ Hsubs_pp[i][1,1] for i = 1:len ])
norm_hpp2 = norm.([ Hsubs_pp[i][1,2] for i = 1:len ])
norm_hpp3 = norm.([ Hsubs_pp[i][2,1] for i = 1:len ])
norm_hpp4 = norm.([ Hsubs_pp[i][2,2] for i = 1:len ])

plot(norm_rs[order],norm_hpp1[order].+1e-5,yaxis=:log) # 8
plot!(norm_rs[order],norm_hpp2[order].+1e-5,yaxis=:log) # 10
plot!(norm_rs[order],norm_hpp3[order].+1e-5,yaxis=:log) # 10
plot!(norm_rs[order],norm_hpp4[order].+1e-5,yaxis=:log) # 10+

i = 5
data_pd = data[i]
Hsubs_pd = data_pd[1]
Ssubs_pd = data_pd[2]

norm_hpd1 = norm.([ Hsubs_pd[i][1,1] for i = 1:len ])
norm_hpd2 = norm.([ Hsubs_pd[i][1,2] for i = 1:len ])


plot(norm_rs[order],norm_hpd1[order].+1e-5,yaxis=:log) # 10
plot!(norm_rs[order],norm_hpd2[order].+1e-5,yaxis=:log) # 10+

i = 6
data_dd = data[i]
Hsubs_dd = data_dd[1]
Ssubs_dd = data_dd[2]

norm_hdd = norm.([ Hsubs_dd[i][1,1] for i = 1:len ])

plot(norm_rs[order],norm_hdd[order].+1e-5,yaxis=:log)

