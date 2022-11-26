install.packages("RcppCNPy")
install.packages("xLLiM")
install.packages("RJSONIO")
library(xLLiM)
library(RcppCNPy)
library(RJSONIO)
#library(matrixcalc)
#library(ramify)
#library(reticulate)



cls_tt = npyLoad("all_cls_hat.npy")
all_theta <- npyLoad("all_theta.npy")
dim(all_theta)
dim(t(cls_tt[1:10000, 3:2501]))

#Works well with K=5
gllim_res <- gllim(t(all_theta[1:9000, 1:5]), t(cls_tt[1:9000, 3:2501]/10), in_K = 30, verb = 1)
res <- gllim_inverse_map(t(cls_tt[9001:10000, 3:2501])/10, gllim_res)
#dists = sqrt(colSums((res$x_exp - t(all_theta[9001:10000, 1:5]))**2))
npySave("all_predictions.npy", res$x_exp)




#rowMeans(abs((res$x_exp - t(all_theta[9001:10000, 1:5]))))
#t <- res$all_sigmastar
#save(t, file="all_sigmas.RData")
#npySave("all_means.npy", res$x_exp)
#npySave("all_weights.npy", res$alpha)
#t <- res$all_sigmastar
#np = import("numpy")
#save(t, file="test.npy")

#for(i in 1:5){
#  path = paste("all_sigma_", as.character(i), ".npy", sep="")
#  npySave(path, res$all_sigmastar[,,i])
#}
         
### Trying with uniform samples:
#cls_tt_uniform = npyLoad("all_cls_hat_uniform.npy")
#all_theta_uniform <- npyLoad("all_theta_uniform.npy")
#dim(all_theta_uniform)
#dim(t(cls_tt_uniform[1:10000, 3:2501]))

#gllim_res_uniform <- gllim(t(all_theta_uniform[1:9000, 1:5]), t(cls_tt_uniform[1:9000, 3:2501]/10), in_K = 5, verb = 1)
#res_uniform <- gllim_inverse_map(t(cls_tt_uniform[9001:10000, 3:2501])/10, gllim_res_uniform)
#dists = sqrt(colSums((res_uniform$x_exp - t(all_theta_uniform[9001:10000, 1:5]))**2))

#rowMeans(abs((res_uniform$x_exp - t(all_theta_uniform[9001:10000, 1:5])))





         
         
         
##Trying with k=30:
#gllim_res <- gllim(t(all_theta[1:9000, 1:5]), t(cls_tt[1:9000, 3:2501]/10), in_K = 20, verb = 1)
#res <- gllim_inverse_map(t(cls_tt[9001:10000, 3:2501])/10, gllim_res)
#dists = sqrt(colSums((res$x_exp - t(all_theta[9001:10000, 1:5]))**2))

#res <- gllim_get_posterior(t(cls_tt[9001:10000, 3:2501])/10, gllim_res)      
#all_res <-  gllim_inverse_map(t(cls_tt[1:10000, 3:2501])/10, gllim_res) 
#all_res_uniform <-  gllim_inverse_map(t(cls_tt_uniform[1:10000, 3:2501])/10, gllim_res_uniform) 
#npySave("all_means.npy", all_res$x_exp)
#npySave("all_weights.npy", all_res$alpha)
#t <- res$all_sigmastar
#np = import("numpy")
#save(t, file="test.npy")

#for(i in 1:5){
#  path = paste("all_sigma_", as.character(i), ".npy", sep="")
#  npySave(path, all_res$all_sigmastar[,,i])
#}

##Analysing uniform results
#all_res_uniform <-  gllim_inverse_map(t(cls_tt_uniform[1:10000, 3:2501])/10, gllim_res_uniform) 
#npySave("all_means_uniform.npy", all_res_uniform$x_exp)
#npySave("all_weights_uniform.npy", all_res_uniform$alpha)

#for(i in 1:5){
#  path = paste("all_sigma_uniform", as.character(i), ".npy", sep="")
#  npySave(path, all_res$all_sigmastar[,,i])
#}
         
#cls_hat_true =cls_tt[10000:10000, 3:2501]       
         
         

#stdrowMeans(abs((res$x_exp - t(all_theta[9001:10000, 1:5])/10)))#/(t(all_theta[9001:10000, 1:5])/10))
#mean(dists)
#hist(dists)






#cls_tt <- t(npyLoad("data/cls_tt.npy"))[3:513, 1:10000]/10
#cls_te <- t(npyLoad("data/cls_te.npy"))[3:513, 1:10000]/10
#cls_ee <- t(npyLoad("data/cls_ee.npy"))[3:513, 1:10000]/10
#all_theta <- t(npyLoad("data/all_theta.npy"))[1:6, 1:10000]

#covariates <- rbind(cls_tt, cls_ee, cls_te)
#gllim_res <- gllim(all_theta, covariates, in_K = 10, verb = 1)

#cls_tt_test <- npyLoad("data_true/cls_tt.npy")[3:513]/10
#cls_ee_test <- npyLoad("data_true/cls_ee.npy")[3:513]/10
#cls_te_test <- npyLoad("data_true/cls_te.npy")[3:513]/10
#covariates_test <- matrix(c(cls_tt_test, cls_ee_test,cls_te_test), ncol=1)
#res <- gllim_inverse_map(covariates_test, gllim_res)

#jsonSaved <- toJSON(gllim_res)
#write(jsonSaved, "data/parameters.json")
#npySave("data/posterior_weights.npy", res$alpha)



#data(data.xllim)  
#dim(data.xllim) #  size 52 y 100
#responses = data.xllim[1:2,] # 2 responses in rows and 100 observations in columns
#covariates = data.xllim[3:52,] # 50 covariates in rows and 100 observations in columns

## Set 5 components in the model
#K = 5

## Step 1: initialization of the posterior probabilities (class assignments) 
## via standard EM for a joint Gaussian Mixture Model
#r = emgm(rbind(responses, covariates), init=K); 

## Step 2: estimation of the model
## Default Lw=0 and cstr$Sigma="i"
#mod = gllim(responses,covariates,in_K=K,in_r=r)
#}