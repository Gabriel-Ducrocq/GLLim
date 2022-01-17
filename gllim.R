install.packages("RJSONIO")
library(xLLiM)
library(RcppCNPy)
library(RJSONIO)

cls_tt <- t(npyLoad("data/cls_tt.npy"))[3:513, 1:10000]/10
cls_te <- t(npyLoad("data/cls_te.npy"))[3:513, 1:10000]/10
cls_ee <- t(npyLoad("data/cls_ee.npy"))[3:513, 1:10000]/10
all_theta <- t(npyLoad("data/all_theta.npy"))[1:6, 1:10000]

covariates <- rbind(cls_tt, cls_ee, cls_te)
gllim_res <- gllim(all_theta, covariates, in_K = 10, verb = 1)

cls_tt_test <- npyLoad("data_true/cls_tt.npy")[3:513]/10
cls_ee_test <- npyLoad("data_true/cls_ee.npy")[3:513]/10
cls_te_test <- npyLoad("data_true/cls_te.npy")[3:513]/10
covariates_test <- matrix(c(cls_tt_test, cls_ee_test,cls_te_test), ncol=1)
res <- gllim_inverse_map(covariates_test, gllim_res)

jsonSaved <- toJSON(gllim_res)
write(jsonSaved, "data/parameters.json")
npySave("data/posterior_weights.npy", res$alpha)