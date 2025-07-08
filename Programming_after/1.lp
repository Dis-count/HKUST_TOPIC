\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  alpha[0] + alpha[1] + alpha[2] + alpha[3] + gamma[0] + gamma[1]
Subject To
 R0: alpha[0] + beta[0,0] >= 1
 R1: alpha[0] + beta[0,1] >= 1
 R2: alpha[1] + beta[1,0] >= 2
 R3: alpha[1] + beta[1,1] >= 2
 R4: alpha[2] + beta[2,0] >= 3
 R5: alpha[2] + beta[2,1] >= 3
 R6: alpha[3] + beta[3,0] >= 4
 R7: alpha[3] + beta[3,1] >= 4
 R8: beta[3,0] - gamma[0] <= 0
 R9: beta[2,0] - gamma[0] <= 0
 R10: beta[0,0] + beta[1,0] - gamma[0] <= 0
 R11: 2 beta[0,0] - gamma[0] <= 0
 R12: beta[3,1] - gamma[1] <= 0
 R13: beta[0,1] + beta[2,1] - gamma[1] <= 0
 R14: 2 beta[1,1] - gamma[1] <= 0
 R15: 3 beta[0,1] - gamma[1] <= 0
 R16: beta[0,1] + beta[1,1] - gamma[1] <= 0
Bounds
 beta[0,0] free
 beta[0,1] free
 beta[1,0] free
 beta[1,1] free
 beta[2,0] free
 beta[2,1] free
 beta[3,0] free
 beta[3,1] free
End
