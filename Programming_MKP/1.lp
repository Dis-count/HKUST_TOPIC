\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  2 alpha[0] + 4 alpha[1] + 10 alpha[2] + gamma[0] + gamma[1] + gamma[2]
   + gamma[3]
Subject To
 R0: alpha[0] + beta[0,0] >= 4
 R1: alpha[0] + beta[0,1] >= 4
 R2: alpha[0] + beta[0,2] >= 4
 R3: alpha[0] + beta[0,3] >= 4
 R4: alpha[1] + beta[1,0] >= 6
 R5: alpha[1] + beta[1,1] >= 6
 R6: alpha[1] + beta[1,2] >= 6
 R7: alpha[1] + beta[1,3] >= 6
 R8: alpha[2] + beta[2,0] >= 8
 R9: alpha[2] + beta[2,1] >= 8
 R10: alpha[2] + beta[2,2] >= 8
 R11: alpha[2] + beta[2,3] >= 8
 R12: - gamma[0] <= 0
 R13: beta[0,0] + beta[1,0] - gamma[0] <= 0
 R14: 2 beta[0,0] - gamma[0] <= 0
 R15: beta[2,0] - gamma[0] <= 0
 R16: - gamma[1] <= 0
 R17: 2 beta[1,1] - gamma[1] <= 0
 R18: beta[0,1] + beta[2,1] - gamma[1] <= 0
 R19: - gamma[2] <= 0
 R20: 2 beta[1,2] - gamma[2] <= 0
 R21: beta[0,2] + beta[2,2] - gamma[2] <= 0
 R22: - gamma[3] <= 0
 R23: beta[1,3] - gamma[3] <= 0
 R24: beta[0,3] - gamma[3] <= 0
Bounds
End
