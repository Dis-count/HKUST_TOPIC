\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  3 alpha[0] + 2 alpha[1] + 2 alpha[2] + 2 alpha[3] + gamma[0] + gamma[1]
   + gamma[2] + gamma[3] + gamma[4] + gamma[5] + gamma[6] + gamma[7]
   + gamma[8] + gamma[9]
Subject To
 R0: alpha[0] + beta[0,0] >= 2
 R1: alpha[0] + beta[0,1] >= 2
 R2: alpha[0] + beta[0,2] >= 2
 R3: alpha[0] + beta[0,3] >= 2
 R4: alpha[0] + beta[0,4] >= 2
 R5: alpha[0] + beta[0,5] >= 2
 R6: alpha[0] + beta[0,6] >= 2
 R7: alpha[0] + beta[0,7] >= 2
 R8: alpha[0] + beta[0,8] >= 2
 R9: alpha[0] + beta[0,9] >= 2
 R10: alpha[1] + beta[1,0] >= 4
 R11: alpha[1] + beta[1,1] >= 4
 R12: alpha[1] + beta[1,2] >= 4
 R13: alpha[1] + beta[1,3] >= 4
 R14: alpha[1] + beta[1,4] >= 4
 R15: alpha[1] + beta[1,5] >= 4
 R16: alpha[1] + beta[1,6] >= 4
 R17: alpha[1] + beta[1,7] >= 4
 R18: alpha[1] + beta[1,8] >= 4
 R19: alpha[1] + beta[1,9] >= 4
 R20: alpha[2] + beta[2,0] >= 6
 R21: alpha[2] + beta[2,1] >= 6
 R22: alpha[2] + beta[2,2] >= 6
 R23: alpha[2] + beta[2,3] >= 6
 R24: alpha[2] + beta[2,4] >= 6
 R25: alpha[2] + beta[2,5] >= 6
 R26: alpha[2] + beta[2,6] >= 6
 R27: alpha[2] + beta[2,7] >= 6
 R28: alpha[2] + beta[2,8] >= 6
 R29: alpha[2] + beta[2,9] >= 6
 R30: alpha[3] + beta[3,0] >= 8
 R31: alpha[3] + beta[3,1] >= 8
 R32: alpha[3] + beta[3,2] >= 8
 R33: alpha[3] + beta[3,3] >= 8
 R34: alpha[3] + beta[3,4] >= 8
 R35: alpha[3] + beta[3,5] >= 8
 R36: alpha[3] + beta[3,6] >= 8
 R37: alpha[3] + beta[3,7] >= 8
 R38: alpha[3] + beta[3,8] >= 8
 R39: alpha[3] + beta[3,9] >= 8
 R40: - gamma[0] <= 0
 R41: - gamma[1] <= 0
 R42: - gamma[2] <= 0
 R43: - gamma[3] <= 0
 R44: - gamma[4] <= 0
 R45: - gamma[5] <= 0
 R46: - gamma[6] <= 0
 R47: - gamma[7] <= 0
 R48: - gamma[8] <= 0
 R49: - gamma[9] <= 0
 R50: 2 beta[0,9] - gamma[9] <= 0
Bounds
End
