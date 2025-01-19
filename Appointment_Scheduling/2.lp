\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  S[2,0] + w_ij_prime[0,0,1,0] + 1.1 w_ij_prime[0,1,2,0]
   + 1.2 w_ij_prime[0,2,3,0] + w_ij_prime[1,1,1,0]
   + 1.1 w_ij_prime[1,2,2,0] + w_ij_prime[2,2,1,0]
Subject To
 R0: A[2] <= 30
 R1: A[0] = 0
 R2: - A[0] + S[0,0] >= 0
 R3: - A[1] + S[1,0] >= 0
 R4: - A[2] + S[2,0] >= 0
 R5: - S[0,0] + S[1,0] >= 8
 R6: - S[1,0] + S[2,0] >= 22
 R7: A[0] - S[0,0] + varT[0,0,1,0] = 0
 R8: A[1] - S[0,0] + varT[0,1,2,0] = 0
 R9: A[2] - S[0,0] + varT[0,2,3,0] = 0
 R10: A[1] - S[1,0] + varT[1,1,1,0] = 0
 R11: A[2] - S[1,0] + varT[1,2,2,0] = 0
 R12: A[2] - S[2,0] + varT[2,2,1,0] = 0
 R13: - w_ij_prime[0,0,1,0] - w_ij_prime[0,1,2,0] - w_ij_prime[0,2,3,0]
   + w_ij[0,0,1,0] = 0
 R14: - w_ij_prime[0,1,2,0] - w_ij_prime[0,2,3,0] + w_ij[0,1,2,0] = 0
 R15: - w_ij_prime[0,2,3,0] + w_ij[0,2,3,0] = 0
 R16: - w_ij_prime[0,1,2,0] - w_ij_prime[0,2,3,0] - w_ij_prime[1,1,1,0]
   - w_ij_prime[1,2,2,0] + w_ij[1,1,1,0] = 0
 R17: - w_ij_prime[0,2,3,0] - w_ij_prime[1,2,2,0] + w_ij[1,2,2,0] = 0
 R18: - w_ij_prime[0,2,3,0] - w_ij_prime[1,2,2,0] - w_ij_prime[2,2,1,0]
   + w_ij[2,2,1,0] = 0
Bounds
General Constraints
 GC0: w_ij[0,0,1,0] = MAX ( varT[0,0,1,0] , 0 )
 GC1: w_ij[0,1,2,0] = MAX ( varT[0,1,2,0] , 0 )
 GC2: w_ij[0,2,3,0] = MAX ( varT[0,2,3,0] , 0 )
 GC3: w_ij[1,1,1,0] = MAX ( varT[1,1,1,0] , 0 )
 GC4: w_ij[1,2,2,0] = MAX ( varT[1,2,2,0] , 0 )
 GC5: w_ij[2,2,1,0] = MAX ( varT[2,2,1,0] , 0 )
End
