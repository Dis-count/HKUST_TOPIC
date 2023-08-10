\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + 2 C10 + 2 C11 + 2 C12
   + 2 C13 + 2 C14 + 2 C15 + 2 C16 + 2 C17 + 2 C18 + 2 C19 + 3 C20 + 3 C21
   + 3 C22 + 3 C23 + 3 C24 + 3 C25 + 3 C26 + 3 C27 + 3 C28 + 3 C29 + 4 C30
   + 4 C31 + 4 C32 + 4 C33 + 4 C34 + 4 C35 + 4 C36 + 4 C37 + 4 C38 + 4 C39
Subject To
 R0: 2 C0 + 3 C10 + 4 C20 + 5 C30 <= 6
 R1: 2 C1 + 3 C11 + 4 C21 + 5 C31 <= 12
 R2: 2 C2 + 3 C12 + 4 C22 + 5 C32 <= 9
 R3: 2 C3 + 3 C13 + 4 C23 + 5 C33 <= 9
 R4: 2 C4 + 3 C14 + 4 C24 + 5 C34 <= 13
 R5: 2 C5 + 3 C15 + 4 C25 + 5 C35 <= 17
 R6: 2 C6 + 3 C16 + 4 C26 + 5 C36 <= 21
 R7: 2 C7 + 3 C17 + 4 C27 + 5 C37 <= 21
 R8: 2 C8 + 3 C18 + 4 C28 + 5 C38 <= 21
 R9: 2 C9 + 3 C19 + 4 C29 + 5 C39 <= 21
 R10: C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 >= 1.31147e-06
 R11: C10 + C11 + C12 + C13 + C14 + C15 + C16 + C17 + C18 + C19
   >= 16.99999934475407
 R12: C20 + C21 + C22 + C23 + C24 + C25 + C26 + C27 + C28 + C29 >= 2
 R13: C30 + C31 + C32 + C33 + C34 + C35 + C36 + C37 + C38 + C39 >= 18
Bounds
Generals
 C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14 C15 C16 C17 C18 C19 C20
 C21 C22 C23 C24 C25 C26 C27 C28 C29 C30 C31 C32 C33 C34 C35 C36 C37 C38
 C39
End
