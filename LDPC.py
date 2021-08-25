size = 12
import xlrd
import random

def swap_row(L,a,b):
    L_temp = L[a]
    L[a] = L[b]
    L[b] = L_temp

def print_matrix(m):
    m_rows = len(m)
    m_cols = len(m[0])
    for row in range(m_rows):
        for col in range(m_cols): 
            print(int(m[row][col]),end = " ")
        print("\n",end = "")

def matrix_tran(m, m_tran):
    m_rows = len(m)
    m_cols = len(m[0])
    for cols in range(m_cols):
        m_tran += [[]]
        for row in range(m_rows):
            m_tran[cols] += [m[row][cols]]

def read_data(data_list,table):
    nrows=table.nrows
    ncols=table.ncols
    for row in range(nrows):
    	data_list.extend([table.row_values(row)])    
    for item in data_list:
        for col in range(ncols):
            if item[col] >= size:
                item[col] = item[col] % size

def construct_LDPC_matrix(data,LDPC_matrix):
    data_rows = len(data)
    data_cols = len(data[0])
    for row in range(data_rows):
        for col in range(data_cols):
            shift = data[row][col]
            for i in range(size):
                if shift == -1:
                    for col_Matrix in range(size):
                        LDPC_matrix[int(row*size+i)] = LDPC_matrix[int(row*size+i)]+[0] 
                else:
                    shift_col = (i+shift) % size
                    for col_Matrix in range(size):
                        if col_Matrix == shift_col:
                            LDPC_matrix[int(row*size+i)]=LDPC_matrix[int(row*size+i)]+[1]
                        else:
                            LDPC_matrix[int(row*size+i)]=LDPC_matrix[int(row*size+i)]+[0]

def gauss_jordan(m):
    m_rows = len(m)
    m_cols = len(m[0])
    for row in range(m_rows):
        max_row = row
        if m[row][m_cols-m_rows+row] != 1:
            for row_2 in range(row+1,m_rows):
                if m[row_2][m_cols-m_rows+row] == 1:
                    max_row = row_2
                    break
            swap_row(m,max_row,row)
        if m[row][m_cols-m_rows+row] != 0:
            for row_2 in range(row+1,m_rows):
                if m[row_2][m_cols-m_rows+row] == 1:
                    for col in range(m_cols):
                        m[row_2][col] = m[row_2][col] ^ m[row][col]                  
    for row in range(m_rows-1, 0-1, -1):
        for row_2 in range(0,row):
            if m[row_2][m_cols-m_rows+row] == 1:
                for col in range(m_cols):
                    m[row_2][col] = m[row_2][col] ^ m[row][col]
       
def construct_LDPC_matrix_tran(m, m_tran):
    m_tran_rows = len(m_tran)
    A_tran_rows = len(m_tran)
    A_tran_cols = len(m)
    for row in range(m_tran_rows):
        for col in range(m_tran_rows):
            if row == col:
                m_tran[row] += [1]
            else:
                m_tran[row] += [0]
    for row in range(A_tran_rows):
        for col in range(A_tran_cols):
            m_tran[row] += [m[col][row]]

def construct_random_vector(u, u_len):
    for i in range(u_len):
        a = random.randint(0,1)
        u[0] += [a]

def matrix_mul(m1, m2, m_mul):
    m1_rows = len(m1)
    m1_cols = len(m1[0])
    m2_rows = len(m2)
    m2_cols = len(m2[0])    
    if m1_cols == m2_rows:
        for row_1 in range(m1_rows):
            m_mul += [[]]
            for col_2 in range(m2_cols):
                dot = 0
                for col_1 in range(m1_cols):
                    dot = dot ^ m1[row_1][col_1] * m2[col_1][col_2]
                m_mul[row_1] += [int(dot)]
    else:
        print("failed")
            
def main():
    data = xlrd.open_workbook('LDPC.xlsx')
    table = data.sheets()[0]
    data_list = []
    
    nrows=table.nrows
    ncols=table.ncols
    LDPC_matrix_original = [[] for i in range(int(size*nrows))]
    LDPC_matrix = [[] for i in range(int(size*nrows))]
  
    read_data(data_list, table)
    construct_LDPC_matrix(data_list, LDPC_matrix_original)
    construct_LDPC_matrix(data_list, LDPC_matrix)
    gauss_jordan(LDPC_matrix)                                   ## [A|I]       
    
    LDPC_matrix_tran = [[] for i in range(int(size*ncols)-int(size*nrows))]
    construct_LDPC_matrix_tran(LDPC_matrix, LDPC_matrix_tran)   ## [I|A_tran]
    
    random_vector = [[]]
    vector_len = len(LDPC_matrix_tran)
    construct_random_vector(random_vector,vector_len)           

    x = []                                                      ## [rand_vec][LDPC_tran]
    matrix_mul(random_vector, LDPC_matrix_tran, x)
    
    s = []                                                      ## [LDPC_origin][x_tran]
    LDPC_matrix_original_tran = []
    matrix_tran(LDPC_matrix_original, LDPC_matrix_original_tran)
    matrix_mul(x, LDPC_matrix_original_tran, s)
    
    print("data list:")
    print_matrix(data_list)
    print("\nLDPC matrix original:")
    print_matrix(LDPC_matrix_original)
    print("\nLDPC matrix:")
    print_matrix(LDPC_matrix)
    print("\nLDPC matrix transpose:")
    print_matrix(LDPC_matrix_tran)
    print("\nrandom vector:")
    print_matrix(random_vector)
    print("\nx = [rand_vect]*[LDPC_tran]:")
    print_matrix(x)
    print("\ns = [x]*[H_origin_tran]:")
    print_matrix(s)
    if 1 in s:
        print("wrong")
    else:
        print("correct")

    return

main()