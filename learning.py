import xlrd
import tensorflow as tf
import numpy as np

size = 3
SNR = 6.5

def swap_row(L, a, b):
    L_temp = L[a] + 0
    L[a] = L[b]
    L[b] = L_temp

def read_data(table, size):
    data_list = []
    nrows = table.nrows
    ncols = table.ncols
    for row in range(nrows):
        data_list.extend([table.row_values(row)])
    for item in data_list:
        for col in range(ncols):
            if item[col] >= size:
                item[col] = item[col] % size
    return data_list

def construct(data, LDPC_matrix, size):
    data_rows = len(data)
    data_cols = len(data[0])
    for row in range(data_rows):
        for col in range(data_cols):
            shift = data[row][col]
            for i in range(size):
                if shift == -1:
                    for j in range(size):
                        LDPC_matrix[int(row * size + i)][int(col * size + j)] = 0
                else:
                    shift_col = (i + shift) % size
                    for j in range(size):
                        if j == shift_col:
                            LDPC_matrix[int(row * size + i)][int(col * size + j)] = 1
                        else:
                            LDPC_matrix[int(row * size + i)][int(col * size + j)] = 0

def gauss_jordan(H):
    m = np.array(H, int)
    m_rows = len(m)
    m_cols = len(m[0])
    for row in range(m_rows):
        max_row = row
        if m[row][m_cols - m_rows + row] != 1:
            for row_2 in range(row + 1, m_rows):
                if m[row_2][m_cols - m_rows + row] == 1:
                    max_row = row_2
                    break
            swap_row(m, max_row, row)
        if m[row][m_cols - m_rows + row] != 0:
            for row_2 in range(row + 1, m_rows):
                if m[row_2][m_cols - m_rows + row] == 1:
                    m[row_2] = np.bitwise_xor(m[row_2], m[row])
    for row in range(m_rows - 1, 0 - 1, -1):
        for row_2 in range(0, row):
            if m[row_2][m_cols - m_rows + row] == 1:
                m[row_2] = np.bitwise_xor(m[row_2], m[row])
    return m

def sgn(a):
    if a < 0:
        return -1
    else:
        return 1

def add_noise(signal, SNR):
    N = len(signal)
    R = 0.5

    Var = 1 / (2 * R * pow(10.0, SNR / 10))
    sigma = pow(Var, 1 / 2)
    noise = np.random.normal(scale=sigma, size=N)

    return signal + noise

def decode_NN(m, sig, code_len, weight, weight_):
    r = 1 * sig
    rows = int(len(m) / code_len)
    cols = code_len
    m_check = []  ## check which nodes = 1
    for row in range(rows):
        m_check += [[]]
        for col in range(cols):
            if m[row*code_len + col] == 1:
                m_check[row] += [col]
    c_matrix = []
    for row in range(rows):  ## initialization
        for col in range(cols):
            if col in m_check[row]:
                c_matrix += [r[col]]
            else:
                c_matrix += [0]
    count = 0
    count_ = 0
    for i in range(5):  ## process
        for row in range(rows):
            temp = [[]]
            for col in m_check[row]:
                R = 1000
                sign = 1
                for col_2 in m_check[row]:
                    if col_2 == col:
                        continue
                    else:
                        sign = sign * tf.sign(c_matrix[row * code_len + col_2])
                        R = tf.minimum(tf.abs(c_matrix[row * code_len + col_2]), R)
                temp[0] += [(R * sign * weight[count])]
            for col in m_check[row]:
                c_matrix[row*code_len + col] = temp[0][0]
                del temp[0][0]
        count += 1
    
        s = [0 for j in range(code_len)]
        for col in range(cols):
            for row in range(rows): ## update signal r
                s[col] += c_matrix[row*code_len + col] * weight[count]
        r = r * weight_[count_] + s
        count += 1
        count_ += 1
        
        for row in range(rows):  ## update matrix for cal r
            for col in m_check[row]:
                c_matrix[row*code_len + col] = r[col] - c_matrix[row*code_len + col]

    r = tf.convert_to_tensor(r)
    y_hat = tf.negative(r)
    
    return y_hat

# Read data
data = xlrd.open_workbook('LDPC.xlsx')
table = data.sheets()[0]
data_list = read_data(table, size)

# Construct H matrix
nrows = table.nrows
ncols = table.ncols
H = np.zeros((size * nrows, size * ncols))
construct(data_list, H, size)

# Construct G matrix [I | A_tran]
H_gauss = H + 0
H_gauss = gauss_jordan(H_gauss)  # [A | I]
I = np.identity(size * (ncols - nrows))
H_tran = np.transpose(H_gauss) + 0
G = np.zeros((size * (ncols - nrows), size * ncols))
for row in range(size * (ncols - nrows)):
    G[row] = np.append(I[row], H_tran[row])

H = H.flatten()

### training ###

def make_batch(batch_size):
    u = np.random.randint(2, size=size * (ncols - nrows))
    x = np.matmul(u, G) % 2
    signal = x * (-2) + 1
    x_train = add_noise(signal, SNR)
    y_train = x

    return x_train, y_train

BATCH_SIZE = size * (ncols - nrows)
sig_len = size * (ncols - nrows)
N = len(G[0])
N_epochs = 200
learning_rate = 0.001

x = tf.placeholder(dtype=tf.float64, shape=(N,), name='x')
y = tf.placeholder(dtype=tf.float64, shape=(N,), name='y')
weight = [tf.Variable(dtype=tf.float64, initial_value=1, trainable=True)
          for _ in range(5*2)]
weight_ = [tf.Variable(dtype=tf.float64, initial_value=1, trainable=True)
           for _ in range(5)]
print(0)
y_hat = decode_NN(H, x, N, weight, weight_)
print(1)
cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_hat))
print(2)
update = tf.train.AdamOptimizer(learning_rate).minimize(cost)
print(3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('Start training......')
    for epoch in range(N_epochs):
        total_err = 0
        print('Epoch: ', epoch + 1)
        for i in range(50):
            x_train, y_train = make_batch(BATCH_SIZE)
            err, _ = sess.run([cost, update], feed_dict={x: x_train, y: y_train})
            total_err += err
        print('Total error = ', total_err)
 
    weight_val = sess.run(weight)
    np.save('weight_6.5_50.npy', weight_val)
    weight_val_ = sess.run(weight_)
    np.save('weight_6.5_50_.npy', weight_val_)
    print('Parameters saved.')
