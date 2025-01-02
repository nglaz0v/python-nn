"""
Пример кодирования с коррекцией ошибок.
"""

def calc_parity_bits(data_bits):
    # Определение количества битов чётности
    m = len(data_bits)
    r = 1
    while 2 ** r - r - 1 < m:
        r += 1
    return r

def get_parity_positions(num_bits):
    positions = [2**i for i in range(num_bits)]
    return positions

def encode_hamming(data_bits):
    m = len(data_bits)
    r = calc_parity_bits(data_bits)
    n = m + r

    encoded = [None] * n
    parity_positions = get_parity_positions(r)

    # Заполняем биты данных
    j = 0
    for i in range(1, n + 1):
        if i in parity_positions:
            encoded[i - 1] = 0  # заполняем место под бит чётности нулём
        else:
            encoded[i - 1] = int(data_bits[j])
            j += 1

    # Вычисляем биты чётности
    for i in range(r):
        parity_pos = 2**i
        parity = 0
        for j in range(1, n + 1):
            if j & parity_pos == parity_pos:
                parity ^= encoded[j - 1]
        encoded[parity_pos - 1] = parity
    return encoded

def introduce_error(encoded_bits, error_position):
    encoded_bits[error_position] ^= 1
    return encoded_bits

def decode_hamming(encoded_bits):
    n = len(encoded_bits)
    r = calc_parity_bits([0] * (n - calc_parity_bits(encoded_bits)))

    error_position = 0

    for i in range(r):
        parity_pos = 2**i
        parity = 0
        for j in range(1, n + 1):
            if j & parity_pos == parity_pos:
                parity ^= encoded_bits[j - 1]
        if parity:
            error_position += parity_pos

    if error_position:
        encoded_bits[error_position - 1] ^= 1

    decoded = []
    parity_positions = get_parity_positions(r)
    for i in range(1, n + 1):
        if i not in parity_positions:
            decoded.append(encoded_bits[i - 1])

    return decoded, error_position

# Пример использования
data_bits = "1011"
print(f"Original data: {data_bits}")

encoded_bits = encode_hamming(data_bits)
print(f"Encoded data: {''.join(map(str, encoded_bits))}")

# Внесение ошибки
error_position = 3  # например, ошибка на позиции 3
encoded_with_error = introduce_error(encoded_bits[:], error_position - 1)
print(f"Encoded data with error: {''.join(map(str, encoded_with_error))} ")

decoded_bits, detected_error_position = decode_hamming(encoded_with_error)
print(f"Decoded data: {''.join(map(str, decoded_bits))}")
print(f"Detected error position: {detected_error_position}")
