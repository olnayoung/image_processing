import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def get_row_col(W, idx):
    row, col = idx // W, idx % W
    return row, col

def hasBit(bitmap, pos):
    return (bitmap >> pos) & 1

def find(s_buf, n):
    H, W = s_buf.size()
    row, col = get_row_col(W, n)

    while s_buf[row, col] != n:
        n = s_buf[row, col]

    # while s_buf[n] != n:
    #     n = s_buf[n]
    return n

def find_n_compress(s_buf, n):
    H, W = s_buf.size()
    row, col = get_row_col(W, n)
    
    id = n
    while s_buf[row, col] != n:
        n = s_buf[row, col]
        s_buf[row, col] = n

    # while s_buf[n] != n:
    #     n = s_buf[n]
    #     s_buf[id] = n
    return n

def union(s_buf, a, b):
    H, W = s_buf.size()

    done = False
    while not done:
        a = find(s_buf, a)
        b = find(s_buf, b)
        
        row_a, col_a = get_row_col(W, a)
        row_b, col_b = get_row_col(W, b)

        if a < b:
            old = s_buf[row_b, col_b]
            s_buf[row_b, col_b] = a
            done = (old == b)
            b = old
        elif b < a:
            old = s_buf[row_a, col_a]
            s_buf[row_a, col_a] = b
            # old = torch.min(s_buf[row_a, col_a], b)
            done = (old == a)
            a = old
        else:
            done = True

        # if a < b:
        #     old = torch.min(s_buf[b], a)
        #     done = (old == b)
        #     b = old
        # elif b < a:
        #     old = torch.min(s_buf[a], b)
        #     done = (old == a)
        #     a = old
        # else:
        #     done = True

def get_connected_componnets(inputs):
    N, C, H, W = inputs.size()
    assert C == 1, "inputs must be [N, 1, H, W] shape"
    assert H % 2 == 0, "height must be an even number"
    assert W % 2 == 0, "width must be an even number"

    labels = torch.zeros((N, C, H, W), dtype=torch.int32, device=inputs.device)
    counts_init = torch.zeros((N, C, H, W), dtype=torch.int32, device=inputs.device)
    counts_final = torch.zeros((N, C, H, W), dtype=torch.int32, device=inputs.device)

    for n in tqdm(range(N)):
        offset = n * H * W

        # Initialize labels
        labels = torch.arange(H * W, device=inputs.device, dtype=torch.int32).view(H, W)

        # Merge
        for row in range(0, H, 2):
            for col in range(0, W, 2):
                idx = row * W + col
                if row >= H or col >= W:
                    continue

                P = 0
                if inputs[n, 0, row, col]:
                    P |= 0x777
                if row + 1 < H and inputs[n, 0, row + 1, col]:
                    P |= 0x777 << 4
                if col + 1 < W and inputs[n, 0, row, col + 1]:
                    P |= 0x777 << 1

                if col == 0:
                    P &= 0xEEEE
                if col + 1 >= W:
                    P &= 0x3333
                elif col + 2 >= W:
                    P &= 0x7777

                if row == 0:
                    P &= 0xFFF0
                if row + 1 >= H:
                    P &= 0xFF

                if P > 0:
                    if hasBit(P, 0) and inputs[n, 0, row - 1, col - 1]:
                        union(labels, idx, idx - 2 * W - 2)
                    if (hasBit(P, 1) and inputs[n, 0, row - 1, col]) or (hasBit(P, 2) and inputs[n, 0, row - 1, col + 1]):
                        union(labels, idx, idx - 2 * W)
                    if hasBit(P, 3) and inputs[n, 0, row - 1, col + 2]:
                        union(labels, idx, idx - 2 * W + 2)
                    if (hasBit(P, 4) and inputs[n, 0, row, col - 1]) or (hasBit(P, 8) and inputs[n, 0, row + 1, col - 1]):
                        union(labels, idx, idx - 2)

        # Compression
        for row in range(0, H, 2):
            for col in range(0, W, 2):
                idx = row * W + col
                if row < H and col < W:
                    find_n_compress(labels, idx)

        # Final labeling
        for row in range(0, H, 2):
            for col in range(0, W, 2):
                idx = row * W + col
                if row >= H or col >= W:
                    continue

                y = labels[row, col] + 1

                if inputs[n, 0, row, col]:
                    labels[row, col] = y
                else:
                    labels[row, col] = 0

                if col + 1 < W:
                    if inputs[n, 0, row, col + 1]:
                        labels[row, col+1] = y
                    else:
                        labels[row, col+1] = 0

                    if row + 1 < H:
                        if inputs[n, 0, row + 1, col + 1]:
                            labels[row+1, col] = y
                        else:
                            labels[row+1, col] = 0

                if row + 1 < H:
                    if inputs[n, 0, row + 1, col]:
                        labels[row+1, col] = y
                    else:
                        labels[row+1, col] = 0

                # y = labels[idx] + 1

                # if inputs[n, 0, row, col]:
                #     labels[idx] = y
                # else:
                #     labels[idx] = 0

                # if col + 1 < W:
                #     if inputs[n, 0, row, col + 1]:
                #         labels[idx + 1] = y
                #     else:
                #         labels[idx + 1] = 0

                #     if row + 1 < H:
                #         if inputs[n, 0, row + 1, col + 1]:
                #             labels[idx + W + 1] = y
                #         else:
                #             labels[idx + W + 1] = 0

                # if row + 1 < H:
                #     if inputs[n, 0, row + 1, col]:
                #         labels[idx + W] = y
                #     else:
                #         labels[idx + W] = 0

        # Get the counting of each pixel
        counts_init = torch.bincount(labels.view(-1), minlength=H * W).view(H, W)

        for row in range(H):
            for col in range(W):
                idx = row * W + col
                # y = labels[idx]
                # if y > 0:
                #     counts_final[idx] = counts_init[y - 1]
                # else:
                #     counts_final[idx] = 0
                y = labels[row, col]
                if y > 0:
                    row_y, col_y = get_row_col(W, y-1)
                    counts_final[n, 0, row, col] = counts_init[row_y, col_y]
                else:
                    counts_final[n, 0, row, col] = 0

    return labels, counts_final