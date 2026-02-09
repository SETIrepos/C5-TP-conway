import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser
from build import conway

def profile(grid_size, **kwargs):
    grid_ping = torch.randint(0, 65536, (grid_size//4, grid_size//4), dtype=torch.uint16, device="cuda")
    grid_pong1 = torch.empty_like(grid_ping)
    grid_pong2 = torch.empty_like(grid_ping)
    game = conway.GameOfLife()

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5): # warmup
        game.step(grid_ping, grid_pong1, grid_pong2, stream)
        grid_ping, grid_pong2 = grid_pong2, grid_ping

    # profile
    start.record(stream)
    for _ in range(kwargs["iterations"]):
        game.step(grid_ping, grid_pong1, grid_pong2, stream)
        grid_ping, grid_pong2 = grid_pong2, grid_ping
    end.record(stream)
    stream.synchronize()

    print(f"FPS={kwargs['iterations']*2 / start.elapsed_time(end) * 1000:.2f}")


def pack_grid(full):
    """Pack a full uint8 grid (H, W) into uint16 blocks where each uint16 encodes a 4x4 block.

    Mapping: bits 0..15 correspond to positions in the 4x4 block in row-major order.
    """
    H, W = full.shape
    if H % 4 != 0 or W % 4 != 0:
        raise ValueError("grid dimensions must be divisible by 4 for 4x4 packing")
    # Reshape to blocks: (H//4, 4, W//4, 4) -> (H//4, W//4, 4, 4)
    blocks = full.view(H // 4, 4, W // 4, 4).permute(0, 2, 1, 3)  # (H4, W4, 4, 4)
    # Build multipliers on the same device but in a signed integer dtype supported by CUDA
    multipliers = (1 << torch.arange(16, device=full.device, dtype=torch.int64)).view(4, 4).to(torch.int32)
    # Compute in int32 to avoid unsupported UInt16 ops on CUDA, then mask and cast back to uint16
    summed = (blocks.to(torch.int32) * multipliers).sum(dim=(2, 3)).to(torch.int32) & 0xFFFF
    packed = summed.to(torch.uint16)
    return packed


def unpack_grid(packed):
    """Unpack uint16 blocks back into a full uint8 grid."""
    H4, W4 = packed.shape
    shifts = torch.arange(16, device=packed.device, dtype=torch.int64).view(4, 4)
    bits = ((packed.to(torch.int64).unsqueeze(-1).unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    full = bits.permute(0, 2, 1, 3).reshape(H4 * 4, W4 * 4)
    return full


def test(**kwargs):
    grid_size = 16
    # Full (unpacked) reference grid
    grid = torch.randint(0, 2, (grid_size, grid_size), dtype=torch.uint8, device="cuda")

    # Reference implementation in pure torch (unpacked)
    kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32, device="cuda")
    neighbors = torch.nn.functional.conv2d(grid.reshape((1, 1, grid_size, grid_size)).to(torch.float32), kernel, padding="same").reshape((grid_size, grid_size)).round().to(torch.uint8)
    torch_out = (neighbors == 3) | (grid & (neighbors == 2))

    # Pack for GPU implementation (each uint16 encodes a 4x4 block)
    packed_grid = pack_grid(grid)
    packed_out1 = torch.zeros_like(packed_grid)
    packed_out2 = torch.zeros_like(packed_grid) # dummy second output for kernel signature

    # Call GPU implementation which expects uint16 packed representation
    game = conway.GameOfLife()
    game.step(packed_grid, packed_out1, packed_out2)

    # Unpack GPU output to full grid for comparison
    out = unpack_grid(packed_out1)

    match = torch.all(torch.eq(torch_out, out))
    if match:
        print('Both implementations match')
    else:
        print('Error, voir la différence:')
        print(torch_out ^ out)


def read_pattern(file_path, tensor):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove comments and empty lines
    pattern_lines = [line.strip() for line in lines if not line.startswith('!') and line.strip()]

    # Convert pattern to a 2D list of 0s and 1s
    pattern = []
    max_cols = 0
    for line in pattern_lines:
        row = [1 if char == 'O' else 0 for char in line]
        max_cols = max(max_cols, len(row))
        pattern.append(row)

    torch_pattern = torch.zeros((len(pattern), max_cols), dtype=torch.uint8, device="cuda")
    for i, row in enumerate(pattern):
        for j, val in enumerate(row):
            torch_pattern[i, j] = val

    # Place the pattern in the tensor
    pattern_height, pattern_width = torch_pattern.shape
    tensor_height, tensor_width = tensor.shape

    # Calculate the top-left corner of the pattern in the tensor
    start_row = (tensor_height - pattern_height) // 2
    start_col = (tensor_width - pattern_width) // 2

    # Ensure the pattern fits within the tensor
    if start_row < 0 or start_col < 0:
        raise ValueError("Pattern is too large to fit in the tensor")

    # Place the pattern in the tensor
    tensor.fill_(0)
    tensor[start_row:start_row + pattern_height, start_col:start_col + pattern_width] = torch_pattern


def update(frameNum, img, grid_ping, grid_pong, game):
    if frameNum % 2 == 1:
        grid_ping, grid_pong = grid_pong, grid_ping
    game.step(grid_ping, grid_pong, None)
    img.set_data(grid_pong.cpu().numpy())
    return img


def show(grid_size, **kwargs):
    grid_ping = torch.randint(0, 2, (grid_size, grid_size), dtype=torch.uint8)
    if kwargs.get("file") is not None:
        read_pattern(kwargs["file"], grid_ping)
    grid_ping = grid_ping.to("cuda")
    grid_pong = torch.empty_like(grid_ping)

    fig, ax = plt.subplots()
    img = ax.imshow(grid_ping.cpu().numpy(), interpolation='nearest')
    game = conway.GameOfLife()

    ani = animation.FuncAnimation(fig, update, fargs=(img, grid_ping, grid_pong, game),
                                  interval=1, save_count=1)
    plt.show()


def pack_unpack_test():
    """Unit test to demonstrate pack/unpack on an 8x8 grid (produces 4 uint16 blocks)."""
    print("Running pack/unpack test on 8x8 grid (4 packed uint16 blocks)")
    grid_size = 8
    grid = torch.randint(0, 2, (grid_size, grid_size), dtype=torch.uint8, device="cpu")
    print("Original grid (8x8):")
    print(grid)

    packed = pack_grid(grid)
    print(f"Packed grid (shape {packed.shape}):")
    print(packed)

    print("Packed blocks in binary:")
    for i in range(packed.shape[0]):
        for j in range(packed.shape[1]):
            val = int(packed[i, j].item())
            print(f" block ({i},{j}) = {val:016b} ({val})")

    unpacked = unpack_grid(packed)
    print("Unpacked grid:")
    print(unpacked)

    equal = torch.equal(grid, unpacked)
    print("Equal after unpack:", bool(equal))
    if not equal:
        print("Difference (original ^ unpacked):")
        print(grid ^ unpacked)
    else:
        print("PASS: pack/unpack preserved grid")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("action", choices=["test", "profile", "show", "packtest"])
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--file")
    parser.add_argument("--iterations", type=int, default=100)
    args = vars(parser.parse_args())
    action = args.pop("action")

    {"test": test, "profile": profile, "show": show, "packtest": lambda **kwargs: pack_unpack_test()}[action](**args)
