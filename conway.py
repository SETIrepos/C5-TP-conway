import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from argparse import ArgumentParser
from build import conway

def profile(grid_size, **kwargs):
    grid_ping = torch.randint(0, 2, (grid_size, grid_size), dtype=torch.int32, device="cuda")
    grid_pong = torch.empty_like(grid_ping)
    game = conway.GameOfLife()

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5): # warmup
        game.step(grid_ping, grid_pong, stream)
        grid_ping, grid_pong = grid_pong, grid_ping

    # profile
    start.record(stream)
    for _ in range(kwargs["iterations"]):
        game.step(grid_ping, grid_pong, stream)
        grid_ping, grid_pong = grid_pong, grid_ping
    end.record(stream)
    stream.synchronize()

    print(f"FPS={kwargs['iterations'] / start.elapsed_time(end) * 1000:.2f}")


def test(**kwargs):
    grid_size = 16
    grid = torch.randint(0, 2, (grid_size, grid_size), dtype=torch.int32, device="cuda")

    # Reference implementation in pure torch
    kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32, device="cuda")
    neighbors = torch.nn.functional.conv2d(grid.reshape((1, 1, grid_size, grid_size)).to(torch.float32), kernel, padding="same").reshape((grid_size, grid_size)).round().to(torch.int32)
    torch_out = (neighbors == 3) | (grid & (neighbors == 2))

    # Our implementation
    out = torch.zeros_like(grid)
    game = conway.GameOfLife()
    game.step(grid, out, None)

    match = torch.allclose(torch_out, out)
    if match:
        print('Both implementations match')
    else:
        print('Error, see difference:')
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

    torch_pattern = torch.zeros((len(pattern), max_cols), dtype=torch.int32, device="cuda")
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
    grid_ping = torch.randint(0, 2, (grid_size, grid_size), dtype=torch.int32)
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("action", choices=["test", "profile", "show"])
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--file")
    parser.add_argument("--iterations", type=int, default=100)
    args = vars(parser.parse_args())
    action = args.pop("action")

    {"test": test, "profile": profile, "show": show}[action](**args)
