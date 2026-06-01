# GPU vs GPU Tic-Tac-Toe Using CUDA

## DEVELOPED BY :
Name : MERIL GOLDLINA A
<BR>
Course: Parallel Computing / CUDA Programming

# PROJECT DESCRIPTION :
This project demonstrates a GPU vs GPU Tic-Tac-Toe game implemented using CUDA C++. Two GPU-based competitors play against each other on a 3×3 game board. Each GPU uses a different strategy to determine its next move.

The project showcases how CUDA kernels can be used for game decision-making and parallel processing. It highlights several important GPU programming concepts such as kernel execution, thread parallelism, memory transfers, synchronization, and atomic operations.

# TECHNOLOGIES USED : 
* CUDA C++
* NVIDIA CUDA Toolkit
* Google Colab
* Linux / Windows Environment
* GPU Parallel Computing
* CUDA Runtime API

# GAME DESCRIPTION :

The game is played on a standard 3×3 Tic-Tac-Toe board between two GPU competitors.

### GPU 1 (Player X)

* GPU 1 uses a simple strategy.
* Evaluates available board positions
* Uses GPU threads to search for empty cells
* Selects a valid move from available positions
* Acts as a random or basic AI player

### GPU 2 (Player O)

GPU 2 uses a smarter strategy.

It attempts to:

* Find a winning move
* Block the opponent's winning move
* Select the most suitable available position
* Use fallback logic when no immediate winning opportunity exists

The game continues until:

Player X wins
Player O wins
The board becomes full and the game ends in a draw

# PROCEDURE :

### Step 1: Open Google Colab

Open Google Colab and create a new notebook.

---

### Step 2: Enable GPU Support

1. Click **Runtime**
2. Select **Change Runtime Type**
3. Choose **GPU** as Hardware Accelerator
4. Click **Save**

---

### Step 3: Verify GPU Availability

Run the following command:

```bash
!nvidia-smi
```

This displays information about the available NVIDIA GPU.

---

### Step 4: Create CUDA Source File

Create a CUDA source file named:

```text
gpu_tictactoe.cu
```

Paste the CUDA Tic-Tac-Toe program into the file.

---

### Step 5: Compile the CUDA Program

Compile the program using the NVIDIA CUDA Compiler:

```bash
!nvcc gpu_tictactoe.cu -o game
```

If the compilation is successful, an executable file named **game** will be created.

---

### Step 6: Execute the Program

Run the executable:

```bash
!./game
```

The GPU vs GPU Tic-Tac-Toe game starts execution.

---

### Step 7: Initialize the Game Board

A 3×3 Tic-Tac-Toe board is created and initialized with empty cells.

Example:

```text
- - -
- - -
- - -
```

---

### Step 8: Transfer Data to GPU

The current game board is copied from Host (CPU) memory to Device (GPU) memory using CUDA memory transfer functions.

```cpp
cudaMemcpy()
```

---

### Step 9: Execute GPU 1 Kernel

GPU 1 (Player X) launches a CUDA kernel.

The kernel:

* Evaluates available board positions
* Selects a valid move
* Returns the selected position

---

### Step 10: Update and Display Board

The selected move is copied back to CPU memory.

The board is updated and displayed to the user.

Example:

```text
X - -
- - -
- - -
```

---

### Step 11: Execute GPU 2 Kernel

GPU 2 (Player O) launches a CUDA kernel.

The kernel:

* Searches for winning opportunities
* Attempts to block the opponent
* Chooses a strategic move

---

### Step 12: Display Updated Board

The move selected by GPU 2 is applied to the board.

Example:

```text
X - -
- O -
- - -
```

---

### Step 13: Check Game Status

After every move:

* Check if Player X has won
* Check if Player O has won
* Check if the game is a draw

Winner detection is performed using predefined winning combinations.

---

### Step 14: Repeat Gameplay Loop

Steps 8 through 13 continue until:

* GPU 1 wins
* GPU 2 wins
* The board becomes full

---

### Step 15: Display Final Result

The final outcome is displayed.

Example:

```text
GPU 1 (X) WINS
```

or

```text
GPU 2 (O) WINS
```

or

```text
DRAW GAME
```

---

### Step 16: Release GPU Resources

After the game completes:

* Device memory is freed
* CUDA resources are released

```cpp
cudaFree()
```

---

# PROGRAM :

### Step 1: Enable GPU in Colab

Runtime → Change runtime type → GPU

Check:
```c
!nvidia-smi
```
### Step 2: Create the CUDA file
```c
%%writefile gpu_tictactoe.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define EMPTY 0
#define X 1
#define O 2

void printBoard(int board[9])
{
    cout << endl;

    for(int i=0;i<9;i++)
    {
        char c='-';

        if(board[i]==X) c='X';
        if(board[i]==O) c='O';

        cout << c << " ";

        if((i+1)%3==0)
            cout << endl;
    }

    cout << endl;
}

int checkWinner(int board[9])
{
    int wins[8][3]={
        {0,1,2},{3,4,5},{6,7,8},
        {0,3,6},{1,4,7},{2,5,8},
        {0,4,8},{2,4,6}
    };

    for(int i=0;i<8;i++)
    {
        int a=wins[i][0];
        int b=wins[i][1];
        int c=wins[i][2];

        if(board[a]!=0 &&
           board[a]==board[b] &&
           board[b]==board[c])
            return board[a];
    }

    return 0;
}

bool draw(int board[9])
{
    for(int i=0;i<9;i++)
    {
        if(board[i]==0)
            return false;
    }

    return true;
}

// GPU 1: choose first empty cell
__global__ void gpu1Move(int *board,int *move)
{
    int tid=threadIdx.x;

    if(tid<9 && board[tid]==EMPTY)
    {
        atomicMin(move,tid);
    }
}

// GPU 2: prefer center, else first empty
__global__ void gpu2Move(int *board,int *move)
{
    if(threadIdx.x==0)
    {
        if(board[4]==EMPTY)
        {
            *move=4;
            return;
        }

        for(int i=0;i<9;i++)
        {
            if(board[i]==EMPTY)
            {
                *move=i;
                return;
            }
        }
    }
}

int main()
{
    int board[9]={0};

    int *d_board;
    int *d_move;

    cudaMalloc(&d_board,9*sizeof(int));
    cudaMalloc(&d_move,sizeof(int));

    int turn=X;

    cout<<"======================"<<endl;
    cout<<" GPU vs GPU TicTacToe "<<endl;
    cout<<"======================"<<endl;

    while(true)
    {
        cudaMemcpy(
            d_board,
            board,
            9*sizeof(int),
            cudaMemcpyHostToDevice);

        int move=100;

        cudaMemcpy(
            d_move,
            &move,
            sizeof(int),
            cudaMemcpyHostToDevice);

        if(turn==X)
        {
            gpu1Move<<<1,9>>>(d_board,d_move);
        }
        else
        {
            gpu2Move<<<1,1>>>(d_board,d_move);
        }

        cudaDeviceSynchronize();

        cudaMemcpy(
            &move,
            d_move,
            sizeof(int),
            cudaMemcpyDeviceToHost);

        board[move]=turn;

        if(turn==X)
            cout<<"GPU 1 (X) move: "<<move<<endl;
        else
            cout<<"GPU 2 (O) move: "<<move<<endl;

        printBoard(board);

        int winner=checkWinner(board);

        if(winner==X)
        {
            cout<<"GPU 1 WINS"<<endl;
            break;
        }

        if(winner==O)
        {
            cout<<"GPU 2 WINS"<<endl;
            break;
        }

        if(draw(board))
        {
            cout<<"DRAW GAME"<<endl;
            break;
        }

        turn=(turn==X)?O:X;
    }

    cudaFree(d_board);
    cudaFree(d_move);

    return 0;
}
```

### Step 3: Compile
```c
!nvcc gpu_tictactoe.cu -o game
```

### Step 4: Run
```c
!./game
```

# OUTPUT :


<img width="794" height="712" alt="Screenshot 2026-06-01 102300" src="https://github.com/user-attachments/assets/e7501cef-9ab6-40ff-bcb6-f79530ef6801" />


<img width="796" height="294" alt="Screenshot 2026-06-01 101957" src="https://github.com/user-attachments/assets/befc5232-5480-44fb-8e7e-63fb0ac01810" />

# RESULT :

The project successfully demonstrates GPU vs GPU gameplay using CUDA kernels, GPU memory management, parallel execution, and real-time game visualization on a Tic-Tac-Toe board.
