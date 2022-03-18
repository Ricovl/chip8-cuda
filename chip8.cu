/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <helper_cuda.h>
#include <iostream>


#define PROGRAM_START 0x200

typedef struct chip8_t {
    uint8_t mem[4096];

    uint16_t pc;
    uint16_t sp;
    uint16_t I;

    uint8_t v[16];

    uint8_t vram[64 * 32];
};

__device__ bool chip8_step(chip8_t *chip, uint8_t *program) {
    uint8_t x_start, y_start, height;
    uint16_t opcode = chip->mem[chip->pc] << 8 | chip->mem[chip->pc + 1];
    // uint8_t tmp = (opcode & 0xff00) >> 8;
    // opcode = ((opcode & 0x00ff) << 8) | tmp;

    printf("[%d] ", chip->pc);

    uint8_t inst = (uint8_t)((opcode & 0xf000) >> 12);
    switch (inst)
    {
    case 0x0:
        switch (opcode)
        {
        case 0x00E0:
            printf("disp_clear()\n");
            memset(chip->vram, 0, 64*32);
            break;
        case 0x00EE:
            printf("return;\n");
            break;
        default:
            printf("Unknown instruction: %04X %d\n", opcode, inst);
            break;
        }

        break;
    case 0x1:
        printf("goto %d;\n", ((opcode & 0x0fff) >> 0)); 
        chip->pc = opcode & 0x0fff;
        chip->pc -= 2;

        break;
    // case 0x2:
    //     break;
    // case 0x3:

    //     break;
    // case 0x4:

    //     break;
    // case 0x5:

    //     break;
    case 0x6:
        printf("V%d = %d;\n", (opcode & 0x0f00) >> 8, ((opcode & 0x00ff) >> 0)); 
        chip->v[(opcode & 0x0f00) >> 8] = (opcode & 0x00ff) >> 0;
        break;
    case 0x7:
        printf("V%d += %d;\n", (opcode & 0x0f00) >> 8, ((opcode & 0x00ff) >> 0)); 
        chip->v[(opcode & 0x0f00) >> 8] += (opcode & 0x00ff) >> 0;
        break;
    // case 0x8:

    //     break;
    // case 0x9:

    //     break;
    case 0xA:
        printf("I = %d;\n", (opcode & 0x0fff) >> 0);
        chip->I = opcode & 0x0fff;
        break;
    // case 0xB:

    //     break;
    // case 0xC:

    //     break;
    case 0xD:
        printf("draw(V%d, V%d, %d);\n", (opcode & 0x0f00) >> 8, (opcode & 0x00f0) >> 4, (opcode & 0x000f) >> 0); 
        printf("draw(%d, %d, %d);\n", chip->v[(opcode & 0x0f00) >> 8], chip->v[(opcode & 0x00f0) >> 4], (opcode & 0x000f) >> 0); 
        x_start = chip->v[(opcode & 0x0f00) >> 8];
        y_start = chip->v[(opcode & 0x00f0) >> 4];
        height = (opcode & 0x000f) >> 0;

        chip->v[0xF] = 0;
        for (int y = 0; y < height; y++) {
            uint8_t pixel = chip->mem[chip->I + y];

            for (int x = 0; x < 8; x++) {
                if ((pixel & (0x80 >> x)) != 0) {
                    if (chip->vram[((y_start + y) * 64) + x_start + x] == 1) {
                        chip->v[0xF] = 1;
                    }
                    chip->vram[((y_start + y) * 64) + x_start + x] ^= 1;
                    printf("xor");
                }
            }
        }

        break;
    // case 0xE:

    //     break;
    // case 0xF:

    //     break;
    default:
        printf("Unknown instruction: %04X %d\n", opcode, inst);
        break;
    }

    chip->pc += 2;
    return true;
}

__global__ void run_chip8(chip8_t *chip, uint8_t *program, int n)
{
    // const char letters[]{'x', 'y', 'z', 'w'};
    printf("test %d, %d, %d\n", blockIdx.x, threadIdx.x, (blockIdx.x * blockDim.x) + threadIdx.x);
    chip8_t *local_chip = &chip[(((blockIdx.x * blockDim.x) + threadIdx.x))];
    unsigned i = 0;

    local_chip->pc = PROGRAM_START;
    memcpy(local_chip->mem + PROGRAM_START, program, n);

    bool step = true;
    while (step && i < 30) {
        step = chip8_step(local_chip, program);

        if (local_chip->pc > n + PROGRAM_START) {
            step = false;
        }
        i++;
    }


}

// #define NUM_BLOCKS 4
// #define NUM_THREADS_PER_BLOCK 8
#define NUM_BLOCKS 2
#define NUM_THREADS_PER_BLOCK 8

int main(int argc, char **argv)
{
    const char *filename = sdkFindFilePath("IBM Logo.ch8", argv[0]);

    // find first CUDA device
    int devID = findCudaDevice(argc, (const char **)argv);

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize len = file.tellg();
    file.seekg(0, std::ios::beg);

    char *program;
    program = (char *)malloc(len);
    if (!file.read(program, len)) {
        printf("Cannot find the input text file\n. Exiting..\n");
        return EXIT_FAILURE;
    }
    file.close();
    std::cout << "Read " << len << " byte corpus from " << filename << std::endl;

    uint8_t *d_program;
    checkCudaErrors(cudaMalloc(&d_program, len));
    checkCudaErrors(cudaMemcpy(d_program, program, len, cudaMemcpyHostToDevice));

    // allocate 4kb for each emulator
    chip8_t *d_chip;
    chip8_t *h_chip = (chip8_t *)malloc(sizeof(chip8_t) * (NUM_BLOCKS * NUM_THREADS_PER_BLOCK));
    checkCudaErrors(cudaMalloc(&d_chip, sizeof(chip8_t) * (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)));


    // Try uncommenting one kernel call at a time
    run_chip8<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_chip , d_program, len);
    checkCudaErrors(cudaMemcpy(h_chip, d_chip, sizeof(chip8_t) * (NUM_BLOCKS * NUM_THREADS_PER_BLOCK), cudaMemcpyDeviceToHost));

    for (int i = 0; i < NUM_BLOCKS * NUM_THREADS_PER_BLOCK; i++) {
        printf("output for %d:\n", i);
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 64; x++) {
                printf("%d", h_chip[i].vram[(y * 64) + x]);
            }
            printf("\n");
        }
    }

    checkCudaErrors(cudaFree(d_program));
    checkCudaErrors(cudaFree(d_chip));

    return EXIT_SUCCESS;
}
