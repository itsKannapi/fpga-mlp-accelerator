`timescale 1ns / 1ps
`include "params.vh"
// ============================================================================
// layer1.v -- hidden layer: 784 int8 pixels -> 32 int8 activations
//
// 32 MACs run in parallel, one per hidden neuron. Each clock, ONE pixel is
// broadcast to all 32 MACs while a single 256-bit weight word supplies each
// MAC its own weight for that pixel. So layer 1 costs 784 accumulate cycles
// no matter how many neurons there are (up to the DSP budget).
//
//   pixel i  ----+------+------+-- ... --+
//                |      |      |         |
//   w1_t[i]  -> MAC0   MAC1   MAC2  ...  MAC31     (byte n -> MAC n)
//                |      |      |         |
//               ReLU + round + shift + clamp       (requantize to int8)
//                |      |      |         |
//              h[0]   h[1]   h[2]  ...  h[31]
//
// Interface
//   start     pulse for one cycle while idle to process one image.
//             Ignored while busy.
//   pix_addr  pixel index 0..783 this block wants next.
//   pix_data  that pixel's value, valid ONE cycle after pix_addr (the
//             timing of a synchronous block RAM read).
//   done      one-cycle pulse. h_flat is valid from then until next start.
//   h_flat    32 int8 activations, neuron n in bits [8n+7:8n].
//   acc_flat  raw 32-bit accumulators (pre-ReLU), for verification.
//
// Latency: 786 cycles from start to done (~7.9 us at 100 MHz).
//
// Pipeline (addr k is presented during the cycle after edge k):
//   edge 1     MACs load their biases
//   edge k+1   pixel k and weight word k arrive from memory
//   edge k+2   MACs accumulate pixel k          (k = 0 .. 783)
//   edge 786   requantized outputs registered, done goes high
// ============================================================================

`ifndef DATA_DIR
`define DATA_DIR "C:/dev/fpga-mlp/rtl/"
`endif

module layer1 (
    input  wire                           clk,
    input  wire                           rst,
    input  wire                           start,
    output reg                            busy,
    output reg                            done,
    output reg  [9:0]                     pix_addr,
    input  wire [7:0]                     pix_data,
    output reg  [`N_HIDDEN*8-1:0]         h_flat,
    output wire [`N_HIDDEN*`ACC_W-1:0]    acc_flat
);

    localparam N_IN  = `N_IN;
    localparam N_H   = `N_HIDDEN;
    localparam ACC_W = `ACC_W;
    localparam SHIFT = `SHIFT1;
    // Rounding constant for round-half-up: 2^(SHIFT-1). Must match
    // requantize() in quantize_export.py exactly.
    localparam RND   = (`ROUND_SHIFT && SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;

    // ------------------------------------------------------------ memories
    // Weight ROM: 784 words x 256 bits, pixel-major (see quantize_export.py).
    // Synchronous read so Vivado maps it to block RAM.
    (* rom_style = "block" *)
    reg [N_H*8-1:0] w_rom [0:N_IN-1];
    reg [N_H*8-1:0] w_word;

    // Biases are 32 constants; these become LUT logic, not a memory.
    reg [ACC_W-1:0] bias [0:N_H-1];

    initial begin
        $readmemh({`DATA_DIR, "w1_t.mem"}, w_rom);
        $readmemh({`DATA_DIR, "b1.mem"},   bias);
    end

    always @(posedge clk)
        w_word <= w_rom[pix_addr];

    // ------------------------------------------------------------ control
    localparam S_IDLE  = 2'd0,   // waiting for start
               S_RUN   = 2'd1,   // presenting addresses 0..783
               S_FLUSH = 2'd2,   // last pixel's data in flight to the MACs
               S_OUT   = 2'd3;   // accumulators final; register outputs

    reg [1:0] state;
    reg       issue;     // an address is being presented this cycle
    reg       mac_load;  // MACs take their bias at the next edge
    reg       mac_en;    // memory data is valid -> MACs accumulate

    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            issue    <= 1'b0;
            mac_load <= 1'b0;
            mac_en   <= 1'b0;
            busy     <= 1'b0;
            done     <= 1'b0;
            pix_addr <= 10'd0;
        end else begin
            done     <= 1'b0;
            mac_load <= 1'b0;
            mac_en   <= issue;   // data for the address issued last cycle

            case (state)
                S_IDLE: if (start) begin
                    busy     <= 1'b1;
                    mac_load <= 1'b1;
                    issue    <= 1'b1;
                    pix_addr <= 10'd0;
                    state    <= S_RUN;
                end

                S_RUN: if (pix_addr == N_IN - 1) begin
                    issue <= 1'b0;
                    state <= S_FLUSH;
                end else begin
                    pix_addr <= pix_addr + 10'd1;
                end

                S_FLUSH: state <= S_OUT;   // final accumulate happens this edge

                S_OUT: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

    // ------------------------------------------------------------ MAC array
    wire signed [ACC_W-1:0] acc [0:N_H-1];

    genvar n;
    generate
        for (n = 0; n < N_H; n = n + 1) begin : g_mac
            mac #(.ACC_W(ACC_W)) u_mac (
                .clk  (clk),
                .load (mac_load),
                .en   (mac_en),
                .a    (pix_data),           // same pixel to every MAC
                .b    (w_word[8*n +: 8]),   // this neuron's weight
                .init (bias[n]),
                .acc  (acc[n])
            );
            assign acc_flat[ACC_W*n +: ACC_W] = acc[n];
        end
    endgenerate

    // ------------------------------------------------------------ requantize
    // int32 accumulator -> ReLU -> (x + 2^(SHIFT-1)) >> SHIFT -> clamp 0..127
    function [7:0] requant(input signed [ACC_W-1:0] x);
        reg [ACC_W-1:0] r;
        begin
            if (x[ACC_W-1])
                requant = 8'd0;                 // ReLU: negative -> 0
            else begin
                r = (x + RND) >> SHIFT;         // x >= 0, so logical shift is exact
                requant = (r > 127) ? 8'd127 : r[7:0];
            end
        end
    endfunction

    integer j;
    always @(posedge clk)
        if (!rst && state == S_OUT)
            for (j = 0; j < N_H; j = j + 1)
                h_flat[8*j +: 8] <= requant(acc[j]);

endmodule
