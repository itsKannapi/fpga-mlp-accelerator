`timescale 1ns / 1ps
// ============================================================================
// mac.v -- single int8 x int8 multiply-accumulate unit
//
// The basic building block of the accelerator. Every neuron's output is
//
//     acc = bias + sum_i ( activation[i] * weight[i] )
//
// and this unit computes exactly that, one term per clock:
//
//     load = 1  ->  acc <= init            (preload the neuron's bias)
//     en   = 1  ->  acc <= acc + a * b     (accumulate one product)
//     both = 0  ->  acc holds its value
//
// load has priority over en, so starting a new neuron is always a clean
// restart regardless of what en is doing that cycle.
//
// Arithmetic contract (must match quantize_export.py bit for bit):
//   a, b  : signed int8   (activations are 0..127, weights -127..127)
//   prod  : signed int16  (int8 * int8 never overflows 16 bits)
//   acc   : signed ACC_W  (32 by default; peak observed |acc| is ~2^19)
//
// The use_dsp attribute asks Vivado to map the multiply and the add onto
// one DSP48E1 slice instead of building them out of LUTs. The Basys 3 has
// 90 of those, which is the real ceiling on how many of these we can run
// in parallel.
// ============================================================================

(* use_dsp = "yes" *)
module mac #(
    parameter ACC_W = 32
)(
    input  wire                    clk,
    input  wire                    load,
    input  wire                    en,
    input  wire signed [7:0]       a,
    input  wire signed [7:0]       b,
    input  wire signed [ACC_W-1:0] init,
    output reg  signed [ACC_W-1:0] acc
);

    // Both operands declared signed, so this is a signed multiply and the
    // 16-bit product is sign-extended correctly when added to acc.
    wire signed [15:0] prod = a * b;

    always @(posedge clk) begin
        if (load)
            acc <= init;
        else if (en)
            acc <= acc + prod;
    end

endmodule
