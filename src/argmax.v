`timescale 1ns / 1ps
`include "params.vh"
// ============================================================================
// argmax.v -- index of the largest of N_OUT signed int32 scores
//
// Scans the scores one per clock instead of comparing all 10 at once. A
// single-cycle 10-way compare is a chain of nine 32-bit comparators and
// muxes -- easily too slow for 100 MHz -- while the whole scan costs 10
// cycles against layer 1's 786. Cheap insurance for timing closure.
//
// TIE-BREAK: the LOWEST index wins, matching numpy.argmax, which produced
// the golden predictions. That falls out of using a strict ">" while
// scanning upward from index 0: a later equal score never replaces an
// earlier one. Using ">=" would silently pick the HIGHEST index on a tie.
//
// Interface
//   start        pulse for one cycle while idle. Ignored while busy.
//   scores_flat  N_OUT signed int32 scores, class n in bits [32n+31:32n].
//                Must stay stable from start until done.
//   done         one-cycle pulse; digit and best valid from then on.
//   digit        winning class index (0..9)
//   best         its score (handy for debugging / confidence display)
//
// Latency: 10 cycles from start to done.
// ============================================================================

module argmax (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         start,
    input  wire [`N_OUT*`ACC_W-1:0]     scores_flat,
    output reg                          busy,
    output reg                          done,
    output reg  [3:0]                   digit,
    output reg  signed [`ACC_W-1:0]     best
);

    localparam N_O   = `N_OUT;
    localparam ACC_W = `ACC_W;

    reg [3:0] i;                    // index being compared this cycle
    reg [3:0] best_idx;

    wire signed [ACC_W-1:0] cand = scores_flat[ACC_W*i +: ACC_W];

    localparam S_IDLE = 2'd0, S_SCAN = 2'd1, S_OUT = 2'd2;
    reg [1:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            busy  <= 1'b0;
            done  <= 1'b0;
            i     <= 4'd0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    // score 0 is the starting champion; scan 1..N_O-1
                    best     <= $signed(scores_flat[ACC_W-1:0]);
                    best_idx <= 4'd0;
                    i        <= 4'd1;
                    busy     <= 1'b1;
                    state    <= S_SCAN;
                end

                S_SCAN: begin
                    if (cand > best) begin          // strict: lowest index wins ties
                        best     <= cand;
                        best_idx <= i;
                    end
                    if (i == N_O - 1)
                        state <= S_OUT;
                    else
                        i <= i + 4'd1;
                end

                S_OUT: begin
                    digit <= best_idx;
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
