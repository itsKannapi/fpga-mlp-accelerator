`timescale 1ns / 1ps
`include "params.vh"
// ============================================================================
// mlp_core.v -- the complete classifier: 784 pixels in, one digit out
//
//   pixels --> layer1 --h[32]--> layer2 --scores[10]--> argmax --> digit
//              786 cyc           34 cyc                 10 cyc
//
// Each stage starts on the previous stage's done pulse. Every stage holds its
// output until its own next start, and a new image can only start once the
// whole chain is idle, so no stage's input changes while it is working.
//
// Interface
//   start        pulse for one cycle while idle to classify one image.
//                Ignored while busy.
//   pix_addr     pixel index 0..783 wanted next
//   pix_data     that pixel, valid ONE cycle after pix_addr (block-RAM timing)
//   busy         high from the cycle after start until done
//   done         one-cycle pulse; digit, scores_flat, h_flat valid from then
//                until the next start
//   digit        predicted class 0..9
//   scores_flat  the 10 raw int32 class scores (for debugging / UART readout)
//   h_flat       the 32 int8 hidden activations (for debugging)
//
// Latency: 832 cycles from start to done (8.32 us at 100 MHz).
//   786 (layer 1) + 1 handoff + 34 (layer 2) + 1 handoff + 10 (argmax)
// ============================================================================

module mlp_core (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         start,
    output wire                         busy,
    output wire                         done,
    output wire [9:0]                   pix_addr,
    input  wire [7:0]                   pix_data,
    output wire [3:0]                   digit,
    output wire [`N_OUT*`ACC_W-1:0]     scores_flat,
    output wire [`N_HIDDEN*8-1:0]       h_flat
);

    wire l1_busy, l1_done;
    wire l2_busy, l2_done;
    wire am_busy;
    wire [`N_HIDDEN*`ACC_W-1:0] acc1_unused;
    wire signed [`ACC_W-1:0]    best_unused;

    // Only start the chain when the WHOLE chain is idle. layer1 alone would
    // accept a start while layer2/argmax are still finishing the last image,
    // and a new layer-1 result would then overwrite h_flat under layer 2.
    // busy drops in the same cycle done pulses (like layer1/layer2), so a new
    // start may arrive right on done; every stage is idle by then.
    reg  busy_r;
    assign busy = busy_r & ~done;
    wire go = start & ~busy;

    always @(posedge clk) begin
        if (rst)       busy_r <= 1'b0;
        else if (go)   busy_r <= 1'b1;     // go wins if it coincides with done
        else if (done) busy_r <= 1'b0;
    end

    layer1 u_layer1 (
        .clk      (clk),
        .rst      (rst),
        .start    (go),
        .busy     (l1_busy),
        .done     (l1_done),
        .pix_addr (pix_addr),
        .pix_data (pix_data),
        .h_flat   (h_flat),
        .acc_flat (acc1_unused)
    );

    layer2 u_layer2 (
        .clk         (clk),
        .rst         (rst),
        .start       (l1_done),
        .busy        (l2_busy),
        .done        (l2_done),
        .h_flat      (h_flat),
        .scores_flat (scores_flat)
    );

    argmax u_argmax (
        .clk         (clk),
        .rst         (rst),
        .start       (l2_done),
        .scores_flat (scores_flat),
        .busy        (am_busy),
        .done        (done),
        .digit       (digit),
        .best        (best_unused)
    );

endmodule
