`timescale 1ns / 1ps
`include "params.vh"
// ============================================================================
// tb_argmax.v -- self-checking testbench for argmax.v
//
// Real MNIST scores never tie, so the tie-break rule (lowest index wins, as
// numpy.argmax does) can only be tested with made-up inputs. Three stages:
//   DIRECTED  hand-picked cases: ties, all-equal, max at either end,
//             all-negative, int32 extremes, signed-vs-unsigned trap
//   RANDOM    2,000 vectors; half drawn from a tiny range (-3..3) so ties
//             are common, half from the full int32 range
//   CONTROL   latency is exactly 10 cycles, done is a single pulse, a start
//             pulse mid-scan is ignored
//
// The reference finds the maximum first, then the first index holding it --
// a different algorithm from the hardware's running comparison.
// ============================================================================

module tb_argmax;

    localparam N_O     = `N_OUT;
    localparam ACC_W   = `ACC_W;
    localparam LATENCY = 10;
    localparam N_RAND  = 2000;

    reg                     clk = 0, rst = 1, start = 0;
    reg  [N_O*ACC_W-1:0]    scores = 0;
    wire                    busy, done;
    wire [3:0]              digit;
    wire signed [ACC_W-1:0] best;

    argmax dut (
        .clk(clk), .rst(rst), .start(start), .scores_flat(scores),
        .busy(busy), .done(done), .digit(digit), .best(best)
    );

    always #5 clk = ~clk;

    integer n_pass = 0, n_fail = 0, f_dir = 0, f_rand = 0, f_ctl = 0, n_ties = 0;
    integer i, t, lat, extra, exp_idx;
    reg signed [ACC_W-1:0] mx, v;

    // ---------------------------------------------------- reference model
    task reference(output integer idx);
        integer k;
        begin
            mx = $signed(scores[ACC_W-1:0]);
            for (k = 1; k < N_O; k = k + 1)
                if ($signed(scores[ACC_W*k +: ACC_W]) > mx) mx = $signed(scores[ACC_W*k +: ACC_W]);
            idx = -1;
            for (k = N_O - 1; k >= 0; k = k - 1)          // last write = lowest index
                if ($signed(scores[ACC_W*k +: ACC_W]) == mx) idx = k;
        end
    endtask

    task set_score(input integer k, input signed [ACC_W-1:0] val);
        scores[ACC_W*k +: ACC_W] = val;
    endtask

    // ---------------------------------------------------- run one vector
    // stray = 1 -> also pulse start in the middle of the scan (must be ignored)
    task run(input integer stray, input [8*24-1:0] what, input integer is_directed);
        begin
            reference(exp_idx);
            start = 1; @(posedge clk); #1; start = 0;
            lat = 0;
            while (!done && lat < 100) begin
                if (stray && lat == 4) start = 1;
                @(posedge clk); #1;
                start = 0;
                lat = lat + 1;
            end
            extra = 0;
            repeat (2) begin @(posedge clk); #1; if (done || busy) extra = extra + 1; end

            if (lat != LATENCY || extra != 0) begin
                f_ctl = f_ctl + 1;
                $display("FAIL  %0s control: latency=%0d (want %0d) extra=%0d", what, lat, LATENCY, extra);
            end
            if (digit !== exp_idx[3:0] || best !== mx) begin
                if (is_directed) f_dir = f_dir + 1; else f_rand = f_rand + 1;
                if (f_dir + f_rand <= 10)
                    $display("FAIL  %0s: expected index %0d (score %0d), got %0d (score %0d)",
                             what, exp_idx, mx, digit, best);
            end else n_pass = n_pass + 1;
        end
    endtask

    task fill_all(input signed [ACC_W-1:0] val);
        integer k;
        for (k = 0; k < N_O; k = k + 1) set_score(k, val);
    endtask

    initial begin
        $display("==== tb_argmax ====");
        repeat (3) @(posedge clk);
        #1 rst = 0;
        @(posedge clk); #1;

        // ================================================= DIRECTED
        fill_all(0);                                          run(1, "all zero -> 0", 1);
        fill_all(0);    set_score(9, 1);                      run(0, "max at last index", 1);
        fill_all(0);    set_score(0, 1);                      run(0, "max at first index", 1);
        fill_all(0);    set_score(3, 50); set_score(7, 50);   run(0, "tie 3 vs 7 -> 3", 1);
        fill_all(0);    set_score(0, 9);  set_score(9, 9);    run(0, "tie 0 vs 9 -> 0", 1);
        fill_all(-5);                                         run(0, "all equal negative", 1);
        fill_all(-1000); set_score(6, -1);                    run(0, "all negative", 1);
        fill_all(32'sh8000_0000); set_score(9, 32'sh7FFF_FFFF); run(0, "int32 extremes", 1);
        fill_all(-7);   set_score(2, -1); set_score(5, 1);    run(0, "signed trap: -1 vs 1", 1);
        fill_all(32'sh8000_0000);                             run(0, "all int32 min", 1);
        for (i = 0; i < N_O; i = i + 1) set_score(i, 100 - i); run(0, "descending", 1);
        for (i = 0; i < N_O; i = i + 1) set_score(i, i);       run(0, "ascending", 1);
        $display("directed : %0d fail", f_dir);

        // ================================================= RANDOM
        for (t = 0; t < N_RAND; t = t + 1) begin
            for (i = 0; i < N_O; i = i + 1) begin
                if (t % 2 == 0) v = ($random % 4);            // -3..3: frequent ties
                else            v = $random;                  // full int32 range
                set_score(i, v);
            end
            reference(exp_idx);
            v = 0;
            for (i = 0; i < N_O; i = i + 1)
                if ($signed(scores[ACC_W*i +: ACC_W]) == mx) v = v + 1;
            if (v > 1) n_ties = n_ties + 1;
            run(0, "random", 0);
        end
        $display("random   : %0d vectors (%0d with a tied maximum), %0d fail", N_RAND, n_ties, f_rand);
        $display("control  : %0d runs, %0d fail  (latency %0d cycles)", 12 + N_RAND, f_ctl, LATENCY);

        n_fail = f_dir + f_rand + f_ctl;
        $display("================================");
        if (n_fail == 0)
            $display("ALL TESTS PASSED  (%0d checks)", n_pass + 12 + N_RAND);
        else
            $display("%0d CHECKS FAILED", n_fail);
        $display("================================");
        $finish;
    end

endmodule
